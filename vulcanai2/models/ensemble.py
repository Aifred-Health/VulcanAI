"""Contains all ensemble models."""
from copy import deepcopy
import numpy as np
import logging
from datetime import datetime
import pickle

import torch
from torch import nn
from torch.optim.lr_scheduler import CosineAnnealingLR

from .basenetwork import BaseNetwork

logger = logging.getLogger(__name__)


class SnapshotNet(BaseNetwork):
    """
    Initialize snapshot ensemble given a template network.

    Parameters
    ----------
    name : str
        String of snapshot ensemble name.
    template_network : BaseNetwork
        Network object which you want to ensemble.
    n_snapshots : int
        Number of snapshots in ensemble.

    Returns
    -------
    network : SnapshotNet

    """

    def __init__(self, name, template_network, n_snapshots=3):
        """Use Network to build model snapshots."""
        # For inheriting from BaseNetwork
        super(SnapshotNet, self).__init__(
            name=name,
            dimensions=template_network.in_dim,
            config=None,  # template_network._config
            save_path=None,  # template_network.save_path
            input_network=None,  # template_network._input_network
            num_classes=template_network._num_classes,
            activation=None,  # template_network.network[0]._activation
            pred_activation=None,  # pred_activation
            optim_spec=None,  # template_network._optim_spec
            lr_scheduler=None,  # template_network.lr_scheduler
            early_stopping=None,  # template_network.early_stopping
            criter_spec=None  # template_network._criter_spec
        )

        if not isinstance(template_network, BaseNetwork):
            raise ValueError(
                "template_network type must inherit from BaseNetwork.")

        self.template_network = deepcopy(template_network)

        if n_snapshots <= 0:
            raise ValueError("n_snapshots must be >=1.")
        self.M = n_snapshots
        self.snapshot_networks = nn.ModuleList()

    def fit(self, train_loader, val_loader, epochs,
            retain_graph=None, valid_interv=4, plot=False):
        """
        Train each model for T/M epochs and controls network learning rate.

        Collects each model in a class variable self.snapshot_networks

        Parameters
        ----------
        train_loader : DataLoader
            Input data and targets to train against
        val_loader : DataLoader
            Input data and targets to validate against
        n_epochs : int
            Total number of epochs (evenly distributed between snapshots)

        Returns
        -------
        None

        """
        # There must be at least one train epoch for each snapshot
        if epochs < self.M:
            logger.warn(
                'Number of epochs to small for number of Snapshots. '
                'Setting epochs to {}.'.format(self.M))
            epochs = self.M

        T = epochs
        # How many epochs each singular network should train for
        network_epochs = T // self.M

        # Temporary but check if it first has an optimizer,
        # if not it will make one
        if self.template_network.optim is None:
            self.template_network._init_trainer()

        self.template_network.lr_scheduler = CosineAnnealingLR(
            optimizer=self.template_network.optim,
            T_max=network_epochs
        )

        for index in range(self.M):
            self.template_network.fit(
                train_loader=train_loader,
                val_loader=val_loader,
                epochs=network_epochs,
                valid_interv=valid_interv,
                plot=plot
            )
            # Save instance of snapshot in a nn.ModuleList
            temp_network = deepcopy(self.template_network)
            self._update_network_name_stack(
                network=temp_network,
                append_str=index)
            self.snapshot_networks.append(temp_network)

    def _update_network_name_stack(self, network, append_str):
        """
        Given a network, append a string to the name of all networks in stack.

        Recursively traverse each input network to update names
        with the appended string.

        Parameters
        ----------
        network : BaseNetwork
            Network stack to update names of with new append_str.
        append_str : int, str
            The characters to append at the end of BaseNetwork stack of names.

        """
        if network._input_network is not None:
            self._update_network_name_stack(network._input_network, append_str)
        network.name = "{}_{}".format(network.name, append_str)

    def forward(self, x):
        """
        Snapshot forward function.

        Collect outputs of all internal networks and average outputs.

        """
        if len(self.snapshot_networks) == 0:
            raise ValueError("SnapshotNet must be trained first.")
        pred_collector = []
        for net in self.snapshot_networks:
            pred_collector.append(net(x))
        pred_collector = torch.stack(pred_collector)
        return torch.mean(input=pred_collector, dim=0)

    def __getstate__(self):
        """Remove Snapshot networks to only save the filename locations."""
        snapshot_dict = dict(self.__dict__)
        del snapshot_dict['_modules']['template_network']
        del snapshot_dict['_modules']['snapshot_networks']
        return snapshot_dict

    def save_model(self, save_path=None):
        """
        Save all ensembled snapshot_networks in a folder with ensemble name.

        Parameters
        ----------
        save_path : str
            The folder path to save models in.

        Returns
        -------
        None

        """
        self.ensemble_file_paths = []
        if save_path is None:
            save_path = r"saved_models/{}_{}/".format(
                self.name, datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
        if not save_path.endswith("/"):
            save_path = save_path + "/"
        for network in self.snapshot_networks:
            logger.info("Saving network {}".format(network.name))
            network.save_model(save_path=save_path)
            self.ensemble_file_paths.append(network.save_path)
        model_file_path = save_path + "snapshot_model.pkl"
        pickle.dump(self, open(model_file_path, "wb"), 2)

    # TODO: Save all in self including the snapshot_networks?
    @classmethod
    def load_model(cls, load_path):
        """
        Load up ensembled models given a folder location.

        Will finddinfuse the snapshot_model.pkl

        Parameters
        ----------
        load_path : str
            Snapshot folder location containing the snapshot_model.pkl.

        Returns
        -------
        snapshot_network : SnapshotNet
            Returns the ensemble network to the same state
            as it was when it was saved.

        """
        # TODO: does this break windows?? no idea.
        if not load_path.endswith("/"):
            load_path = load_path + "/"
        model_file_path = load_path + "snapshot_model.pkl"
        snap_skeleton = pickle.load(open(model_file_path, 'rb'))
        networks = nn.ModuleList()
        for network_file in snap_skeleton.ensemble_file_paths:
            net = BaseNetwork.load_model(network_file)
            networks.append(net)
        # Generate the ensemble
        snap_skeleton.snapshot_networks = networks
        # Reinstantiate the most recently trained model as the template
        snap_skeleton.template_network = networks[-1]
        return snap_skeleton
