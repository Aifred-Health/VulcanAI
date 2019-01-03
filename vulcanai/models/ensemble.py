"""Contains all ensemble models."""
from copy import deepcopy
import logging
from datetime import datetime
import pickle
import os

import torch
from torch import nn
from torch.optim.lr_scheduler import CosineAnnealingLR

from .basenetwork import BaseNetwork

logger = logging.getLogger(__name__)


class SnapshotNet(BaseNetwork):
    """
    Initialize snapshot ensemble given a template network.

    A wrapper class for any Network inheriting from BaseNetwork to
    train the template network using Snapshot Ensembling.

    Parameters:
        name : str
            String of snapshot ensemble name.
        template_network : BaseNetwork
            Network object which you want to ensemble.
        n_snapshots : int
            Number of snapshots in ensemble.

    Returns:
        network : SnapshotNet

    """

    def __init__(self, name, template_network, n_snapshots=3):
        """Use Network to build model snapshots."""
        # TODO: Should these be defaulted to the values of template_network?
        super(SnapshotNet, self).__init__(
            name=name,
            config=None,  # template_network._config
            in_dim=template_network.in_dim,
            save_path=None,  # template_network.save_path
            input_networks=None,  # template_network.input_networks
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
        self.network = nn.ModuleList()
        self.out_dim = self.template_network.out_dim
        if n_snapshots <= 0:
            raise ValueError("n_snapshots must be >=1.")
        self.n_snapshots = n_snapshots

    def fit(self, train_loader, val_loader, epochs,
            retain_graph=None, valid_interv=4, plot=False):
        """
        Train each model for T/M epochs and controls network learning rate.

        Collects each model in a class variable self.network

        Parameters:
            train_loader : DataLoader
                Input data and targets to train against
            val_loader : DataLoader
                Input data and targets to validate against
            epochs : int
                Total number of epochs (evenly distributed between snapshots)

        Returns:
            None

        """
        # There must be at least one train epoch for each snapshot
        if epochs < self.n_snapshots:
            logger.warn(
                'Number of epochs to small for number of Snapshots. '
                'Setting epochs to {}.'.format(self.n_snapshots))
            epochs = self.n_snapshots

        T = epochs
        # How many epochs each singular network should train for
        network_epochs = T // self.n_snapshots

        # Temporary but check if it first has an optimizer,
        # if not it will make one
        if self.template_network.optim is None:
            self.template_network._init_trainer()

        self.template_network.lr_scheduler = CosineAnnealingLR(
            optimizer=self.template_network.optim,
            T_max=network_epochs
        )

        for index in range(self.n_snapshots):
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
            self.network.append(temp_network)

    def _update_network_name_stack(self, network, append_str):
        """
        Given a network, append a string to the name of all networks in stack.

        Recursively traverse each input network to update names
        with the appended string.

        Parameters:
            network : BaseNetwork
                Network stack to update names of with new append_str.
            append_str : int, str
                The characters to append at the end of BaseNetwork stack of names.

        """
        if network.input_networks:
            for in_net in network.input_networks.values():
                self._update_network_name_stack(in_net, append_str)
        network.name = "{}_{}".format(network.name, append_str)

    def forward(self, inputs, **kwargs):
        """
        Snapshot forward function.

        Collect outputs of all internal networks and average outputs.

        Parameters:
            inputs : torch.Tensor
                Input tensor to pass through self.

        Returns:
            output : torch.Tensor

        """
        if len(self.network) == 0:
            raise ValueError("SnapshotNet needs to be trained.")

        pred_collector = []
        for net in self.network:
            pred_collector.append(net(inputs))
        # Stack outputs along a new 0 dimension to be averaged
        pred_collector = torch.stack(pred_collector)

        return torch.mean(input=pred_collector, dim=0)

    def save_model(self, save_path=None):
        """
        Save all ensembled network in a folder with ensemble name.

        Parameters:
            save_path : str
                The folder path to save models in.

        Returns:
            None

        """
        if not save_path:
            save_path = r"saved_models/"

        if not save_path.endswith("/"):
            save_path = save_path + "/"

        save_path = save_path + "{}_{}/".format(
            self.name, datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
        logger.info("No save path provided, saving to {}".format(save_path))

        for network in self.network:
            logger.info("Saving network {}".format(network.name))
            network.save_model(save_path=save_path)

        if not os.path.exists(save_path):
            os.makedirs(save_path)

        model_save_path = save_path + "model.pkl"
        self.save_path = save_path
        pickle.dump(self, open(model_save_path, "wb"), 2)
