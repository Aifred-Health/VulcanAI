"""Contains all ensemble models."""
from copy import deepcopy
import numpy as np
import logging
from datetime import datetime
import pickle

from torch import nn
from torch.optim.lr_scheduler import CosineAnnealingLR

from .basenetwork import BaseNetwork
from .metrics import Metrics

logger = logging.getLogger(__name__)


class SnapshotNet(object):
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
        # super(SnapshotNet, self).__init__(
        #     name, template_network.in_dim, template_network._config,
        #     template_network.save_path, template_network._input_network,
        #     template_network._num_classes,
        #     template_network.network[0]._activation,
        #     None, template_network._optim_spec,
        #     template_network.lr_scheduler, template_network.early_stopping,
        #     template_network._criter_spec)

        self.name = name
        if not isinstance(template_network, BaseNetwork):
            raise ValueError(
                "template_network type must inherit from BaseNetwork.")

        self.template_network = deepcopy(template_network)
        self._num_classes = template_network._num_classes
        self.metrics = Metrics(self._num_classes)

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

    def forward_pass(self, data_loader, convert_to_class=False):
        """
        Run forward on all snapshot networks, collect and return results.

        Parameters
        ----------
        data_loader : DataLoader
            Input data to pass through networks
        convert_to_class : boolean
            Whether to convert class probabilities to classes

        Returns
        -------
        prediction : numpy.ndarray
            For raw probabilities the shape is [batch, num_classes].
            For class probabilities the shape is [batch]

        """
        if len(self.snapshot_networks) == 0:
            raise ValueError("SnapshotNet must be trained first.")
        prediction_collection = []
        for network in self.snapshot_networks:
            logger.info("Getting output from {}".format(network.name))
            prediction_collection.append(
                network.forward_pass(
                    data_loader=data_loader,
                    convert_to_class=False))
        prediction_collection = np.array(prediction_collection)
        raw_prediction = np.mean(
            prediction_collection, axis=0, dtype='float32')
        if convert_to_class:
            return self.metrics.get_class(raw_prediction)
        else:
            return raw_prediction

    def run_test(self, data_loader, figure_path=None, plot=False):
        """Will conduct the test suite to determine ensemble strength."""
        return self.metrics.run_test(
            network=self,
            data_loader=data_loader,
            figure_path=figure_path,
            plot=plot)

    def __getstate__(self):
        """Remove Snapshot networks to only save the filename locations."""
        snapshot_dict = dict(self.__dict__)
        del snapshot_dict['template_network']
        del snapshot_dict['snapshot_networks']
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
