"""Contains all ensemble models."""
from copy import deepcopy
import numpy as np
import logging
from datetime import datetime

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
        self.name = name
        if not isinstance(template_network, BaseNetwork):
            raise ValueError(
                "template_network type must inherit from BaseNetwork.")

        self.template_network = deepcopy(template_network)

        if n_snapshots <= 0:
            raise ValueError("n_snapshots must be >=1.")
        self.M = n_snapshots
        self.snapshot_networks = {}

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
            # Save instance of snapshot in a dictionary
            snaps_name = "{}_{}".format(self.name, index)
            self.snapshot_networks[snaps_name] = \
                deepcopy(self.template_network)

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
        prediction_collection = []
        for key, network in self.snapshot_networks.items():
            logger.info("Getting output from {}".format(key))
            prediction_collection.append(
                network.forward_pass(
                    data_loader=data_loader,
                    convert_to_class=False))
        prediction_collection = np.array(prediction_collection)
        raw_prediction = np.mean(
            prediction_collection, axis=0, dtype='float32')
        if convert_to_class:
            # TODO: be able to use Metrics.get_class outside
            return Metrics.get_class(None, raw_prediction)
        else:
            return raw_prediction

    # TODO: Fix bc it writes in the same folder several models
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
        if save_path is None:
            save_path = r"saved_models/{}_{}/".format(
                self.name, datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
        for key, network in self.snapshot_networks.items():
            logger.info("Saving network {}".format(key))
            network.save_model(save_path=save_path)

    # TODO: Fix to load the correct models
    # @classmethod
    # def load_model(cls, load_path):
    #     """Load up ensembled models given a folder location."""
    #     networks = []
    #     for model_file in sorted(os.listdir(load_path)):
    #         if model_file.endswith('.network'):
    #             file = os.path.join(load_path, model_file)
    #             networks += [BaseNetwork.load_model(file)]

    #     return None
