
from copy import deepcopy
import numpy as np

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR

from .basenetwork import BaseNetwork
from .metrics import Metrics

class SnapshotNet(object):
    """Uses Network to build model snapshots."""
    def __init__(self, name, template_network, n_snapshots=3):
        """
        Initialize snapshot ensemble given a network.

        :param name: string of snapshot ensemble name
        :param template_network: Network object which you want to ensemble
        :param n_snapshots: number of snapshots in ensemble
        :param n_epochs: total number of epochs
        """
        self.name = name
        if not isinstance(template_network, BaseNetwork):
            raise ValueError("template_network type must inherit from BaseNetwork.")

        self.template_network = deepcopy(template_network)

        if n_snapshots <= 0:
            raise ValueError("n_snapshots must be >=1.")
        self.M = n_snapshots
        self.snapshot_networks = {}

    def fit(self, train_loader, val_loader, epochs,
            retain_graph=None, valid_interv=4, plot=False):
        """
        Train each model for T/M epochs and sets new network learning rate.
        Collects each model in a class variable self.snapshot_networks
        """
        
        # There must be at least one train epoch for each snapshot
        if epochs < self.M:
            print('Number of epochs to small for number of Snapshots. '
                  'Setting epochs to {}.'.format(self.M))
            epochs = self.M

        T = epochs
        # How many epochs each singular network should train for
        network_epochs = T  // self.M
        
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
                epochs = network_epochs,
                valid_interv=valid_interv,
                plot=plot
            )
            # Save instance of snapshot in a dictionary
            snap_name = "{}_{}".format(self.name, index)
            self.snapshot_networks[snap_name] = deepcopy(self.template_network)

    def forward_pass(self, data_loader, convert_to_class=False):
        """
        Get output of ensemble.
        :param data_loader: Numpy matrix to make the predictions on
        :param convert_to_class: return class predictions from ensemble
        """
        prediction_collection = []
        for net_key in self.snapshot_networks.keys():
            print("Getting output from {}".format(net_key))
            prediction_collection.append(
                self.snapshot_networks[net_key].forward_pass(
                    data_loader=data_loader,
                    convert_to_class=False))
        prediction_collection = np.array(prediction_collection)
        raw_prediction = np.mean(prediction_collection, axis=0,
                                 dtype='float32')
        if convert_to_class:
            # TODO: be able to use Metrics.get_class outside
            return Metrics.get_class(None, raw_prediction)
        else:
            return raw_prediction

    def save_model(self, save_path='models'):
        """Save all ensembled snapshot_networks in a folder with ensemble name."""
        ensemble_path = "{}{}".format(self.timestamp, self.name)
        new_save_path = os.path.join(save_path, ensemble_path)
        if not os.path.exists(new_save_path):
            print ('Creating {} folder'.format(new_save_path))
            os.makedirs(new_save_path)

        for model in self.snapshot_networks:
            model.save_model(save_path=new_save_path)

        self.save_ensemble_metadata(new_save_path)

    @classmethod
    def load_model(cls, ensemble_path):
        """Load up ensembled models given a folder location."""
        json_file = "{}_metadata.json".format(
            os.path.join(ensemble_path, os.path.basename(ensemble_path))
        )
        with open(json_file, 'r') as file:
            config = json.load(file)

        networks = []
        for model_file in sorted(os.listdir(ensemble_path)):
            if model_file.endswith('.network'):
                file = os.path.join(ensemble_path, model_file)
                networks += [Network.load_model(file)]

        snap = SnapshotNet(
            name='snap1',
            template_network=networks[0],
            n_snapshots=config[ensemble_path]['n_snapshots']
        )
        snap.snapshot_networks = networks
        return snap

    def save_ensemble_metadata(self, file_path):
        """
        Will save ensemble configuration.
        Args:
            file_path: the npz file path without the npz
        """
        config = {
            "{}".format(file_path): {
                "n_snapshots": self.M,
                "init_learning_rate": self.template_network.init_learning_rate,
                "networks": [{n.name: n.save_name} for n in self.networks]
            }
        }

        json_file = "{}_metadata.json".format(
            os.path.join(file_path, os.path.basename(file_path))
        )
        print ('Saving metadata to {}'.format(json_file))
        with open(json_file, 'w') as file:
            json.dump(config, file)

    def save_record(self, save_path='records'):
        """Save individual training curves for all networks."""
        ensemble_path = "{}{}".format(self.timestamp, self.name)
        new_save_path = os.path.join(save_path, ensemble_path)
        if not os.path.exists(new_save_path):
            print ('Creating {} folder'.format(new_save_path))
            os.makedirs(new_save_path)

        for model in self.networks:
            model.save_record(save_path=new_save_path)