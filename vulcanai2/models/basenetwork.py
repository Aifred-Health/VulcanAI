# -*- coding: utf-8 -*-
"""Defines the basenetwork class"""
# Core imports
import abc
import torch
from torch.autograd import Variable
from torch import nn
import sys

# Vulcan imports
from .layers import *
from .utils import network_summary
from .metrics import Metrics
from ..plotters.visualization import display_record

# Generic imports
import pydash as pdash
from tqdm import tqdm, trange
from datetime import datetime
import logging
import os
import pickle
import time
from collections import OrderedDict as odict
import numpy as np

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

import warnings

warnings.filterwarnings("ignore")

sns.set(style='dark')
logger = logging.getLogger(__name__)


class BaseNetwork(nn.Module):
    """
    Defines the BaseNetwork object.

    Parameters
    ----------
    name : str
        The name of the network. Used when saving the file.
    config : dict
        The configuration of the network module, as a dict.
    in_dim : tuple
        The input dimensions of the network. Not required to specify when the
        network has input_networks.
    save_path : str
        The name of the file to which you would like to save this network.
    input_networks : list of BaseNetwork
        A network object provided as input.
    num_classes : int or None
        The number of classes to predict.
    activation : torch.nn.Module
        The desired activation function for use in the network.
    pred_activation : torch.nn.Module
        The desired activation function for use in the prediction layer.
    optim_spec : dict
        A dictionary of parameters for the desired optimizer.
    lr_scheduler : torch.optim.lr_scheduler
        A callable torch.optim.lr_scheduler
    early_stopping : str or None
        So far just 'best_validation_error' is implemented.
    criter_spec : dict
        criterion specification with name and all its parameters.

    Returns
    -------
    network : BaseNetwork

    """

    def __init__(self, name, config, in_dim=None, save_path=None,
                 input_networks=None, num_classes=None,
                 activation=nn.ReLU(), pred_activation=None,
                 optim_spec={'name': 'Adam', 'lr': 0.001},
                 lr_scheduler=None, early_stopping=None,
                 criter_spec=nn.CrossEntropyLoss()):
        """Define, initialize, and build the BaseNetwork."""
        super(BaseNetwork, self).__init__()

        self._name = name
        if in_dim is not None:
            if isinstance(in_dim, int):
                self.in_dim = tuple([in_dim])
            else:
                self.in_dim = in_dim
        else:
            if input_networks is None:
                raise ValueError("BaseNetwork must have either in_dim or \
                                input_networks")

        self._config = config
        self._save_path = save_path

        # Turn into list if not list
        if input_networks is not None and \
            not isinstance(input_networks, list):
            input_networks = [input_networks]

        if input_networks is not None and \
            not isinstance(input_networks, nn.ModuleList):
            self.input_networks = nn.ModuleList(input_networks)
        else:
            self.input_networks = input_networks

        self._num_classes = num_classes

        self._optim_spec = optim_spec
        self._lr_scheduler = lr_scheduler
        self._early_stopping = early_stopping
        self._criter_spec = criter_spec

        if self._num_classes:
            self.metrics = Metrics(self._num_classes)

        self.optim = None
        self.criterion = None
        self.epoch = 0

        self.record = dict(
            epoch=[],
            train_error=[],
            train_accuracy=[],
            validation_error=[],
            validation_accuracy=[]
        )

        # Creates the Network, and re-writes the self.in_dim
        self._create_network(
            activation=activation,
            pred_activation=pred_activation)

        # Compute self.out_dim of the network
        self.out_dim = self._compute_out_dim()

    @abc.abstractmethod
    def _merge_input_network_outputs(self, inputs):
        """Abstract method used to define how to handle multi-inputs."""
        pass

    def forward(self, inputs, **kwargs):
        """
        Perform a forward pass through the modules.

        If the network is defined with `num_classes` then it contains a
        classification layer/network tail. The inputs will be passed
        through the networks and then through the classifier.
        If not, the input is passed through the network and
        returned without passing through a classification layer.

        Parameters
        ----------
        inputs : list(torch.Tensor)
            The inputs to pass throught the network.

        Returns
        -------
        output : torch.Tensor

        """

        if not isinstance(inputs, list):
            inputs = [inputs]

        if self.input_networks is not None:
            net_outs = []
            for net, x in zip(self.input_networks, inputs):
                net_outs.append(net(x))
            output = self._merge_input_network_outputs(net_outs)
        else:
            output = torch.cat(inputs, dim=1)

        network_output = self.network(output)

        if self._num_classes:
            class_output = self.network_tail(network_output)
            return class_output
        else:
            return network_output

    def _compute_out_dim(self):
        if self.network is not None:
            out = self.network(torch.ones([1, *self.in_dim]))
            if self._num_classes:
                out = self.network_tail(out)
            return tuple(out.shape[1:])
        else:
            return None

    @property
    def name(self):
        """
        Return the name.

        Returns
        -------
        name : string
            The name of the network.
        """
        return self._name

    @name.setter
    def name(self, value):
        self._name = value

    @property
    def save_path(self):
        """
        Returns the save path
        :return: the save path of the network
        """
        return self._save_path

    @save_path.setter
    def save_path(self, value):
        if not value:
            self._save_path = "{}_{date:%Y-%m-%d_%H:%M:%S}/".format(
                self.name, date=datetime.now())
        else:
            self._save_path = value

    @property
    def lr_scheduler(self):
        """
        Return the network lr_scheduler.

        Returns
        -------
        lr_scheduler : torch.optim.lr_scheduler

        """
        return self._lr_scheduler

    @lr_scheduler.setter
    def lr_scheduler(self, value):
        self._lr_scheduler = value

    @property
    def early_stopping(self):
        """
        Return the stopping rule.

        Returns
        -------
        stopping_rule : str
            The stoping rule

        """
        return self._early_stopping

    @early_stopping.setter
    def early_stopping(self, value):
        self._early_stopping = value

    @property
    def criter_spec(self):
        """
        Return the criterion specification.

        Returns
        -------
        _criter_spec : dict
            The criterion specification.

        """
        return self._criter_spec

    @criter_spec.setter
    def criter_spec(self, value):
        self._criter_spec = value

    def get_layers(self):
        """
        Returns an ordered dict of all modules in this network (layers).

        Returns
        -------
        layers : OrderedDict()

        """
        return self._modules

    def get_weights(self):
        """
        Return a dictionary containing a whole state of the module

        Returns
        -------
        weights : dict
            A dictionary containing a whole state of the module.
        """
        return self.state_dict()

    @abc.abstractmethod
    def _create_network(self, **kwargs):
        """
        Define the network. Abstract method that needs to be overridden.

        Returns
        -------
        None

        """
        self.network = None
        pass

    def freeze(self, apply_inputs=False):
        """
        Freeze network weights so training doesn't modify them.

        Parameters
        ----------
        apply_inputs : boolean
            Whether to freeze all input networks recursively

        Returns
        -------
        None

        """
        self._toggle_freeze(freeze=True, apply_inputs=apply_inputs)

    def unfreeze(self, apply_inputs=False):
        """
        Unfreeze network weights so training does modify them.

        Parameters
        ----------
        apply_inputs : boolean
            Whether to unfreeze all input networks recursively

        Returns
        -------
        None

        """
        self._toggle_freeze(freeze=False, apply_inputs=apply_inputs)

    def _toggle_freeze(self, freeze, apply_inputs):
        # Freeze core network parameters
        for params in self.network.parameters():
            # If freeze is True, set requires_grad to False
            # If freeze is False, set requires_grad to True
            params.requires_grad = not freeze
        # Freeze prediction layer parameters
        if 'network_tail' in self._modules:
            for params in self.network_tail.parameters():
                # If freeze is True, set requires_grad to False
                # If freeze is False, set requires_grad to True
                params.requires_grad = not freeze
        # Recursively toggle freeze on
        if apply_inputs and self.input_networks is not None:
            for network in self.input_networks:
                network._toggle_freeze(
                    freeze=freeze,
                    apply_inputs=apply_inputs)

    def _init_optimizer(self, optim_spec):
        optim_class = getattr(torch.optim, optim_spec["name"])
        optim_spec = pdash.omit(optim_spec, "name")
        return optim_class(self.parameters(), **optim_spec)

    @staticmethod
    def _init_criterion(criterion_spec):
        return criterion_spec

    def _init_trainer(self):
        self.optim = self._init_optimizer(self._optim_spec)
        # TODO: Use logger to describe if the optimizer is changed.
        self.criterion = self._init_criterion(self._criter_spec)

    def fit(self, train_loader, val_loader, epochs,
            retain_graph=None, valid_interv=4, plot=False):
        """
        Trains the network on the provided data.

        Parameters
        ----------
        train_loader : DataLoader
            The DataLoader object containing the training data.
        val_loader : DataLoader
            The DataLoader object containing the validation data.
        epochs : int
            The number of epochs to train for.
        retain_graph : {None, True, False}
            Whether retain_graph will be true when .backwards is called.
        valid_interv : int
            Specifies the period of epochs before validation calculation.

        Returns
        -------
        None

        """

        # In case there is already one, don't overwrite it.
        # Important for not removing the ref from a lr scheduler
        if self.optim is None:
            self._init_trainer()

        try:
            if plot:
                fig_number = plt.gcf().number + 1 if plt.fignum_exists(1) else 1
                plt.show()

            for epoch in trange(0, epochs, desc='Epoch: ', ncols=80):

                train_loss, train_acc = self._train_epoch(train_loader,
                                                          retain_graph)
                if self.lr_scheduler is not None:
                    self.lr_scheduler.step(epoch=epoch)

                valid_loss = valid_acc = np.nan
                if epoch % valid_interv == 0:
                    valid_loss, valid_acc = self._validate(val_loader)

                tqdm.write("\n Epoch {}:\n"
                           "Train Loss: {:.6f} | Test Loss: {:.6f} |"
                           "Train Acc: {:.4f} | Test Acc: {:.4f}".format(
                    self.epoch,
                    train_loss,
                    valid_loss,
                    train_acc,
                    valid_acc
                ))

                self.record['epoch'].append(self.epoch)
                self.record['train_error'].append(train_loss)
                self.record['train_accuracy'].append(train_acc)
                self.record['validation_error'].append(valid_loss)
                self.record['validation_accuracy'].append(valid_acc)

                if plot is True:
                    plt.ion()
                    plt.figure(fig_number)
                    display_record(record=self.record)

                self.epoch += 1

        except KeyboardInterrupt:
            logger.warning(
                "\n\n**********KeyboardInterrupt: "
                "Training stopped prematurely.**********\n\n")

    def _train_epoch(self, train_loader, retain_graph):
        """
        Trains the network for 1 epoch.

        Parameters
        ----------
        train_loader : DataLoader
            The DataLoader object containing the dataset to train on.

        Returns
        -------
        (train_loss, train_accuracy) : (float, float)
            Returns the train loss and accuracy.

        """
        self.train()  # Set model to training mode

        train_loss_accumulator = 0.0
        train_accuracy_accumulator = 0.0
        pbar = trange(len(train_loader.dataset), desc='Training.. ')


        for batch_idx, t in enumerate(train_loader):
            # import pudb; pu.db
            # for idx, d in enumerate(data):
            #     data[idx] = Variable(d, requires_grad=True)
            # targets = Variable(targets)

            data, targets = t

            if torch.cuda.is_available():
                for idx, d in enumerate(data):
                    data[idx] = d.cuda()
                targets = targets.cuda()
                self.cuda()

            # Forward + Backward + Optimize
            # import pudb; pu.db
            predictions = self(data)

            train_loss = self.criterion(predictions, targets)
            train_loss_accumulator += train_loss.item()

            self.optim.zero_grad()
            train_loss.backward(retain_graph=retain_graph)
            self.optim.step()

            batch_len = len(data[0])

            if batch_idx % 10 == 0:
                # Update tqdm bar
                if ((batch_idx + 10) * batch_len) <= len(train_loader.dataset):
                    pbar.update(10 * batch_len)
                else:
                    pbar.update(len(train_loader.dataset) -
                        int(batch_idx * batch_len))

            train_accuracy_accumulator += self.metrics.get_score(predictions,
                                                                 targets)

        pbar.close()

        train_loss = train_loss_accumulator * \
            train_loader.batch_size / len(train_loader.dataset)
        train_accuracy = train_accuracy_accumulator * \
            train_loader.batch_size / len(train_loader.dataset)

        return train_loss, train_accuracy

    def _validate(self, val_loader):
        """
        Validate the network on the validation data.

        Parameters
        ----------
        val_loader : DataLoader
            The DataLoader object containing the dataset to evaluate on

        Returns
        -------
        (val_loss, val_accuracy) : (float, float)
            Returns the validation loss and accuracy

        """
        self.eval()  # Set model to evaluate mode

        val_loss_accumulator = 0.0
        val_accuracy_accumulator = 0.0
        pbar = trange(len(val_loader.dataset), desc='Validating.. ')

        for batch_idx, t in enumerate(val_loader):

            data, targets = t

            # for idx, d in enumerate(data):
            #     data[idx] = Variable(d, requires_grad=False)
            # targets = Variable(targets, requires_grad=False)

            if torch.cuda.is_available():
                for idx, d in enumerate(data):
                    data[idx] = d.cuda()
                targets = targets.cuda()
                self.cuda()

            predictions = self(data)

            validation_loss = self.criterion(predictions, targets)
            val_loss_accumulator += validation_loss.item()

            batch_len = len(data[0])

            if batch_idx % 10 == 0:
                # Update tqdm bar
                if ((batch_idx + 10) * batch_len) <= len(val_loader.dataset):
                    pbar.update(10 * batch_len)
                else:
                    pbar.update(len(val_loader.dataset) - int(batch_idx * batch_len))
            val_accuracy_accumulator += self.metrics.get_score(predictions,
                                                               targets)

        pbar.close()
        validation_loss = val_loss_accumulator * \
            val_loader.batch_size / len(val_loader.dataset)
        validation_accuracy = val_accuracy_accumulator * \
            val_loader.batch_size / len(val_loader.dataset)

        return validation_loss, validation_accuracy

    def run_test(self, data_loader, figure_path=None, plot=False):
        """Will conduct the test suite to determine model strength."""
        return self.metrics.run_test(
            network=self,
            data_loader=data_loader,
            figure_path=figure_path,
            plot=plot)

    # TODO: Instead of self.cpu(), use is_cuda to know if you can use gpu
    def forward_pass(self, data_loader, convert_to_class=False):
        """
        Allow the user to pass data through the network.

        Parameters
        ----------
        data_loader : DataLoader
            DataLoader object to make the pass with.
        convert_to_class : boolean
            If true, list of class predictions instead of class probabilites.

        Returns
        -------
        outputs : numpy.ndarray
            Numpy matrix with the output. Same shape as network out_dim.

        """
        self.eval()
        # prediction_shape used to aggregate network outputs
        # (e.g. with or without class conversion)
        pred_collector = torch.tensor([])
        for batch_idx, (*data, _) in enumerate(data_loader):

            if torch.cuda.is_available():
                data = data.cuda()
                self.cuda()
            # Get raw network output
            predictions = self(data)
            if self._num_classes:
                # Get probabilities
                predictions = nn.Softmax(dim=1)(predictions)
                if convert_to_class:
                    predictions = torch.tensor(
                        self.metrics.get_class(in_matrix=predictions.cpu())).float()
            # Aggregate predictions
            pred_collector = torch.cat([pred_collector, predictions.cpu()])
        # Tensor comes in as float so convert back to int if returning classes
        if self._num_classes and convert_to_class:
            pred_collector = pred_collector.long()
        if isinstance(pred_collector, torch.Tensor):
                pred_collector = pred_collector.detach().numpy()
        return pred_collector

    def save_model(self, save_path=None):
        """
        Save the model (and its' input networks)
        :param save_path: The save directory (not a file)
        :return: save path, for recursive purposes
        """

        if not save_path:
            save_path = r"saved_models/"

        if not save_path.endswith("/"):
            save_path = save_path + "/"

        save_path = save_path + "{}_{}/".format(
            self.name, datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
        logger.info("No save path provided, saving to {}".format(save_path))
        # recursive recursive recursive
        if self.input_networks is not None:
            for i, input_network in enumerate(self.input_networks):
                input_network.save_model(save_path)

        if not os.path.exists(save_path):
            os.makedirs(save_path)

        self.save_path = save_path  # TODO: I don't think this is necessary

        # to improve: # object.__getstate__() https://docs.python.org/3/library/pickle.html#example
        model_file_path = save_path + "model.pkl"
        state_dict_file_path = save_path + "state_dict.pkl"
        pickle.dump(self, open(model_file_path, "wb"), 2)
        pickle.dump(self.state_dict, open(state_dict_file_path, "wb"), 2)  # TODO: pretty sure this isn't necessary

    # TODO: update the state dict and push to the appropriate device.
    # TODO: caitrin save the optimizers state dict? even though this is included with our instance?
    # https://pytorch.org/tutorials/beginner/saving_loading_models.html#what-is-a-state-dict
    # TODO: implement, add in classification and input layers?
    @classmethod
    def load_model(cls, load_path, load_complete_model_stack=True):
        """
        Load the model from the given directory.
        :param load_path: The load directory (not a file)
        :param load_ensemble: Whether to load all parent networks as well. Not yet implemented.
        :return: a network object
        """

        if not load_path.endswith("/"):  # TODO: does this break windows?? no idea.
            load_path = load_path + "/"

        model_file_path = load_path + "model.pkl"  # TODO: is it dumb to have a constant name?

        instance = pickle.load(open(model_file_path, 'rb'))

        return instance
