# -*- coding: utf-8 -*-
"""Defines the basenetwork class."""
# Core imports
import abc
import torch
from torch import nn

# Vulcan imports
from .layers import *
from .utils import set_tensor_device

from .metrics import Metrics
from ..plotters.visualization import display_record, get_save_path

# Generic imports
import pydash as pdash
from tqdm import tqdm, trange
from datetime import datetime
import logging
import os
from os import environ
import pickle
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

    Parameters:
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
            The number of classes to predict. In the binary
            case always specify 2 classes no matter the shape of the data. A
            value of 1 is used for predicting a single continuous value.
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
        device : str or torch.device
            Sets the network module to the relevant device. If cuda available
            in the host console, "cuda:0" will be run which can be overriden.

    Returns:
        network : BaseNetwork

    """

    def __init__(self, name, config, in_dim=None, save_path=None,
                 input_networks=None, num_classes=None,
                 activation=nn.ReLU(), pred_activation=None,
                 optim_spec={'name': 'Adam', 'lr': 0.001},
                 lr_scheduler=None, early_stopping=None,
                 criter_spec=nn.CrossEntropyLoss(),
                 device="cuda:0"):
        """Define, initialize, and build the BaseNetwork."""
        super(BaseNetwork, self).__init__()

        assert isinstance(name, str)
        self._name = name
        self._config = config
        self._save_path = save_path

        # Turn into list if not list
        if input_networks and not isinstance(input_networks, list):
            input_networks = [input_networks]

        if input_networks:
            self.input_networks = nn.ModuleDict()
            for in_net in input_networks:
                self._add_input_network(in_net)
        else:
            self.input_networks = input_networks

        self._set_final_layer_parameters(num_classes, pred_activation)

        self._num_classes = num_classes

        self._lr_scheduler = lr_scheduler
        self._early_stopping = early_stopping

        if self._num_classes:
            self.metrics = Metrics()

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

        if in_dim:
            if isinstance(in_dim, int):
                self.in_dim = tuple([in_dim])
            else:
                self.in_dim = in_dim
        else:
            if input_networks is None:
                raise ValueError("BaseNetwork must have either in_dim or \
                                input_networks")

        # Creates the Network, and re-writes the self.in_dim
        self._create_network(
            activation=activation,
            pred_activation=pred_activation)

        # Compute self.out_dim of the network
        self.out_dim = self._get_out_dim()

        self.device = device
        self._optim_spec = optim_spec
        self._criter_spec = criter_spec

    def _add_input_network(self, in_network):
        """
        Add a new network to  an input for this network.

        New input network will exist at the end of the current set of
        input_networks which will need to correspond with .

        Parameters:
            in_network : BaseNetwork
                A network to append to the set of self.input_networks.

        Returns:
            None

        """
        assert isinstance(in_network, BaseNetwork)
        assert in_network is not self
        if not self.input_networks:
            self.input_networks = nn.ModuleDict()
        else:
            assert isinstance(self.input_networks, nn.ModuleDict)
        self.input_networks[in_network.name] = in_network
        self.in_dim = self._get_in_dim()

    def _set_final_layer_parameters(self, pred_activation, criter_spec):
        """
        Sets and checks the parameters used in the final output layer.

        Final transform is needed in forward pass in the case of
        nn.CrossEntropyLoss as this class combines nn.NLLLoss and softmax,
        meaning the outputs are not softmax transformed.

        Parameters:
            pred_activation : torch.nn.Module
                The desired activation function for the prediction layer.
            criter_spec : dict
                criterion specification with name and all its parameters.

        """
        self._final_transform = None

        if isinstance(criter_spec, nn.CrossEntropyLoss):
            if pred_activation:
                raise ValueError("The nn.CrossEntropyLoss class combines  \
                            nn.NLLLoss and softmax for improved efficiency, \
                            you cannot set pred_activation when \
                            criter_spec is set to an instance of this class. \
                            Set pred_activation to none or change criter_spec."
                                 )
            self._final_transform = nn.Softmax(dim=1)

    @abc.abstractmethod
    def _merge_input_network_outputs(self, inputs):
        """Abstract method used to define how to handle multi-inputs."""
        raise NotImplementedError

    def extra_repr(self):
        """Set the extra representation of the module."""
        return '(device): torch.'+self.device.__repr__()

    @torch.no_grad()
    def _get_out_dim(self):
        """
        Return the network output shape.

        Perform a single forward pass using made-up data.

        Returns:
            shape : tuple
                The output shape of the network.

        """
        if self.network:
            out = self.network(torch.ones([1, *self.in_dim]))
            return tuple(out.shape[1:])
        else:
            return None

    @torch.no_grad()
    def _get_in_dim(self):
        """
        Return the network input shape.

        Perform a single forward pass through all the input networks
        and merge together to get input shape of this network.

        Returns:
            shape : tuple
                The input shape of the network.

        """
        # Create empty input tensors
        in_tensors = []
        for in_net in self.input_networks.values():
            in_tensors.append(torch.ones([1, *in_net.out_dim]))
        output = self._merge_input_network_outputs(in_tensors)
        return tuple(output.shape[1:])

    @property
    def device(self):
        """
        Return the relevant device associalted with network module.

        Returns:
            device : torch.device
                Relevant device associalted with the network module.

        """
        return next(self.network.parameters()).device

    @device.setter
    def device(self, device):
        """
        Network device setter.

        If the user specifies invalid device id, raises
        RuntimeError for invalid device ordinal.

        Parameters:
            device : str or torch.device
                The device to transfer network to.

        """
        device = torch.device(device if torch.cuda.is_available() else 'cpu')
        if self.network:
            self.network.to(device=device)
        if self.optim:
            for state in self.optim.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = set_tensor_device(v, device=device)

    @property
    def is_cuda(self):
        """
        Return  boolean about whether the network is on cuda device (i.e gpu).

        Returns:
            is_cuda : boolean
                Specifies whether the network is on gpu or not.

        """
        return next(self.network.parameters()).is_cuda

    @property
    def name(self):
        """
        Return the name.

        Returns:
            name : string
                The name of the network.

        """
        return self._name

    @name.setter
    def name(self, value):
        self._name = value

    @property
    def save_path(self):
        """Return the save path of the network.

        Returns:
            save_path : string
                The save path of the network.

        """
        return self._save_path

    @save_path.setter
    def save_path(self, value):
        if not value:
            self._save_path = "{}_{date:%Y-%m-%d_%H_%M_%S}/".format(
                self.name, date=datetime.now())
        else:
            self._save_path = value

    @property
    def lr_scheduler(self):
        """
        Return the network lr_scheduler.

        Returns:
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

        Returns:
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

        Returns:
            _criter_spec : dict
                The criterion specification.

        """
        return self._criter_spec

    @criter_spec.setter
    def criter_spec(self, value):
        self._criter_spec = value

    def get_layers(self):
        """
        Return an ordered dict of all modules in this network (layers).

        Returns:
            layers : OrderedDict()

        """
        return self._modules

    def get_weights(self):
        """
        Return a dictionary containing a whole state of the module.

        Returns:
            weights : dict
                A dictionary containing a whole state of the module.

        """
        return self.state_dict()

    @abc.abstractmethod
    def _create_network(self, **kwargs):
        """
        Define the network. Abstract method that needs to be overridden.

        Returns:
            None

        """
        self.network = None
        pass

    def freeze(self, apply_inputs=False):
        """
        Freeze network weights so training doesn't modify them.

        Parameters:
            apply_inputs : boolean
                Whether to freeze all input networks recursively

        Returns:
            None

        """
        self._toggle_freeze(freeze=True, apply_inputs=apply_inputs)

    def unfreeze(self, apply_inputs=False):
        """
        Unfreeze network weights so training does modify them.

        Parameters:
            apply_inputs : boolean
                Whether to unfreeze all input networks recursively

        Returns:
            None

        """
        self._toggle_freeze(freeze=False, apply_inputs=apply_inputs)

    def _toggle_freeze(self, freeze, apply_inputs):
        # Freeze core network parameters
        for params in self.network.parameters():
            # If freeze is True, set requires_grad to False
            # If freeze is False, set requires_grad to True
            params.requires_grad_(not freeze)
        # Recursively toggle freeze on
        if apply_inputs and self.input_networks:
            for in_net in self.input_networks.values():
                in_net._toggle_freeze(
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

    def _assert_same_devices(self, comparison_device=None):
        """
        Will check if all incoming networks are on the same device.

        Raises specific error about which device the networks need to
        be re-assigned to.

        Specific for when calling fit becuase of grad calculation across
        several devices not compatible with optimizer. Temporary until
        replaced with nn.DataParallel or better multi-gpu implmentation.

        Parameters:
            comparison_device : str or torch.device
                The device to compare current device to.

        """
        if comparison_device is None:
            comparison_device = self.device
        incompatible_collector = {}
        if self.input_networks:
            for net_name, net in self.input_networks.items():
                if net.input_networks:
                    net._assert_same_devices(comparison_device)
                if net.device != comparison_device:
                    incompatible_collector[net_name] = net.device
        if incompatible_collector:
            raise ValueError(
                "The following input networks' devices do not "
                "match deepest network's device '{}':\n{}".format(
                    comparison_device,
                    incompatible_collector))

    def fit(self, train_loader, val_loader, epochs,
            retain_graph=None, valid_interv=4, plot=False, save_path=None):
        """
        Train the network on the provided data.

        Parameters:
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
            plot : boolean
                Whether or not to plot training metrics in real-time.
            save_path : str
                Path to save graphics at

        Returns:
            None

        """
        # Check all networks are on same device.
        self._assert_same_devices()

        # In case there is already one, don't overwrite it.
        # Important for not removing the ref from a lr scheduler
        if self.optim is None:
            self._init_trainer()

        try:
            if plot:
                fig_number = plt.gcf().number + 1 if plt.fignum_exists(1) else 1
                plt.show()

            if save_path:
                if save_path.endswith('/'):
                    save_path = save_path + self.name + '_'
                else:
                    save_path = save_path + '/' + self.name + '_'
                save_path = get_save_path(save_path, vis_type='train')
            for epoch in trange(epochs, desc='Epoch: '):

                train_loss, train_acc = self._train_epoch(train_loader,
                                                          retain_graph)
                if self.lr_scheduler:
                    self.lr_scheduler.step(epoch=epoch)

                valid_loss = valid_acc = np.nan
                if epoch % valid_interv == 0:
                    valid_loss, valid_acc = self._validate(val_loader)

                tqdm.write(
                    "\n Epoch {}:\n"
                    "Train Loss: {:.6f} | Test Loss: {:.6f} |"
                    "Train Acc: {:.4f} | Test Acc: {:.4f}".format(
                        self.epoch,
                        train_loss,
                        valid_loss,
                        train_acc,
                        valid_acc))

                self.record['epoch'].append(self.epoch)
                self.record['train_error'].append(train_loss)
                self.record['train_accuracy'].append(train_acc)
                self.record['validation_error'].append(valid_loss)
                self.record['validation_accuracy'].append(valid_acc)

                if plot:
                    plt.ion()
                    plt.figure(fig_number)
                    display_record(record=self.record, save_path=save_path)

                self.epoch += 1

        except KeyboardInterrupt:
            logger.warning(
                "\n\n**********KeyboardInterrupt: "
                "Training stopped prematurely.**********\n\n")

    def _train_epoch(self, train_loader, retain_graph):
        """
        Trains the network for 1 epoch.

        Parameters:
            train_loader : DataLoader
                The DataLoader object containing the dataset to train on.

        Returns:
            (train_loss, train_accuracy) : (float, float)
                Returns the train loss and accuracy.

        """
        self.train()  # Set model to training mode

        train_loss_accumulator = 0.0
        train_accuracy_accumulator = 0.0
        pbar = trange(len(train_loader.dataset), desc='Training.. ')

        for data, targets in train_loader:
            data = set_tensor_device(data, device=self.device)
            targets = set_tensor_device(targets, device=self.device)

            # Forward + Backward + Optimize
            predictions = self(data)
            train_loss = self.criterion(predictions, targets)
            train_loss_accumulator += train_loss.item()

            self.optim.zero_grad()
            train_loss.backward(retain_graph=retain_graph)
            self.optim.step()

            # here can add more options in the future if other metrics
            # will be used
            average = None

            if self._num_classes == 1:
                metric = "mse"
            else:
                metric = "accuracy"

            # will be fixed in the future
            train_accuracy_accumulator += self.metrics.get_score(
                targets=targets,
                predictions=predictions,
                metrics=metric,
                average=average)[metric]

            pbar.update(train_loader.batch_size)
        pbar.close()

        train_loss = train_loss_accumulator * \
            train_loader.batch_size / len(train_loader.dataset)
        train_accuracy = train_accuracy_accumulator * \
            train_loader.batch_size / len(train_loader.dataset)

        return train_loss, train_accuracy

    @torch.no_grad()
    def _validate(self, val_loader):
        """
        Validate the network on the validation data.

        Parameters:
            val_loader : DataLoader
                The DataLoader object containing the dataset to evaluate on

        Returns:
            (val_loss, val_accuracy) : (float, float)
                Returns the validation loss and accuracy

        """
        self.eval()  # Set model to evaluate mode

        val_loss_accumulator = 0.0
        val_accuracy_accumulator = 0.0
        pbar = trange(len(val_loader.dataset), desc='Validating.. ')

        for data, targets in val_loader:

            data = set_tensor_device(data, device=self.device)
            targets = set_tensor_device(targets, device=self.device)

            predictions = self(data)
            validation_loss = self.criterion(predictions, targets)
            val_loss_accumulator += validation_loss.item()

            # here can add more options in the future if other metrics
            # will be used
            average = None

            if self._num_classes == 1:
                metric = "mse"
            else:
                metric = "accuracy"

            # will be fixed in the future
            val_accuracy_accumulator += self.metrics.get_score(
                targets=targets,
                predictions=predictions,
                metrics=metric,
                average=average)[metric]

            pbar.update(val_loader.batch_size)
        pbar.close()

        validation_loss = val_loss_accumulator * \
            val_loader.batch_size / len(val_loader.dataset)
        validation_accuracy = val_accuracy_accumulator * \
            val_loader.batch_size / len(val_loader.dataset)

        return validation_loss, validation_accuracy

    def run_test(self, data_loader, plot=False, save_path=None, pos_label=1,
                 transform_outputs=False, transform_callable=None, **kwargs):
        """
        Will conduct the test suite to determine network strength. Using
        metrics.run_test

        Parameters:
            data_loader : DataLoader
                A DataLoader object to run the test with.
            save_path : string
                Folder to place images in.
            plot: bool
                Determine if graphs should be plotted in real time.
            pos_label: int
                The label that is positive in the binary case for macro
                calculations.
            transform_outputs : boolean
                Not used in the multi-class case.
                If true, transform outputs using metrics.transform_outputs.
                If no transform_callable is provided then the defaults in
                metrics.transform_outputs will be used: class converstion for
                one-hot encoded, and identity for one-dimensional outputs.
                Multiple class multiple outputs are not yet supported.
            transform_callable: callable
                Not used in the multi-class case.
                Used to transform values if transform_outputs is true,
                otherwise defaults in metrics.transform_outputs will be used.
                An example could be np.round
            kwargs: dict of keyworded parameters
                Values passed to transform callable (function parameters)

        Returns:
            results : dict

        """
        return self.metrics.run_test(
            network=self,
            data_loader=data_loader,
            save_path=save_path,
            plot=plot,
            pos_label=pos_label,
            transform_outputs=transform_outputs,
            transform_callable=transform_callable,
            **kwargs
        )

    def cross_validate(self, data_loader, k, epochs,
                       average_results=True, retain_graph=None,
                       valid_interv=4, plot=False, save_path=None,
                       transform_outputs=False, transform_callable=None,
                       **kwargs):
        """
        Perform k-fold cross validation given a Network and DataLoader object.

        Parameters:
            network : BaseNetwork
                Network descendant of BaseNetwork.
            data_loader : torch.utils.data.DataLoader
                The DataLoader object containing the totality of the data to use
                for k-fold cross validation.
            k : int
                The number of folds to split the training into.
            epochs : int
                The number of epochs to train the network per fold.
            average_results : boolean
                Whether or not to return results from all folds or just an average.
            retain_graph : {None, boolean}
                Whether retain_graph will be true when .backwards is called.
            valid_interv : int
                Specifies after how many epochs validation should occur.
            plot : boolean
                Whether or not to plot all results in prompt and charts.
            save_path : str
                Where to save all figures and results.
            transform_outputs : boolean
                Not used in the multi-class case.
                If true, transform outputs using metrics.transform_outputs.
                If no transform_callable is provided then the defaults in
                metrics.transform_outputs will be used: class converstion for
                one-hot encoded, and identity for one-dimensional outputs.
                Multiple class multiple outputs are not yet supported.
            transform_callable: callable
                Not used in the multi-class case.
                Used to transform values if transform_outputs is true,
                otherwise defaults in metrics.transform_outputs will be used.
                An example could be np.round
            kwargs: dict of keyworded parameters
                Values passed to transform callable (function parameters)


        Returns:
            results : dict
                If average_results is on, return dict of floats.
                If average_results is off, return dict of float lists.

        """
        # TODO: deal with repeated default parameters
        return self.metrics.cross_validate(
            network=self,
            data_loader=data_loader,
            k=k,
            epochs=epochs,
            average_results=average_results,
            retain_graph=retain_graph,
            valid_interv=valid_interv,
            plot=plot,
            save_path=save_path,
            transform_outputs=transform_outputs,
            transform_callable=transform_callable,
            **kwargs)

    def forward(self, inputs, **kwargs):
        """
        Perform a forward pass through the modules.

        If the network is defined with `num_classes` then it contains a
        classification layer/network tail. The inputs will be passed
        through the networks and then through the classifier.
        If not, the input is passed through the network and
        returned without passing through a classification layer.

        Parameters:
            inputs : list(torch.Tensor)
                The inputs to pass throught the network.

        Returns:
            output : torch.Tensor

        """
        if not isinstance(inputs, list):
            inputs = [inputs]

        if self.input_networks:
            net_outs = []
            # Loop through all input networks and pass through respective
            # input data tensors to collect their outputs. Use the specified
            # merge_inputs functionality to combine all the outputs to create
            # the input for this network.
            for in_net, x in zip(self.input_networks.values(), inputs):
                net_outs.append(in_net(x))
            output = self._merge_input_network_outputs(net_outs)
        else:
            output = torch.cat(inputs, dim=1)

        output = set_tensor_device(output, device=self.device)
        # Return actionable error if input shapes don't match up.
        if output.shape[1:] != self.in_dim:
            raise ValueError(
                "Input data incorrect dimension shape for network: {}. "
                "Expecting shape {} but recieved shape {}".format(
                    self.name, self.in_dim, output.shape[1:]))
        return self.network(output)

    @torch.no_grad()
    def forward_pass(self, data_loader, transform_outputs=False,
                     transform_callable=None, **kwargs):
        """
        Allow the user to pass data through the network.

        Parameters:
            data_loader : DataLoader
                DataLoader object to make the pass with.
            transform_outputs : boolean
                If true, transform outputs using metrics.transform_outputs.
                If no transform_callable is provided then the defaults in
                metrics.transform_outputs will be used: class converstion for
                one-hot encoded, and identity for one-dimensional outputs.
                Multiple class multiple outputs are not yet supported.
            transform_callable: callable
                Used to transform values if transform_outputs is true,
                otherwise defaults in metrics.transform_outputs will be used.
                An example could be np.round
            kwargs: dict of keyworded parameters
                Values passed to transform callable (function parameters)

        Returns:
            outputs : numpy.ndarray
                Numpy matrix with the output. Same shape as network out_dim.

        """
        self.eval()
        # prediction_shape used to aggregate network outputs
        # (e.g. with or without class conversion)
        # so far always a float.
        dtype = torch.float
        pred_collector = torch.tensor([], dtype=dtype, device=self.device)
        for data, _ in data_loader:
            # Get raw network output
            raw_outputs = self(data)
            if self._num_classes:
                if self._final_transform:
                    predictions = self._final_transform(raw_outputs)
                else:
                    predictions = raw_outputs

                if transform_outputs:
                    predictions = torch.tensor(
                        self.metrics.transform_outputs(
                            in_matrix=predictions,
                            transform_callable=transform_callable, **kwargs
                        ),
                        device=self.device)
            else:
                predictions = raw_outputs

            # Aggregate predictions
            pred_collector = torch.cat([pred_collector, predictions])
        # TODO: check this
        return pred_collector.cpu().numpy()

    def save_model(self, save_path=None):
        """
        Save the model (and it's input networks).

        Parameters:
            save_path : str
                The save directory (not a file)

        Returns:
            save_path : str
                The save path where you'll find the model directly.

        """
        if not save_path:
            save_path = r"saved_models/"

        if not save_path.endswith("/"):
            save_path = save_path + "/"

        save_path = save_path + "{}_{}/".format(
            self.name, datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
        logger.info("No save path provided, saving to {}".format(save_path))
        # Recursively save the input networks as well.
        if self.input_networks:
            for in_net in self.input_networks.values():
                in_net.save_model(save_path)

        if not os.path.exists(save_path):
            os.makedirs(save_path)

        self.save_path = save_path

        # TODO: to improve: # object.__getstate__()
        model_file_path = save_path + "model.pkl"
        state_dict_file_path = save_path + "state_dict.pkl"
        pickle.dump(self, open(model_file_path, "wb"), 2)
        pickle.dump(self.state_dict, open(state_dict_file_path, "wb"), 2)
        return self.save_path

    @classmethod
    def load_model(cls, load_path, load_complete_model_stack=True):
        """
        Load the model from the given directory.

        Parameters:
            load_path : str
                The load directory (not a file)
            load_complete_model_stack : boolean
                Whether to load all parent networks as well. Not yet implemented.

        Returns:
            network : BaseNetwork
                A network object with all components intact.

        """
        # TODO: does this break windows?? no idea.
        if not load_path.endswith("/"):
            load_path = load_path + "/"

        model_file_path = load_path + "model.pkl"

        instance = pickle.load(open(model_file_path, 'rb'))

        return instance
