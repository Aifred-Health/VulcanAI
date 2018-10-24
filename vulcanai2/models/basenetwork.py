# -*- coding: utf-8 -*-
"""Defines the basenetwork class"""
# Core imports
import abc
import torch
from torch.autograd import Variable
from torch import nn

# Vulcan imports
from .layers import *
from .utils import get_size
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
    """Base class upon which all Vulcan NNs will be based."""

    def __init__(self, name, config, in_dim=None, save_path=None,
                 input_networks=None, num_classes=None,
                 activation=nn.ReLU(), pred_activation=None,
                 optim_spec={'name': 'Adam', 'lr': 0.001},
                 lr_scheduler=None, early_stopping=None,
                 criter_spec=nn.CrossEntropyLoss()):
        """
        Defines the network object.

        Parameters
        ----------
        name : str
            The name of the network. Used when saving the file.
        in_dim : list of tuples
            The input dimensions of the network. If inferencing from input_networks,
            it must be a list of either out_dim or conv_flat_dim.
        config : dict
            The configuration of the network module, as a dict.
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

        """
        super(BaseNetwork, self).__init__()

        self._name = name
        
        # self._in_dim = self.in_dim
        if in_dim is not None:
            # Must be a list of tuples(ConvNet)/int(DenseNet)
            if isinstance(in_dim, (tuple, int)):
                self.in_dim = [in_dim] 
            else:
                self.in_dim = in_dim
        else:
            self.in_dim = []
            for net in input_networks:
                self.in_dim.append(net.out_dim)

        self._config = config
        self._save_path = save_path

        # TODO: See if using nn.ModuleDict is faster
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

        self._create_network(
            activation=activation,
            pred_activation=pred_activation)

        out_shapes = self.get_output_shapes(
            network=self.network, input_size=self._in_dim)
        
        self.out_dim = out_shapes[list(out_shapes)[-1]]['output_shape'][1:]
        
        if len(self.out_dim) > 1:
            self.out_dim = tuple(self.out_dim)
        else:
            self.out_dim = self.out_dim[0]
        if self._num_classes:
            out_shapes = self.get_output_shapes(
                network=self.network_tail, input_size=self.out_dim)
            self.out_dim = out_shapes[list(out_shapes)[-1]]['output_shape'][1:]

    @abc.abstractmethod
    def _merge_input_network_outputs(self, inputs):
        """Abstract method used to define how to handle multi-inpus."""
        pass

    def forward(self, inputs, **kwargs):
        """
        Perform a forward pass through the module/modules.
        If the network is defined with `num_classes` then it is
        assumed to be the last network which contains a classification
        layer/network tail. The inputs will be passed
        through the networks and then through the classifier.
        If not, the input is passed through the network and
        returned without passing through a classification layer.
        :param x: input list(torch.Tensor)
        :return: output torch.Tensor        
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
        
    @property
    def name(self):
        """
        Returns the name.

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
        Returns the lr_scheduler
        :return: the lr_scheduler
        """
        return self._lr_scheduler

    @lr_scheduler.setter
    def lr_scheduler(self, value):
        self._lr_scheduler = value

    @property
    def early_stopping(self):
        """
        Returns the stopping rule
        :return: The stoping rule
        """
        return self._early_stopping

    @early_stopping.setter
    def stopping_rule(self, value):
        self._early_stopping = value

    @property
    def criter_spec(self):
        """
        Returns the criterion spec.
        :return: the criterion spec.
        """
        return self._criter_spec

    @criter_spec.setter
    def criter_spec(self, value):
        self._criter_spec = value

    def get_output_shapes(self, network=None, input_size=None):
        """
        Returns the summary of shapes of all layers in the network
        :return: OrderedDict of shape of each layer in the network
        """
        if not input_size:
            input_size = self.in_dim
        # input_size must be a list
        if isinstance(input_size, (tuple, int)):
            input_size = [input_size]

        def register_hook(module):
            """
            Registers a backward hook
            For more info: https://pytorch.org/docs/stable/_modules/torch/tensor.html#Tensor.register_hook
            """
            def hook(module, input, output):
                """
                https://github.com/pytorch/tutorials/blob/8afce8a213cb3712aa7de1e1cf158da765f029a7/beginner_source/former_torchies/nn_tutorial.py#L146
                """
                class_name = str(module.__class__).split('.')[-1].split("'")[0]
                module_idx = len(summary)
                # Test
                m_key = '%s-%i' % (class_name, module_idx + 1)
                summary[m_key] = odict()
                summary[m_key]['input_shape'] = list(input[0].size())
                summary[m_key] = get_size(summary[m_key], output)
                # Test
                params = 0
                if hasattr(module, 'weight'):
                    params += torch.prod(torch.LongTensor(list(module.weight.size())))
                    if module.weight.requires_grad:
                        summary[m_key]['trainable'] = True
                    else:
                        summary[m_key]['trainable'] = False
                if hasattr(module, 'bias'):
                    params += torch.prod(torch.LongTensor(list(module.bias.size())))
                # Test
                summary[m_key]['nb_params'] = params
            if not isinstance(module, nn.Sequential) and \
                    not isinstance(module, nn.ModuleList) and \
                    not (module == self):
                hooks.append(module.register_forward_hook(hook))

        x = []
        for in_size in input_size:
            if isinstance(in_size, tuple):
                x.append(Variable(torch.rand(1, *in_size)))
            else: 
                x.append(Variable(torch.rand(1, *[in_size])))
        
        # create properties
        summary = odict()
        hooks = []

        if not network:
            # register hook
            self.apply(register_hook)
            # make a forward pass
            self.cpu()(x)
        else:
            # register hook
            network.apply(register_hook)
            # make a forward pass
            network.cpu()(torch.cat(x, 1)) # self.network does not multiinput

        # remove these hooks
        for h in hooks:
            h.remove()

        return summary

    def get_layers(self):
        """
        Returns an ordered dict of all modules contained in this module (layers).
        :return: OrderedDict() of all modules.
        """
        return self._modules

    def get_weights(self):
        """
        Returns a dictionary containing a whole state of the module
        :return: A dictionary containing a whole state of the module
        """
        return self.state_dict()

    def print_model_structure(self):
        shapes = self.get_output_shapes()
        for k, v in shapes.items():
            print('{}:'.format(k))
            if isinstance(v, odict):
                for k2, v2 in v.items():
                    print('\t {}: {}'.format(k2, v2))

    @abc.abstractmethod
    def _create_network(self, **kwargs):
        """
        Defines the network. Abstract method that needs to be overridden.
        :return: None
        """
        self.network = None
        pass

    def _init_optimizer(self, optim_spec):
        optim_class = getattr(torch.optim, optim_spec["name"])
        optim_spec = pdash.omit(optim_spec, "name")
        return optim_class(self.parameters(), **optim_spec)

    @staticmethod
    def _init_criterion(criterion_spec):
        return criterion_spec

    def _init_trainer(self):
        self.optim = self._init_optimizer(self._optim_spec)
        self.criterion = self._init_criterion(self._criter_spec)
    
    def fit(self, train_loader, val_loader, epochs,
            retain_graph=None, valid_interv=4, plot=False):
        """
        Trains the network on the provided data.
        :param train_loader: The DataLoader object containing the training data
        :param val_loader: The DataLoader object containing the validation data
        :param epochs: The number of epochs
        :param retain_graph: Specifies whether retain_graph will be true when .backwards is called.
        :param valid_interv: Specifies the number of epochs before validation occurs.
        :return: None
        """

        self._init_trainer()

        try:
            if plot is True:
                fig_number = plt.gcf().number + 1 if plt.fignum_exists(1) else 1
                plt.show()

            for epoch in trange(0, epochs, desc='Epoch: ', ncols=80):

                train_loss, train_acc = self._train_epoch(train_loader, retain_graph)

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
            logger.warning("\n\n**********KeyboardInterrupt: Training stopped prematurely.**********\n\n")

    def _train_epoch(self, train_loader, retain_graph):

        self.train()  # Set model to training mode

        train_loss_accumulator = 0.0
        train_accuracy_accumulator = 0.0
        pbar = trange(len(train_loader.dataset), desc='Training.. ')
        for batch_idx, (data, targets) in enumerate(train_loader):
            
            data, targets = Variable(data), Variable(targets)

            if torch.cuda.is_available():
                data, targets = data.cuda(), targets.cuda()
                self.cuda()

            # Forward + Backward + Optimize
            # TODO: Remove temp
            predictions = self([data, data])

            train_loss = self.criterion(predictions, targets)
            train_loss_accumulator += train_loss.item()

            self.optim.zero_grad()
            train_loss.backward(retain_graph=retain_graph)
            self.optim.step()

            if batch_idx % 10 == 0:
                # Update tqdm bar
                if ((batch_idx + 10) * len(data)) <= len(train_loader.dataset):
                    pbar.update(10 * len(data))
                else:
                    pbar.update(len(train_loader.dataset) - int(batch_idx * len(data)))

            train_accuracy_accumulator += self.metrics.get_score(predictions, targets)

        pbar.close()

        train_loss = train_loss_accumulator * len(data) / len(train_loader.dataset)
        train_accuracy = train_accuracy_accumulator * len(data) / len(train_loader.dataset)

        return train_loss, train_accuracy

    def _validate(self, val_loader):
        """
        Validates the network on the validation data
        :return: (val_loss, accuracy, avg_accuracy, IoU, mIoU, conf_mat) # TODO: update this
        """
        self.eval()  # Set model to evaluate mode

        val_loss_accumulator = 0.0
        val_accuracy_accumulator = 0.0
        pbar = trange(len(val_loader.dataset), desc='Validating.. ')

        for batch_idx, (data, targets) in enumerate(val_loader):

            data, targets = Variable(data, requires_grad=False), Variable(targets, requires_grad=False)

            if torch.cuda.is_available():
                data, targets = data.cuda(), targets.cuda()
                self.cuda()

            predictions = self([data, data])

            validation_loss = self.criterion(predictions, targets)
            val_loss_accumulator += validation_loss.item()

            # self.metrics.update(predictions.data.cpu().numpy(), targets.cpu().numpy())
            if batch_idx % 10 == 0:
                # Update tqdm bar
                if ((batch_idx + 10) * len(data)) <= len(val_loader.dataset):
                    pbar.update(10 * len(data))
                else:
                    pbar.update(len(val_loader.dataset) - int(batch_idx * len(data)))
            val_accuracy_accumulator += self.metrics.get_score(predictions, targets)

        pbar.close()
        validation_loss = val_loss_accumulator * len(data) / len(val_loader.dataset)
        validation_accuracy = val_accuracy_accumulator * len(data) / len(val_loader.dataset)

        return validation_loss, validation_accuracy
    
    def run_test(self, data_loader, figure_path=None, plot=False):
        """
        Will conduct the test suite to determine model strength.
        """
        return self.metrics.run_test(
            network=self,
            data_loader=data_loader,
            figure_path=figure_path,
            plot=plot)

    # TODO: Instead of self.cpu(), use is_cuda to know if you can use gpu
    def forward_pass(self, data_loader, convert_to_class=False):
        """
        Allow the implementer to quickly get outputs from the network.

        :param data_loader: DataLoader object to make the predictions on
        :param convert_to_class: If true, list of class predictions instead
                                 of class probabilites

        :return: Numpy matrix with the output probabilities
                 for each class unless otherwise specified.
        """
        self.eval()
        # prediction_shape used to aggregate network outputs
        # (e.g. with or without class conversion)
        pred_collector = torch.tensor([])
        for batch_idx, (data, _) in enumerate(data_loader):
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
            save_path = r"saved_models/{}_{}/".format(self.name, datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
            logger.info("No save path provided, saving to {}".format(save_path))

        if not save_path.endswith("/"):
            save_path = save_path + "/"

        module_save_path = save_path + "{name}/".format(name=self.name)

        if not os.path.exists(module_save_path):
            os.makedirs(module_save_path)  # let this throw an error if it already exists

        # recursive recursive recursive
        if self.input_networks is not None:
            for i, input_network in enumerate(self.input_networks):
                # "MultiInputNN_"+save_path+"_{}".format(i)
                input_network.save_model(module_save_path)

        self.save_path = save_path  # TODO: I don't think this is necessary

        # to improve: # object.__getstate__() https://docs.python.org/3/library/pickle.html#example
        model_file_path = module_save_path + "model.pkl"
        state_dict_file_path = module_save_path + "state_dict.pkl"
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