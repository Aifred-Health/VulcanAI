# -*- coding: utf-8 -*-
"""Defines the basenetwork class"""
# Core imports
import abc
from torch.autograd import Variable
import torch.nn.modules.loss as loss

# Vulcan imports
from .layers import *
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

sns.set()
logger = logging.getLogger(__name__)


class BaseNetwork(nn.Module):
    """
    Base class upon which all Vulcan NNs will be based.
    """
    # TODO: not great to use mutables as arguments.
    # TODO: reorganize these.
    def __init__(self, name, dimensions, config, save_path=None, input_networks=None, num_classes=None, 
                 activation=nn.ReLU(), pred_activation=nn.Softmax(dim=1), optim_spec={'name': 'Adam', 'lr': 0.001},
                 lr_scheduler=None, early_stopping=None, criter_spec=nn.CrossEntropyLoss):
        """
        Defines the network object.
        :param name: The name of the network. Used when saving the file.
        :param dimensions: The dimensions of the network.
        :param config: The config, as a dict.
        :param save_path: The name of the file to which you would like to save this network.
        :param input_networks: A network object provided as input
        :param num_classes: The number of classes to predict.
        :param activation: The desired activation function for use in the network. Of type torch.nn.Module.
        :param pred_activation: The desired activation function for use in the prediction layer. Of type torch.nn.Module
        :param optim_spec: A dictionary of parameters for the desired optimizer.
        :param lr_scheduler: A callable torch.optim.lr_scheduler
        :param stopping_rule: A string. So far just 'best_validation_error' is implemented.
        :param criter_spec: criterion specification dictionary with name of criterion and all parameters necessary.
        """
        super(BaseNetwork, self).__init__()

        self._name = name
        self._dimensions = dimensions
        self._config = config

        self._save_path = save_path
        
        if not isinstance(input_networks, (odict, type(None))):
            raise TypeError("input_networks must be a OrderdDict")
        else:
            self._input_networks = input_networks

        self._num_classes = num_classes
        
        self._optim_spec = optim_spec
        self._lr_scheduler = lr_scheduler
        self._stopping_rule = stopping_rule
        self._criter_spec = criter_spec

        self._activation = activation
        self._pred_activation = pred_activation

        if self._num_classes:
            self.metrics = Metrics(self._num_classes)
        
        self.optim = None
        self._itr = 0

        self.train_loader = None
        self.val_loader = None
        self.epochs = None
        self.retain_graph = False
        self.valid_interv = None
        self.record = None

        self._create_network()

    # TODO: where to do typechecking... just let everything fail?
    
    @abc.abstractmethod
    def _forward(self):
        pass

    @property
    def name(self):
        """
        Returns the name.
        :return: the name of the network.
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
            self._save_path = "{}_{date:%Y-%m-%d_%H:%M:%S}/".format(self.name, date=datetime.now())
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
    def stopping_rule(self):
        """
        Returns the stopping rule
        :return: The stoping rule
        """
        return self._stopping_rule

    @stopping_rule.setter
    def stopping_rule(self, value):
        self._stopping_rule = value

    @property
    def criter_spec(self):
        """
        Returns the criterion spec.
        :return: the criterion spec.
        """
        return self._stopping_rule

    @criter_spec.setter
    def criter_spec(self, value):
        self._criter_spec = value

    def get_flattened_size(self, network):
        """
        Returns the flattened output size of the conv network's last layer.
        :param network: The network to flatten
        :return: The flattened output size of the conv network's last layer.
        """
        with torch.no_grad():
            x = torch.ones(1, *self.in_dim)
            x = network(x)
            return x.numel()
    
    def get_output_size(self):
        """
        Returns the output size of the network's last layer
        :return: The output size of the network's last layer
        """
        with torch.no_grad():
            x = torch.ones(1, self.in_dim)
            x = self(x)  # x = network(x)
            return x.size()[1]

    def get_size(self, summary_dict, output):
        """
        Helper function for the function get_output_shapes
        """
        if isinstance(output, tuple):
            for i in range(len(output)):
                summary_dict[i] = odict()
                summary_dict[i] = self.get_size(summary_dict[i], output[i])
        else:
            summary_dict['output_shape'] = list(output.size())
        return summary_dict
    
    def get_output_shapes(self, input_size=None):
        """
        Returns the summary of shapes of all layers in the network
        :return: OrderedDict of shape of each layer in the network
        """
        if not input_size:
            if self._input_networks:
                input_size = self._input_networks._dimensions
            else:
                input_size = self._dimensions

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
            
                m_key = '%s-%i' % (class_name, module_idx+1)
                summary[m_key] = odict()
                summary[m_key]['input_shape'] = list(input[0].size())
                summary[m_key] = self.get_size(summary[m_key], output)
            
                params = 0
                if hasattr(module, 'weight'):
                    params += torch.prod(torch.LongTensor(list(module.weight.size())))
                    if module.weight.requires_grad:
                        summary[m_key]['trainable'] = True
                    else:
                        summary[m_key]['trainable'] = False
                if hasattr(module, 'bias'):
                    params += torch.prod(torch.LongTensor(list(module.bias.size())))
            
                summary[m_key]['nb_params'] = params
                
            if not isinstance(module, nn.Sequential) and \
                not isinstance(module, nn.ModuleList) and \
                    not (module == self):
                hooks.append(module.register_forward_hook(hook))
        
        # check if there are multiple inputs to the network
        if isinstance(input_size[0], (list, tuple)):
            x = [Variable(torch.rand(1, *in_size)) for in_size in input_size]
        else:
            x = Variable(torch.rand(1, *input_size))
        
        # create properties
        summary = odict()
        hooks = []
        # register hook
        self.apply(register_hook)
        # make a forward pass
        self.cpu()(x)
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
        for k, v in shapes.items() :
            print('{}:'.format(k))
            if isinstance(v, odict):
                for k2, v2 in v.items():
                    print('\t {}: {}'.format(k2, v2))

    @abc.abstractmethod
    def _create_network(self):
        """
        Defines the network. Abstract method that needs to be overridden.
        :return: None
        """
        pass
    
    def forward(self, *inputs):
        """Perform a forward pass through the module/modules."""

        if self._input_networks:
            if isinstance(self._input_networks, odict) and isinstance(inputs, list):
                outputs = []
                # Need to add a check for input_networks.keys == inputs.keys !

                for key, value in self._input_networks:
                    outputs.append(value._forward(inputs[int(key)]))
                network_output = self._forward(torch.cat(outputs, 1))
            else: 
                network_output = self._forward(self._input_networks['0']._forward(inputs))
        else:
            network_output = self._forward((inputs))

        return network_output

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
        self.epoch = 0

    def fit(self, train_loader, val_loader, epochs, 
            retain_graph=None, valid_interv=4, plot=False):
        """
        Trains the network on the provided data.
        :param train_loader: The DataLoader object containing the training data
        :param val_loader: The DataLoader object containing the validation data
        :param epochs: The number of epochs
        :param retain_graph: Specifies whether retain_graph will be true when .backwards is called.
        :param valid_interv:
        :return: None
        """

        self.train_loader = train_loader
        self.val_loader = val_loader
        self.epochs = epochs
        self.retain_graph = retain_graph

        self._init_trainer()

        self.record = dict(
            epoch=[],
            train_error=[],
            train_accuracy=[],
            validation_error=[],
            validation_accuracy=[]
        )

        try:
            if plot is True:
                fig_number = plt.gcf().number + 1 if plt.fignum_exists(1) else 1
                plt.show()
                
            for epoch in trange(self.epoch, epochs, desc='Epoch: ', ncols=80):

                train_loss, train_acc = self._train_epoch()
                
                valid_loss = valid_acc = np.nan
                if epoch % valid_interv == 0:
                    valid_loss, valid_acc = self._validate()

                tqdm.write("\n Epoch {}:\n"
                           "Train Loss: {:.6f} | Test Loss: {:.6f} |"
                           "Train Acc: {:.4f} | Test Acc: {:.4f}".format(
                                epoch,
                                train_loss,
                                valid_loss,
                                train_acc,
                                valid_acc
                                ))

                self.record['epoch'].append(epoch)
                self.record['train_error'].append(train_loss)
                self.record['train_accuracy'].append(train_acc)
                self.record['validation_error'].append(valid_loss)
                self.record['validation_accuracy'].append(valid_acc)

                if plot is True:
                    plt.ion()
                    plt.figure(fig_number)
                    display_record(record=self.record)

        except KeyboardInterrupt:
            print("\n\n**********KeyboardInterrupt: Training stopped prematurely.**********\n\n")

    def _train_epoch(self):

        self.train()  # Set model to training mode

        train_loss_accumulator = 0.0
        train_accuracy_accumulator = 0.0
        pbar = trange(len(self.train_loader.dataset), desc='Training.. ')
        for batch_idx, (data, targets) in enumerate(self.train_loader):

            data, targets = Variable(data), Variable(targets)

            if torch.cuda.is_available():
                data, targets = data.cuda(), targets.cuda()

            # Forward + Backward + Optimize
            predictions = self(data)

            train_loss = self.criterion(predictions, targets)
            train_loss_accumulator += train_loss.item()

            self.optim.zero_grad()
            train_loss.backward(retain_graph=self.retain_graph)
            self.optim.step()

            if batch_idx % 10 == 0:
                # Update tqdm bar
                if ((batch_idx+10)*len(data)) <= len(self.train_loader.dataset):
                    pbar.update(10 * len(data))
                else:
                    pbar.update(len(self.train_loader.dataset) - int(batch_idx*len(data)))

            train_accuracy_accumulator += self.metrics.get_score(predictions, targets)

        pbar.close()

        # noinspection PyUnboundLocalVariable
        train_loss = train_loss_accumulator*len(data)/len(self.train_loader.dataset)
        train_accuracy = train_accuracy_accumulator*len(data)/len(self.train_loader.dataset)

        return train_loss, train_accuracy

    # noinspection PyUnboundLocalVariable
    def _validate(self):
        """
        Validates the network on the validation data
        :return: (val_loss, accuracy, avg_accuracy, IoU, mIoU, conf_mat) # TODO: update this
        """
        self.eval()  # Set model to evaluate mode

        val_loss_accumulator = 0.0
        val_accuracy_accumulator = 0.0
        pbar = trange(len(self.val_loader.dataset), desc='Validating.. ')

        for batch_idx, (data, targets) in enumerate(self.val_loader):

            data, targets = Variable(data, requires_grad=False), Variable(targets, requires_grad=False)

            if torch.cuda.is_available():
                data, targets = data.cuda(), targets.cuda()

            predictions = self(data)

            validation_loss = self.criterion(predictions, targets)
            val_loss_accumulator += validation_loss.item()

            self.metrics.update(predictions.data.cpu().numpy(), targets.cpu().numpy())
            if batch_idx % 10 == 0:
                # Update tqdm bar
                if ((batch_idx+10)*len(data)) <= len(self.val_loader.dataset):
                    pbar.update(10 * len(data))
                else:
                    pbar.update(len(self.val_loader.dataset) - int(batch_idx*len(data)))
            val_accuracy_accumulator += self.metrics.get_score(predictions, targets)

        self.metrics.reset()
        pbar.close()
        validation_loss = val_loss_accumulator*len(data)/len(self.val_loader.dataset)
        validation_accuracy = val_accuracy_accumulator*len(data)/len(self.val_loader.dataset)

        return validation_loss, validation_accuracy

    def run_test(self, test_x, test_y, figure_path=None, plot=False):
        """
        Will conduct the test suite to determine model strength.
        """
        return self.metrics.run_test(self, test_x, test_y, figure_path, plot)

    # TODO: Instead of self.cpu(), use is_cuda to know if you can use gpu
    def forward_pass(self, input_data, convert_to_class=False):
        """
        Allow the implementer to quickly get outputs from the network.

        Args:
            input_data: Numpy matrix to make the predictions on
            convert_to_class: If true, output the class
                             with highest probability

        Returns: Numpy matrix with the output probabilities
                 with each class unless otherwise specified.
        """
        output = self.cpu()(torch.Tensor(input_data)).data
        if convert_to_class:
            return self.metrics.get_class(output)
        else:
            return output

    # TODO: this is copy pasted - edit as appropriate
    def save_model(self, save_path='models'):
        """
        Will save the model parameters to a npz file.
        Args:
            save_path: the location where you want to save the params
        """
        if self.input_networks is not None:
            if not hasattr(self.input_networks['network'], 'save_name'):
                self.input_networks['network'].save_model()

        if not os.path.exists(save_path):
            print('Path not found, creating {}'.format(save_path))
            os.makedirs(save_path)
        file_path = os.path.join(save_path, "{}{}".format(self.timestamp,
                                                          self.name))
        save_name = '{}.network'.format(file_path)
        print('Saving model as: {}'.format(save_name))

        with open(save_name, 'wb') as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)

        self.save_metadata(file_path)

    @classmethod
    def load_model(cls, load_path):
        """
        Will load the model parameters from npz file.
        Args:
            load_path: the exact location where the model has been saved.
        """
        print('Loading model from: {}'.format(load_path))
        with open(load_path, 'rb') as f:
            instance = pickle.load(f)
        return instance

    # def save_metadata(self, file_path):
    #     """
    #     Save network metadata information to the specified file path
    #     :param file_path: file path
    #     :return: None
    #     """
    #     raise NotImplementedError
