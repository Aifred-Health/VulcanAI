# -*- coding: utf-8 -*-
"""Defines the basenetwork class"""
# Core imports
import abc
import torch
from torch.autograd import Variable
from torch import nn

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

sns.set(style='dark')
logger = logging.getLogger(__name__)


class BaseNetwork(nn.Module):
    """
    Defines the BaseNetwork object.

    Parameters
    ----------
    name : str
        The name of the network. Used when saving the file.
    dimensions : list of tuples
        The dimensions of the network.
    config : dict
        The configuration of the network module, as a dict.
    save_path : str
        The name of the file to which you would like to save this network.
    input_network : list of BaseNetwork
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

    # TODO: not great to use mutables as arguments.
    # TODO: reorganize these.
    def __init__(self, name, dimensions, config, save_path=None,
                 input_network=None, num_classes=None,
                 activation=nn.ReLU(), pred_activation=None,
                 optim_spec={'name': 'Adam', 'lr': 0.001},
                 lr_scheduler=None, early_stopping=None,
                 criter_spec=nn.CrossEntropyLoss()):
        """Define, initialize, and build the BaseNetwork."""
        super(BaseNetwork, self).__init__()

        self._name = name
        if isinstance(dimensions, tuple):
            self._dimensions = [dimensions]
        elif isinstance(dimensions, int):
            self._dimensions = (dimensions,)
        else:
            self._dimensions = dimensions
        self._config = config

        self._save_path = save_path

        if not isinstance(input_networks, type(None)) and \
            not isinstance(input_networks, list):
            self._input_networks = [input_networks]
        else:
            self._input_networks = input_networks

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

        if self._dimensions:
            if len(self._dimensions)==1: # Check if length of self._dimensions list is 1, then self.in_dim is the first tuple of self._dimensions
                self.in_dim = self._dimensions[0]
            else:
                # if len(self._dimensions) > 1, then iterate and concatenate all dims
                self.in_dim = []
                for net in self._input_networks:
                    self.in_dim.append(net.in_dim)
        #import pdb; pdb.set_trace()

        self._create_network(
            activation=activation,
            pred_activation=pred_activation)
        
        out_shapes = self.get_output_shapes(network=self.network,input_size=self.in_dim)
        self.out_dim = out_shapes[list(out_shapes)[-1]]['output_shape'][1:] 
        if self._num_classes:
            out_shapes = self.get_output_shapes(network=self.network_tail,input_size=self.out_dim)
            self.out_dim = out_shapes[list(out_shapes)[-1]]['output_shape'][1:] 

    # TODO: where to do typechecking... just let everything fail?

    def forward(self, inputs):
        """Perform a forward pass through the module/modules."""
        if self._input_networks:
            if not isinstance(inputs, list):
                inputs = [inputs]
            models=self._input_networks
            
            outputs = []
            for model, x in zip(models, inputs):
                outputs.append(model(x))
            network_output = self._forward(torch.cat(outputs))
        else:
            network_output = self._forward((inputs))
        
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
    def early_stopping(self, value):
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

    def get_output_shapes(self, network=None, input_size=None):
        """
        Returns the summary of shapes of all layers in the network
        :return: OrderedDict of shape of each layer in the network
        """
        if not network:
            network = self
        if not input_size:
            input_size = self.in_dim

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

                m_key = '%s-%i' % (class_name, module_idx + 1)
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

        # check if input_size is not a int
        if isinstance(input_size, int):
            input_size = [input_size]
        # check if there are multiple inputs to the network
        if isinstance(input_size[0], (list, tuple)):
            x = [Variable(torch.rand(1, *in_size)) for in_size in input_size]        
        else:
            x = Variable(torch.rand(1, *input_size))
        # create properties
        summary = odict()
        hooks = []
        # register hook
        network.apply(register_hook)
        # make a forward pass
        network.cpu()(x)
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
        if apply_inputs and self._input_network is not None:
            self._input_network._toggle_freeze(
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
        :param train_loader: The DataLoader object containing the training data
        :param val_loader: The DataLoader object containing the validation data
        :param epochs: The number of epochs
        :param retain_graph: Specifies whether retain_graph will be true when .backwards is called.
        :param valid_interv: Specifies the number of epochs before validation occurs.
        :return: None
        """

        # In case there is already one, don't overwrite it.
        # Important for not removing the ref from a lr scheduler
        if self.optim is None:
            self._init_trainer()

        try:
            if plot is True:
                fig_number = plt.gcf().number + 1 if plt.fignum_exists(1) else 1
                plt.show()

            for epoch in trange(0, epochs, desc='Epoch: ', ncols=80):

                train_loss, train_acc = self._train_epoch(train_loader, retain_graph)
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
                self = self.cuda()

            # Forward + Backward + Optimize
            predictions = self([data])

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

            predictions = self([data])

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
                self = self.cuda()
                data = data.cuda()
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
        if self._input_networks is not None:
            if self._input_network:
                self._input_network.save_model(save_path)
            else:
                # TODO: should rewrite the save_path structure
                for i, input_network in enumerate(self._input_networks):
                    input_network.save_model("MultiInputNN_"+save_path+"_{}".format(i))

        self.save_path = module_save_path  # TODO: I don't think this is necessary

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

        # my_tensor = my_tensor.to(torch.device('cuda')).
