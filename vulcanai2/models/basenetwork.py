# -*- coding: utf-8 -*-
"""Defines the basenetwork class"""
import abc
from torch.autograd import Variable
import torch.nn.modules.loss as loss
from .layers import *
from .metrics import Metrics
import pydash as pdash
from tqdm import tqdm, trange
from datetime import datetime
import logging
import os
import pickle
from collections import OrderedDict

logger = logging.getLogger(__name__)


# noinspection PyDefaultArgument
class BaseNetwork(nn.Module):
    """
    Base class upon which all Vulcan NNs will be based.
    """
    # TODO: not great to use mutables as arguments.
    # TODO: reorganize these.
    def __init__(self, name, dimensions, config, save_path=None, input_network=None, num_classes=None, 
                 activation=nn.ReLU(), pred_activation=nn.Softmax(dim=1), optim_spec={'name': 'Adam', 'lr': 0.001},
                 lr_scheduler=None, stopping_rule='best_validation_error', criter_spec=None):
        """
        Defines the network object.
        :param name: The name of the network. Used when saving the file.
        :param dimensions: The dimensions of the network.
        :param config: The config, as a dict.
        :param save_path: The name of the file to which you would like to save this network.
        :param input_network: A network object provided as input
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

        self._input_network = input_network
        self._num_classes = num_classes
        
        self._optim_spec = optim_spec
        self._lr_scheduler = lr_scheduler
        self._stopping_rule = stopping_rule
        self._criter_spec = criter_spec

        self._activation = activation
        self._pred_activation = pred_activation
        
        self._create_network(self._activation, self._pred_activation)

        if self._num_classes:
            self.metrics = Metrics(self._num_classes)
        
        self.optim = None
        self._itr = 0

    # TODO: where to do typechecking... just let everything fail?

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
            self._save_path = "{}_{date:%Y-%m-%d_%H:%M:%S}/".format(self.name, date=datetime.datetime.now())
        else:
            self._save_path = value

    @property
    def lr_scheduler(self):
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
        :return: The criterion spec.
        """
        return self._stopping_rule

    @criter_spec.setter
    def criter_spec(self, value):
        self._criter_spec = value

    @abc.abstractmethod
    def _create_network(self, activation, pred_activation):
        pass

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
        :param network: The input network
        :return: The output size of the network's last layer
        """
        with torch.no_grad():
            x = torch.ones(1, self.in_dim)
            x = self(x)# x = network(x)
            return x.size()[1]

    def get_size(self, summary_dict, output):
        """
        Helper function for the function get_output_shapes
        """
        if isinstance(output, tuple):
            for i in range(len(output)):
                summary_dict[i] = OrderedDict()
                summary_dict[i] = self.get_size(summary_dict[i],output[i])
        else:
            summary_dict['output_shape'] = list(output.size())
        return summary_dict
    
    def get_output_shapes(self, input_size):
        """
        Returns the summary of shapes of all layers in the network
        """
        def register_hook(module):
            def hook(module, input, output):
                class_name = str(module.__class__).split('.')[-1].split("'")[0]
                module_idx = len(summary)
            
                m_key = '%s-%i' % (class_name, module_idx+1)
                summary[m_key] = OrderedDict()
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
                    params +=  torch.prod(torch.LongTensor(list(module.bias.size())))
            
                summary[m_key]['nb_params'] = params
                
            if not isinstance(module, nn.Sequential) and \
                not isinstance(module, nn.ModuleList) and \
                not (module == self):
                hooks.append(module.register_forward_hook(hook))
        
        # check if there are multiple inputs to the network
        if isinstance(input_size[0], (list, tuple)):
            x = [Variable(torch.rand(1,*in_size)) for in_size in input_size]
        else:
            x = Variable(torch.rand(1,*input_size))
        
        # create properties
        summary = OrderedDict()
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
        return self._modules

    def get_weights(self):
        """
        Returns a dict containing the parameters of the network. 
        """
        return self.state_dict()

    # #TODO: figure out how this works in conjunction with optimizer
    # #TODO: fix the fact that you copy pasted this
    def cuda(self, device_id=None):
        """Moves all model parameters and buffers to the GPU.
        Arguments:
            device_id (int, optional): if specified, all parameters will be
                copied to that device
        """
        self.is_cuda = True
        return self._apply(lambda t: t.cuda(device_id))
    
    def cpu(self):
        """Moves all model parameters and buffers to the CPU."""
        self.is_cuda = False
        return self._apply(lambda t: t.cpu())

    @abc.abstractmethod
    def _create_network(self, activation, pred_activation):
        pass

    def init_layers(self, layers):
        '''
        Initializes all of the layers 
        '''
        bias_init = 0.01
        for layer in layers:
            classname = layer.__class__.__name__
            if 'BatchNorm' in classname:
                torch.nn.init.uniform_(layer.weight.data)
                torch.nn.init.constant_(layer.bias.data, bias_init)
            elif 'Linear' in classname:
                torch.nn.init.xavier_uniform_(layer.weight.data)
                torch.nn.init.constant_(layer.bias.data, bias_init)
            else:
                pass

    def _init_optimizer(self, optim_spec):
        OptimClass = getattr(torch.optim, optim_spec["name"])
        optim_spec = pdash.omit(optim_spec, "name")
        return OptimClass(self.parameters(), **optim_spec)

    def _get_criterion(self, criterion_spec):
        CriterionClass = getattr(loss, criterion_spec["name"])
        criterion_spec = pdash.omit(criterion_spec, "name")
        return CriterionClass(**criterion_spec)

    def _init_trainer(self):
        self.optim = self._init_optimizer(self._optim_spec)
        self.criterion = self._get_criterion(self._criter_spec)

        self.valid_interv = 2*len(self.train_loader)
        self.epoch = 0


    def fit(self, train_loader, val_loader, epochs, retain_graph=False, valid_interv=None):
        
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.epochs = epochs
        self.retain_graph = retain_graph
        if valid_interv:
            self.valid_interv = valid_interv

        self._init_trainer()

        try:
            for epoch in trange(self.epoch, epochs, desc='Epoch: ', ncols=80):
                train_loss, train_acc= self.train_epoch()
                valid_loss, acc, avg_acc, iou, miou, conf_mat = self.validate()
                tqdm.write("\n Epoch {}:\nTrain Loss: {:.6f} | Test Loss: {:.6f} | Train Acc: {:.2f} | Test Acc: {:.2f}".format(epoch, train_loss, valid_loss, train_acc, avg_acc))
        
        except KeyboardInterrupt:
            print("\n\n**********Training stopped prematurely.**********\n\n")       


    def train_epoch(self):
        self.train()  # Set model to training mode

        train_loss = 0
        pbar = trange(len(self.train_loader.dataset), desc ='Training.. ')
        for batch_idx, (data, targets) in enumerate(self.train_loader):

            data, targets = Variable(data), Variable(targets)

            if torch.cuda.is_available():
                data, targets = data.cuda(), targets.cuda()

            # Forward + Backward + Optimize
            predictions = self(data)
            loss = self.criterion(predictions, targets)
            
            train_loss += loss.item()

            self.optim.zero_grad()
            loss.backward(retain_graph=self.retain_graph)
            self.optim.step()

            if batch_idx % 10 == 0:
                # Update tqdm bar
                if ((batch_idx+10)*len(data)) <= len(self.train_loader.dataset):
                    pbar.update(10 * len(data))
                else:
                    pbar.update(len(self.train_loader.dataset) - int(batch_idx*len(data)))

            _, acc = self.metrics.get_accuracy(predictions, targets)

        pbar.close()
        return (train_loss*len(data)/len(self.train_loader.dataset), acc)
            
    def validate(self):
        self.eval()  # Set model to evaluate mode
        
        val_loss = 0
        pbar = trange(len(self.val_loader.dataset), desc ='Validating.. ')
        for batch_idx, (data, targets) in enumerate(self.val_loader):
                                            
            data, targets = Variable(data, requires_grad=False), Variable(targets, requires_grad=False)

            if torch.cuda.is_available():
                data, targets = data.cuda(), targets.cuda()
            
            predictions = self(data)
            loss = self.criterion(predictions, targets)
            loss_data = float(loss.item())
            val_loss += loss_data / len(self.val_loader.dataset)

            self.metrics.update(predictions.data.cpu().numpy(), targets.cpu().numpy())
            if batch_idx % 10 == 0:
                # Update tqdm bar
                if ((batch_idx+10)*len(data)) <= len(self.val_loader.dataset):
                    pbar.update(10 * len(data))
                else:
                    pbar.update(len(self.val_loader.dataset) - int(batch_idx*len(data)))

        accuracy, avg_accuracy, IoU, mIoU, conf_mat = self.metrics.get_scores()
        self.metrics.reset()
        pbar.close()
        return (val_loss, accuracy, avg_accuracy, IoU, mIoU, conf_mat)
    
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
        # if convert_to_class:
        #     return self.cpu().metrics.get_class(self(torch.Tensor(input_data)))
        # else:
        #     return self.cpu()(torch.Tensor(input_data))

    #TODO: this is copy pasted - edit as appropriate
    def save_model(self, save_path='models'):
        """
        Will save the model parameters to a npz file.
        Args:
            save_path: the location where you want to save the params
        """
        if self.input_network is not None:
            if not hasattr(self.input_network['network'], 'save_name'):
                self.input_network['network'].save_model()

        if not os.path.exists(save_path):
            print('Path not found, creating {}'.format(save_path))
            os.makedirs(save_path)
        file_path = os.path.join(save_path, "{}{}".format(self.timestamp,
                                                          self.name))
        self.save_name = '{}.network'.format(file_path)
        print('Saving model as: {}'.format(self.save_name))

        with open(self.save_name, 'wb') as f:
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

    def save_record(self):
        pass


    def save_metadata(self, file_path):
        pass

    def _transfer_optimizer_state_to_right_device(self):
        # Since the optimizer state is loaded on CPU, it will crashed when the
        # optimizer will receive gradient for parameters not on CPU. Thus, for
        # each parameter, we transfer its state in the optimizer on the same
        # device as the parameter itself just before starting the optimization.
        for group in self.optimizer.param_groups:
            for p in group['params']:
                if p in self.optimizer.state:
                    for _, v in self.optimizer.state[p].items():
                        if torch.is_tensor(v) and p.device != v.device:
                            v.data = v.data.to(p.device)
