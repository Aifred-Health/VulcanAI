import abc

import torch
import torch.nn as nn
from torch import optim
from torch.autograd import Variable
import torch.nn.modules.loss as Loss

from .layers import *
from .metrics import Metrics

import time
import pydash as pdash
from tqdm import tqdm, trange, tnrange, tgrange
from datetime import datetime
import numpy as np

class BaseNetwork(nn.Module):

    def __init__(self, name, dimensions, config, save_path=None, input_network=None, num_classes=None, 
                activation=nn.ReLU(), pred_activation=nn.Softmax(dim=1), optim_spec={'name': 'Adam', 'lr': 0.001}, 
                lr_scheduler=None, stopping_rule='best_validation_error', criter_spec=None):
        """
        :param name:
        :param dimensions:
        :param config:
        :param save_path:
        :param input_network:
        :param num_classes:
        :param activation:
        :param pred_activation:
        :param optimizer:
        :param learning_rate:
        :param lr_scheduler:
        :param stopping_rule:
        :param criterion:
        :return:
        """
        super(BaseNetwork, self).__init__()
        self._name = name
        self._dimensions = dimensions
        self._config = config

        self._save_path = save_path

        self._input_network = input_network #TODO: change and check type here?
        self._num_classes = num_classes
        self._activation = activation
        self._pred_activation = pred_activation
        self._optim_spec = optim_spec
        self._lr_scheduler = lr_scheduler
        self._stopping_rule = stopping_rule
        self._criter_spec = criter_spec
        
        self._create_network()
        self._itr = 0


    #TODO: where to do typechecking... just let everything fail?
    #TODO: add on additional if you want to be able to re-create a network?

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, value):
        self._name = value

    @property
    def save_path(self):
        return self._save_path

    @save_path.setter
    def save_path(self, value):
        if not value:
            self.save_path = "{}_{date:%Y-%m-%d_%H:%M:%S}/".format(self.name, date=datetime.datetime.now())
        else:
            self._save_path = value

    @property
    def learning_rate(self):
        return self._learning_rate

    @learning_rate.setter
    def learning_rate(self, value):
        self._learning_rate = value

    @property
    def lr_scheduler(self):
        return self._lr_scheduler

    @lr_scheduler.setter
    def lr_scheduler(self, value):
        self._lr_scheduler = value

    @property
    def stopping_rule(self):
        return self._stopping_rule

    @stopping_rule.setter
    def stopping_rule(self, value):
        self._stopping_rule = value

    @property
    def criterion(self):
        return self._criterion

    def get_conv_output_size(self):
        """
        Helper function to calculate the size of the flattened 
        features after the last conv layer
        """
        with torch.no_grad():
            x = torch.ones(1, *self.in_dim)
            x = self.conv_network(x)
            return x.numel()

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
    def _create_network(self):
        pass

    def get_all_layers(self):
        layers = []
        for l_name, l in self.input_network['network'].network.named_children():
            if isinstance(l, nn.Sequential):
                for subl_name, subl in l.named_children():
                    layers.append(subl)
            else:
                for param in l.parameters():
                    self.input_dimensions= param.size(0)

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
        CriterionClass = getattr(Loss, criterion_spec["name"])
        criterion_spec = pdash.omit(criterion_spec, "name")
        return CriterionClass(**criterion_spec)

    def _init_trainer(self):
        self.optim = self._init_optimizer(self._optim_spec)
        self.criterion = self._get_criterion(self._criter_spec)

        self.valid_interv = 2*len(self.train_loader)
        self.metrics = Metrics(self._num_classes)
        self.epoch = 0


    def fit(self, train_loader, val_loader, epochs, retain_graph=False, valid_interv=None,
                                                            print_iu=False):
        
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.epochs = epochs
        self.retain_graph = retain_graph
        if valid_interv:
            self.valid_interv = valid_interv
        self.print_iu = print_iu

        self._init_trainer()

        for epoch in trange(self.epoch, epochs, desc='Epoch: ', ncols=80):
            self.epoch = epoch
            self.train_epoch()                    


    def train_epoch(self):

        self.train()  # Set model to training mode

        for batch_idx, (data, targets) in tqdm(
                                            enumerate(self.train_loader), 
                                            total=len(self.train_loader),
                                            desc='Training at epoch=%d' % self.epoch, 
                                            ncols=80,
                                            leave=False):

            itr = batch_idx + self.epoch * len(self.train_loader)

            self._itr = itr

            if self._itr > 0 and self._itr % self.valid_interv == 0:
                self.validate()

            data, targets = Variable(data), Variable(targets)

            assert self.training

            if torch.cuda.is_available():
                data, targets = data.cuda(), targets.cuda()

            # Forward + Backward + Optimize
            self.optim.zero_grad()
            predictions = self(data)
            loss = self.criterion(predictions, targets)
            loss /= len(targets)
            loss_data = float(loss.item()) 
            loss.backward(retain_graph=self.retain_graph)
            self.optim.step()

            correct, acc = self.metrics.get_accuracy(predictions, targets)
        tqdm.write("\n Epoch {}: Train Loss: {}, Acc: {}%".format(self.epoch, loss_data, correct))

    def validate(self):
        training = self.training
        self.eval()  # Set model to evaluate mode

        val_loss = 0
        for batch_idx, (data, targets) in tqdm(
                                            enumerate(self.val_loader), 
                                            total=len(self.val_loader),
                                            desc='Validating at epoch=%d' % self.epoch, 
                                            ncols=80,
                                            leave=False):

            data, targets = Variable(data), Variable(targets)

            if torch.cuda.is_available():
                data, targets = data.cuda(), targets.cuda()

            predictions = self(data)
            loss = self.criterion(predictions, targets)
            loss_data = float(loss.item())
            val_loss += loss_data / len(targets)

            self.metrics.update(predictions.data.max(1)[1].cpu().numpy(), targets.cpu().numpy())

        score, class_iou = self.metrics.get_scores()
        tqdm.write("    Valid Loss: {}".format(val_loss))
        for k, v in score.items():
            tqdm.write("        {}{}".format(k, v))
        if self.print_iu is True:
            for i in range(self._num_classes):
                tqdm.write("        {}: {}".format(i, class_iou[i]))
        
        if training:
            self.train()

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


    def save_metadata(self):
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