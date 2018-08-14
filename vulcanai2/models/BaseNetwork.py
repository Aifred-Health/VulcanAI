import abc

import torch
import torch.nn as nn
from torch import optim
from torch.autograd import Variable

import torch.nn.modules.loss as Loss
from .Layers import * 

import time
import pydash as pdash
from tqdm import tqdm
from datetime import datetime

class BaseNetwork(nn.Module):

    #TODO: this should be the same for every model, given that you pass a config??
    #TODO: come up with an alternate way to validate config
    #TODO: deal with stopping rules
    #TODO: do you have to call any nn.module methods?? do you even actually want to subclass at this point if you have layers that subclass nn.module??
    #TODO: will need to create a wrapper class to use non default keworded parameters for all torch objects
    #TODO: reorder these?
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
        if self._input_network is None:
            self._network = nn.Sequential()
        elif isinstance(self._input_network, nn.Sequential):
            self._network = self._input_network
        else:
            raise ValueError(
                "Input network must be torch.nn.Sequential"
            )
        self._create_network()


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

    # #TODO: deal with the fact that you copied this
    # #TODO: figure out if you really need this?
    # def prepare_batch(self, batch):
    #     if self.is_cuda:
    #         batch = self.cuda_tf()(batch)
    #     if self.mode == 'eval':
    #         batch = self.detach_tf()(batch)
    #     return batch

    def get_all_layers(self):
        layers = []
        for l_name, l in self.input_network['network'].network.named_children():
            if isinstance(l, nn.Sequential):
                for subl_name, subl in l.named_children():
                    layers.append(subl)
            else:
                for param in l.parameters():
                    self.input_dimensions= param.size(0)


    def _initialize_optimizer(self, optim_spec):
        OptimClass = getattr(torch.optim, optim_spec["name"])
        optim_spec = pdash.omit(optim_spec, "name")
        return OptimClass(self.parameters(), **optim_spec)

    def _get_criterion(self, criterion_spec):
        CriterionClass = getattr(Loss, criterion_spec["name"])
        criterion_spec = pdash.omit(criterion_spec, "name")
        return CriterionClass(**criterion_spec)
                

    def _initialize_scheduler(self, optimizer):
        return self._scheduler(optimizer)


    #TODO: include plot as parameter
    #TODO: include stopping rules
    #TODO: use_gpu should probably go somewhere else in the future...
    def fit(self, train_loader, val_loader, epochs, retain_graph=False):
        
        self.optim = self._initialize_optimizer(self._optim_spec)
        self.criterion = self._get_criterion(self._criter_spec)
        #if self._scheduler:
        #    scheduler = self._initialize_scheduler(self, optimizer)

        try:
            for epoch in range(epochs):
                print("--> Epoch: {}/{}".format(
                    epoch,
                    epochs - 1
                ))

                #if self._scheduler:
                #    self._scheduler.step(epoch)

                for mode in ['train', 'val']:
                    if mode == 'train':
                        self.train()  # Set model to training mode

                        loader = tqdm(train_loader, total=len(train_loader))

                        for batch_idx, (data, target) in enumerate(loader):

                            data = Variable(data)
                            target = Variable(target)

                            if torch.cuda.is_available():
                                data = data.cuda()
                                target = target.cuda()

                            # Forward + Backward + Optimize
                            self.zero_grad()
                            self.optim.zero_grad()
                            output = self(data)
                            loss = self.criterion(output, target)
                            loss.backward(retain_graph=retain_graph)
                            self.optim.step()
                        print('\tTrain set: Loss: {:.6f}'.format(loss.item()))

                    else:
                        self.eval()  # Set model to evaluate mode
                        test_loss = 0
                        correct = 0
                        with torch.no_grad():
                            for data, target in val_loader:

                                data = Variable(data)
                                target = Variable(target)

                                if torch.cuda.is_available():
                                    data = data.cuda()
                                    target = target.cuda()

                                output = self(data)
                                test_loss += nn.CrossEntropyLoss()(output, target)
                                _, pred = torch.max(output.data, 1)
                                #pred = output.max(1)[1] # get the index of the max log-probability
                                correct += (pred == target).sum()

                        test_loss /= len(val_loader.dataset)
                        print('\n\tTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
                            test_loss, correct, len(val_loader.dataset),
                            100. * correct / len(val_loader.dataset)))
        except KeyboardInterrupt:
            print("\n\n**********Training stopped prematurely.**********\n\n")
        finally:
            print(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

    #THIS IS FOR INFERENCE
    def predict(self):

        """
        Allow the implementer to quickly get outputs from the network.
        Args:
            input_data: Numpy matrix to make the predictions on
            convert_to_class: If true, output the class
                             with highest probability
        Returns: Numpy matrix with the output probabilities
                 with each class unless otherwise specified.
        """

        pass

    #TODO: this is for the test data
    def evaluate(self):
        pass

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
