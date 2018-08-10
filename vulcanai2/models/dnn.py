__author__ = 'Caitrin'

import torch.nn as nn
import torch.nn.functional as F
<<<<<<< HEAD
import torch.nn.modules.activation as activations
import torch.optim as optim
from .BaseNetwork import BaseNetwork
=======
>>>>>>> 1c926e4157d6f7ce8c63511c5080ceb821cefe96
import jsonschema
from .BaseNetwork import BaseNetwork

class DNNConfig():
    def __init__(self, units, dropouts):
        self.units = units
        self.dropouts = dropouts

class DNN(BaseNetwork):
<<<<<<< HEAD
    def __init__(self, name, dimensions, config, save_path=None, input_network=None, num_classes=None, 
                activation=activations.Softmax, pred_activation=activations.Softmax, optimizer=optim.Adam, 
                learning_rate=0.001, lr_scheduler=None, stopping_rule='best_validation_error', criterion=None):
=======
    def __init__(self, name, dimensions, config, save_path=None, input_network=None, num_classes=None, activation=activations.Softmax, pred_activation=activations.Softmax, optimizer=optim.Adam, learning_rate=0.001, lr_scheduler=None, stopping_rule='best_validation_error', criterion=None):
>>>>>>> 1c926e4157d6f7ce8c63511c5080ceb821cefe96
        super().__init__(name, dimensions, config, save_path, input_network, num_classes, activation, pred_activation, optimizer, learning_rate, lr_scheduler, stopping_rule, criterion)


    def _create_network(self):

        self._network = nn.sequential()

        units = self.config.units
        dropouts = self.config.dropouts

        if len(units) != len(dropouts):
            raise ValueError(
                "Cannot build network: units and dropouts don't correspond"
            )

        print("Creating {} Network...".format(self.name))
        if self.input_network is None:
            print('\tInput Layer:')
            self.input_dim = self.input_dimensions[1]
            layer = layers.InputUnit(
                              in_channels=self.input_dim,
                              out_channels=self.input_dim,
                              bias=True)
            layer_name = "{}_input".format(self.name)
            self.network.add_module(layer_name, layer)
            print('\t\t{}'.format(layer))
        else:
            for l_name, l in self.input_network['network'].network.named_children():
                self.network.add_module(l_name, l)
            layer = l
            layer_name = l_name
            self.input_dim = layer.out_channels

            print('Appending layer {} from {} to {}'.format(
                self.input_network['layer'],
                self.input_network['network'].name,
                self.name))

            #To stitch ConvLayer ---> DenseLayer
            if layer.__class__.__name__== "ConvUnit": #Src: https://discuss.pytorch.org/t/flatten-layer-of-pytorch-build-by-sequential-container/5983/3
                print("Flatening...")
                layer_name = "flatten"
                layer = layers.FlattenUnit(
                                    out_channels=self.input_dim
                                    )
                self.network.add_module(layer_name, layer)
                layer = self.network[-1]
                print('\t\t{}'.format(layer))
                self.input_dim = layer.out_features

        #if nonlinearity.__name__ == 'selu':
        #    network = torch.nn.BatchNorm2d(network)

        print('\tHidden Layer:')
        for i, (num_units, prob_dropout) in enumerate(zip(units, dropouts)):
            layer_name ="{}_dense_{}".format(self.name, i)
            layer = layers.DenseUnit(
                              in_channels=self.input_dim,
                              out_channels=num_units,
                              bias=True,
                              norm=None,
                              activation=self.nonlinearity,
                              dp=prob_dropout)
            self.network.add_module(layer_name, layer)
            print('\t\t{}'.format(layer))
            self.input_dim = layer.out_channels


        if self.num_classes is not None and self.num_classes != 0:
            self.create_classification_layer()