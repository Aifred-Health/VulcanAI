import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from .baseNetwork import BaseNetwork
from .layers import InputUnit, DenseUnit, ConvUnit, FlattenUnit
import jsonschema

#TODO: perform typechecking

class DNNConfig():
    def __init__(self, units, dropouts):
        self.units = units
        self.dropouts = dropouts

class DNN(BaseNetwork, nn.Module):

    def __init__(self, name, dimensions, config, save_path=None, input_network=None, num_classes=None, 
                activation=nn.Softmax(dim=1), pred_activation=nn.Softmax(dim=1), optim_spec={'name': 'Adam', 'lr': 0.001},
                lr_scheduler=None, stopping_rule='best_validation_error', criter_spec={'name': 'CrossEntropyLoss'}):
        
        nn.Module.__init__(self)
        super(DNN, self).__init__(name, dimensions, config, save_path, input_network, num_classes, 
                activation, pred_activation, optim_spec, lr_scheduler, stopping_rule, criter_spec)

        self._scheduler = None # NOTE: Temporary Assignment - Priya

    def _create_network(self):

        units = self._config.get('units')
        dropouts = self._config.get('dropouts')

        if len(units) != len(dropouts):
            raise ValueError(
                "Cannot build network: units and dropouts don't correspond"
            )

        print("Creating {} Network...".format(self.name))
        if self._input_network is None:
            print('\tInput Layer:')
            self.input_dim = self._dimensions[1]
            layer = InputUnit(
                              in_channels=self.input_dim,
                              out_channels=self.input_dim,
                              bias=True)
            layer_name = "{}_input".format(self.name)
            self._network.add_module(layer_name, layer)
            print('\t\t{}'.format(layer))
        else:
            for l_name, l in self._input_network['network'].network.named_children():
                self._network.add_module(l_name, l)
            layer = l
            layer_name = l_name
            self.input_dim = layer.out_channels

            print('Appending layer {} from {} to {}'.format(
                self._input_network['layer'],
                self._input_network['network'].name,
                self.name))

            #To stitch ConvLayer ---> DenseLayer
            if layer.__class__.__name__== "ConvUnit": #Src: https://discuss.pytorch.org/t/flatten-layer-of-pytorch-build-by-sequential-container/5983/3
                print("Flatening...")
                layer_name = "flatten"
                layer = FlattenUnit(
                                    out_channels=self.input_dim
                                    )
                self._network.add_module(layer_name, layer)
                layer = self._network[-1]
                print('\t\t{}'.format(layer))
                self.input_dim = layer.out_features

        #if nonlinearity.__name__ == 'selu':
        #    network = torch.nn.BatchNorm2d(network)

        print('\tHidden Layer:')
        for i, (num_units, prob_dropout) in enumerate(zip(units, dropouts)):
            layer_name ="{}_dense_{}".format(self.name, i)
            layer = DenseUnit(
                              in_channels=self.input_dim,
                              out_channels=num_units,
                              norm=None,
                              activation=self._activation,
                              dp=prob_dropout)
            self._network.add_module(layer_name, layer)
            print('\t\t{}'.format(layer))
            self.input_dim = layer.out_channels


        if self._num_classes is not None and self._num_classes != 0:
            self.create_classification_layer()