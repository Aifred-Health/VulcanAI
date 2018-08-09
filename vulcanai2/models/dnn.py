__author__ = 'Caitrin'
__author__ = 'Caitrin'
import torch.nn as nn
import torch.nn.functional as F
from AbstractNetwork import AbstractNetwork
import jsonschema
import schemas

class DNN(AbstractNetwork):
    """Class to generate networks and train them."""
    def __init__(self, name, dimensions, config, save_path=None, input_network=None, num_classes=None, activation=activations.Softmax, pred_activation=activations.Softmax, optimizer=optim.Adam, learning_rate=0.001, lr_scheduler=None, stopping_rule='best_validation_error', criterion=None):
        pass #TODO: use super init, but with **kwargs?


    def create_network(self, config, nonlinearity):

        jsonschema.validate(config, schemas.dense_network)

        self.create_dense_network(
            units=config.get('units'),
            dropouts=config.get('dropouts'),
            nonlinearity=nonlinearity
        )

        if self.num_classes is not None and self.num_classes != 0:
            self.create_classification_layer(
                num_classes=self.num_classes,
                nonlinearity=self.pred_activation
            )


    def create_dense_network(self, units, dropouts, nonlinearity):
        """
        Generate a fully connected layer.
        Args:
            units: The list of number of nodes to have at each layer
            dropouts: The list of dropout probabilities for each layer
            nonlinearity: Nonlinearity from Lasagne.nonlinearities
        Returns: the output of the network (linked up to all the layers)
        """
        if len(units) != len(dropouts):
            raise ValueError(
                "Cannot build network: units and dropouts don't correspond"
            )

        print("Creating {} Network...".format(self.name))
        if self.input_network is None:
            print('\tInput Layer:')
            self.input_dim = self.input_dimensions[1]
            layer = InputUnit(
                              in_channels=self.input_dim,
                              out_channels=self.input_dim,
                              bias=True)
            layer_name = "{}_input".format(self.name)
            self.network.add_module(layer_name, layer)
            print('\t\t{}'.format(layer))
            self.layers.append(layer)
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
                layer = FlattenUnit(
                                    out_channels=self.input_dim
                                    )
                self.network.add_module(layer_name, layer)
                layer = self.network[-1]
                self.layers.append(layer)
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
                              bias=True,
                              norm=None,
                              activation=nonlinearity,
                              dp=prob_dropout)
            self.network.add_module(layer_name, layer)
            self.layers.append(layer)
            print('\t\t{}'.format(layer))
            self.input_dim = layer.out_channels

