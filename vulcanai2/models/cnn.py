__author__ = 'Caitrin'
import torch.nn as nn
import torch.nn.functional as F



class CNN(object):
    """Class to generate networks and train them."""

    def __init__(self, name, dimensions, input_var, y, config,
                 input_network=None, num_classes=None, activation='rectify',
                 pred_activation='softmax', optimizer='adam', stopping_rule='best_validation_error',
                 learning_rate=0.001):

        """
        Initialize network specified.
        Args:
            name: string of network name
            dimensions: the size of the input data matrix
            input_var: theano tensor representing input matrix
            y: theano tensor representing truth matrix
            config: Network configuration (as dict)
            input_network: None or a dictionary containing keys (network, layer).
                network: a Network object
                layer: an integer corresponding to the layer you want output
            num_classes: None or int. how many classes to predict
            activation:  activation function for hidden layers
            pred_activation: the classifying layer activation
            optimizer: which optimizer to use as the learning function
            learning_rate: the initial learning rate
        """
        self.name = name
        self.layers = []
        self.cost = None
        self.val_cost = None
        self.input_dimensions = dimensions
        self.config = config
        self.learning_rate = learning_rate
        self.init_learning_rate = learning_rate
        self.stopping_rule = stopping_rule
        if not optimizers.get(optimizer, False):
            raise ValueError(
                'Invalid optimizer option: {}. '
                'Please choose from:'
                '{}'.format(optimizer, optimizers.keys()))
        if not activations.get(activation, False) or \
           not activations.get(pred_activation, False):
            raise ValueError(
                'Invalid activation option: {} and {}. '
                'Please choose from:'
                '{}'.format(activation, pred_activation, activations.keys()))
        self.activation = activation
        self.pred_activation = pred_activation
        self.optimizer = optimizer
        self.input_var = input_var
        self.y = y
        self.input_network = input_network
        self.input_params = None
        if self.input_network is not None:
            if self.input_network.get('network', False) is not False and \
               self.input_network.get('layer', False) is not False and \
               self.input_network.get('get_params', None) is not None:

                #self.input_var = lasagne.layers.get_all_layers(
                #    self.input_network['network']
                #)[0].input_var

                for l_name, l in self.input_network['network'].network.named_children():
                    if isinstance(l, nn.Sequential):
                        for subl_name, subl in l.named_children():
                            for param in subl.parameters():
                                self.input_dimensions= param.size(0)
                    else:
                        for param in l.parameters():
                            self.input_dimensions= param.size(0)

                #if self.input_network.get('get_params', False):
                #    self.input_params = self.input_network['network'].params

            else:
                raise ValueError(
                    'input_network for {} requires {{ network: type Network,'
                    ' layer: type int, get_params: type bool}}. '
                    'Only given keys: {}'.format(
                        self.name, self.input_network.keys()
                    )
                )
        self.num_classes = num_classes
        self.network = nn.Sequential()
        self.create_network(
            config=self.config,
            nonlinearity=activations[self.activation]
        )

    def get_all_layers(self):
        layers = []
        for l_name, l in self.input_network['network'].network.named_children():
            if isinstance(l, nn.Sequential):
                for subl_name, subl in l.named_children():
                    layers.append(subl)
            else:
                for param in l.parameters():
                    self.input_dimensions= param.size(0)



    def create_network(self, config, nonlinearity):
        """
        Abstract function to create any network given a config dict.
        Args:
            config: dict. the network configuration
            nonlinearity: string. the nonlinearity to add onto each layer
        returns a network.
        """
        import jsonschema
        import schemas

        mode = config.get('mode')
        if mode == 'dense':
            jsonschema.validate(config, schemas.dense_network)

            self.create_dense_network(
                units=config.get('units'),
                dropouts=config.get('dropouts'),
                nonlinearity=nonlinearity
            )
        elif mode == 'conv':
            jsonschema.validate(config, schemas.conv_network)

            self.create_conv_network(
                filters=config.get('filters'),
                filter_size=config.get('filter_size'),
                stride=config.get('stride'),
                pool_mode=config['pool'].get('mode'),
                pool_stride=config['pool'].get('stride'),
                nonlinearity=nonlinearity
            )
        else:
            raise ValueError('Mode {} not supported.'.format(mode))

        if self.num_classes is not None and self.num_classes != 0:
            self.create_classification_layer(
                num_classes=self.num_classes,
                nonlinearity=activations[self.pred_activation]
            )

    def create_conv_network(self, filters, filter_size, stride,
                            pool_mode, pool_stride, nonlinearity):
        """
        Create a convolutional network (1D, 2D, or 3D).
        Args:
            filters: list of int. number of kernels per layer
            filter_size: list of int list. size of kernels per layer
            stride: list of int list. stride of kernels
            pool_mode: string. pooling operation
            pool_stride: list of int list. down_scaling factor
            nonlinearity: string. nonlinearity to use for each layer
        Returns a conv network
        """
        conv_dim = len(filter_size[0])
        lasagne_pools = ['max', 'average_inc_pad', 'average_exc_pad']
        if not all(len(f) == conv_dim for f in filter_size):
            raise ValueError('Each tuple in filter_size {} must have a '
                             'length of {}'.format(filter_size, conv_dim))
        if not all(len(s) == conv_dim for s in stride):
            raise ValueError('Each tuple in stride {} must have a '
                             'length of {}'.format(stride, conv_dim))
        if not all(len(p) == conv_dim for p in pool_stride):
            raise ValueError('Each tuple in pool_stride {} must have a '
                             'length of {}'.format(pool_stride, conv_dim))
        if pool_mode not in lasagne_pools:
            raise ValueError('{} pooling does not exist. '
                             'Please use one of: {}'.format(pool_mode, lasagne_pools))

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

        print('\tHidden Layer:')
        for i, (f, f_size, s, p_s) in enumerate(zip(filters,
                                                    filter_size,
                                                    stride,
                                                    pool_stride)):
            layer_name = "{}_conv{}D_{}".format(
                                    self.name, conv_dim, i)
            layer = ConvUnit(
                            conv_dim=conv_dim,
                            in_channels=self.input_dim,
                            out_channels=f,
                            kernel_size=f_size,
                            stride=s,
                            pool_size=p_s,
                            activation=nonlinearity)
            self.network.add_module(layer_name, layer)
            self.layers.append(layer)
            print('\t\t{}'.format(layer))
            self.input_dim = layer.out_channels


    def train(self, epochs, train_loader, test_loader, criterion, optimizer, change_rate=None, use_gpu=False, engine="base"):

        engine.train(epochs, t)


        #TODO: I don't think this should actually be here
