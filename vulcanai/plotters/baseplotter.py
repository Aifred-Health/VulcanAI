# -*- coding: utf-8 -*-
"""Defines the Graph class."""

# Core imports
from graphviz import Digraph
import torch
from torch.autograd import Variable

# Generic imports
from collections import defaultdict

THEMES = {
    "basic": {
        "background_color": "#FFFFFF",
        "fill_color": "#E8E8E8",
        "outline_color": "#000000",
        "font_color": "#000000",
        "font_name": "Times",
        "font_size": "10",
        "margin": "0,0",
        "padding":  "1.0,0.5",
    },
    "blue": {
        "background_color": "#FFFFFF",
        "fill_color": "#BCD6FC",
        "outline_color": "#7C96BC",
        "font_color": "#202020",
        "font_name": "Verdana",
        "font_size": "10",
        "margin": "0,0",
        "padding":  "1.0,0.5",
    },
}


class Graph(object):
    """ 
    Builds the graph (dot) for the BaseNetwork model and
    saves the graph source code (.gv) in the current directory.
    #TODO: Modify the saving directory.
    
    Parameters:
        model : BaseNetwork
            The model for which graph is built.
        input : list of torch.Tensor or torch.Tensor
            The input data to the BaseNetwork. Must be a list if the model is
            a multi-input type BaseNetwork.
        theme_color : dict
            The color schema for the model's graph.
        node_shape : str
            Sets the shape of the graph nodes.

    Example:

        >>> graph = Graph(model, x, theme_color='blue', node_shape='oval')
        >>> graph.view() # Saves a .pdf of the graph.dot object and opens the saved file
        using the pdf viewer

    """

    def __init__(self, model, input, theme_color='basic',
                 node_shape='ellipse'):
        self.model = model
        self.input = input
        self.params = dict(model.named_parameters())

        self.layer_idx = 0

        self.node_dict = defaultdict(Node)
        self.theme = THEMES[theme_color]
        self.node_shape = node_shape

        self.dot = Digraph()
        self.dot.attr("graph", 
                 bgcolor=self.theme["background_color"],
                 color=self.theme["outline_color"],
                 fontsize=self.theme["font_size"],
                 fontcolor=self.theme["font_color"],
                 fontname=self.theme["font_name"],
                 margin=self.theme["margin"],
                 pad=self.theme["padding"])
        self.dot.attr("node", shape=self.node_shape, 
                 style="filled", margin="0,0",
                 fillcolor=self.theme["fill_color"],
                 color=self.theme["outline_color"],
                 fontsize=self.theme["font_size"],
                 fontcolor=self.theme["font_color"],
                 fontname=self.theme["font_name"])
        self.dot.attr("edge", style="solid", 
                 color=self.theme["outline_color"],
                 fontsize=self.theme["font_size"],
                 fontcolor=self.theme["font_color"],
                 fontname=self.theme["font_name"])

        self.seen = set() # control depict
        self.mem_id_to_cus_id = dict() # control recursive

        if self.params is not None:
            assert all(isinstance(p, Variable) for p in self.params.values())
            self.param_map = {id(v): k for k, v in self.params.items()}
        
        self.make_dot()

    def size_to_str(self, size):
        return '\n(' + (', ').join(['%d' % v for v in size]) + ')'

    def make_dot(self):
        """ 
        Generates the graphviz object.
        """
        var = self.model(self.input)
        if isinstance(var, tuple):
            var = var[0]
        self._add_nodes(var.grad_fn)

    def _add_nodes(self, var, parent_id=None):

        cur_id = None

        if var not in self.seen:
            # add current node
            if torch.is_tensor(var):
                self.dot.node(str(id(var)), 
                              self.size_to_str(var.size()), 
                              fillcolor='orange')
            elif hasattr(var, 'variable'):
                u = var.variable
                name = self.param_map[id(u)] if self.params is not None else ''
                node_name = '%s\n %s' % (name, self.size_to_str(u.size()))
                self.dot.node(str(id(var)), node_name, fillcolor='lightblue')
            else:
                cur_id = str(type(var).__name__) + str(self.layer_idx)
                self.dot.node(str(id(var)), cur_id)
                ### add layer info
                # id
                self.node_dict[cur_id].id = cur_id
                self.mem_id_to_cus_id[str(id(var))] = cur_id
                # type
                self.node_dict[cur_id].type = str(type(var).__name__)
                # parent
                if (parent_id is not None) and (parent_id is not ''):
                    self.node_dict[cur_id].parents.append(parent_id)
                self.layer_idx += 1

            self.seen.add(var)

            # visit children
            if hasattr(var, 'next_functions'):
                # obtain parameter shape
                for u in var.next_functions:
                    if (u[0] is not None) and (torch.is_tensor(u[0]) == False)\
                        and (hasattr(u[0], 'variable')):
                        assert cur_id is not None, 'bug'
                        self.node_dict[cur_id].param_shapes.append(u[0].variable.size())
                # obtain child_id
                for u in var.next_functions:
                    if u[0] is not None:
                        # connect with current node
                        self.dot.edge(str(id(u[0])), str(id(var)))
                        # append children id
                        if (torch.is_tensor(u[0]) == False) and\
                            (hasattr(u[0], 'variable') == False):
                            if u[0] not in self.seen:
                                child_id = str(type(u[0]).__name__) + str(self.layer_idx)
                            else:
                                child_id = self.mem_id_to_cus_id[str(id(u[0]))]
                            assert cur_id is not None, 'bug'
                            self.node_dict[cur_id].children.append(child_id)

                        self._add_nodes(var=u[0], parent_id=cur_id)

            if hasattr(var, 'saved_tensors'):
                for t in var.saved_tensors:
                    self.dot.edge(str(id(t)), str(id(var)))
                    self._add_nodes(t)
        else:
            if (torch.is_tensor(var) == False) and\
                (hasattr(var, 'variable') == False):
                cur_id = self.mem_id_to_cus_id[str(id(var))]
                ## add layer info
                assert (parent_id is not None) and (parent_id is not '')
                # parent
                if (parent_id is not None) and (parent_id is not ''):
                    self.node_dict[cur_id].parents.append(parent_id)

    def view(self):
        """
        Saves the source code of self.dot to file and opens the
        rendered result in a viewer.
        """
        self.dot.view()

    def __getitem__(self, key):
        if isinstance(key, list):
            return [self.node_dict.get(k) for k in key]
        else:
            return self.node_dict.get(key)


class Node(object):
    """ 
    Node object for the Graph.

    """
    def __init__(self):
        self.parents = list()
        self.children = list()
        self.param_shapes = list()
        self.input_shapes = list()
        self.output_shapes = list()
        self.id = ''
        self.type = ''