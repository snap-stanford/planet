import logging
import os
import random
import sys
from functools import partial

import numpy as np
import torch
from torch import nn

sys.path.insert(0, "../")

from torch_geometric.nn import RGCNConv
# from gcn_models.conv_layers.rgcn_conv import RGCNConv
from gcn_models.conv_layers.rgat_conv import RGATConv
from gcn_models.conv_layers.rgat_conv_simple import RGATConvSimple
from gcn_models.conv_layers.hgt_conv import HGTConv

def set_logger(args):
    """
    Configures logging to write logs to both a file and the console.
    
    Args:
        args: An argument object containing logging configurations such as log directory.
    """
    args.do_train = True  # Ensure training mode is enabled
    args.print_on_screen = True  # Enable console output
    
    # Determine log file based on training mode
    log_file = os.path.join(args.log_dir, 'train.log' if args.do_train else 'test.log')
    
    # Set up logging configuration
    logging.basicConfig(
        format='%(asctime)s %(levelname)-8s %(message)s',
        level=logging.INFO,
        datefmt='%Y-%m-%d %H:%M:%S',
        filename=log_file,
        filemode='a'
    )
    
    # If enabled, also log to the console
    if args.print_on_screen:
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s %(levelname)-8s %(message)s')
        console.setFormatter(formatter)
        logging.getLogger('').addHandler(console)

def get_activation(args):
    """
    Returns the activation function based on the argument.
    
    Args:
        args: An argument object containing the activation function type.
    
    Returns:
        A PyTorch activation function class or lambda function.
    """
    if args.activation == 'relu':
        return nn.ReLU
    elif args.activation == 'leaky_relu':
        return lambda: nn.LeakyReLU(negative_slope=0.2)
    elif args.activation == 'elu':
        return nn.ELU
    elif args.activation == 'prelu':
        return nn.PReLU

def get_norm_layer(args):
    """
    Returns the normalization layer based on the argument.
    
    Args:
        args: An argument object containing the normalization layer type.
    
    Returns:
        A PyTorch normalization layer class.
    """
    if args.norm_layer == 'batchnorm':
        return nn.BatchNorm1d
    elif args.norm_layer == 'layernorm':
        return nn.LayerNorm
    elif args.norm_layer == 'none':
        return nn.Identity

def get_conv_layer(args):
    """
    Returns the graph convolutional layer based on the argument.
    
    Args:
        args: An argument object containing the convolution type.
    
    Returns:
        A PyTorch Geometric convolutional layer class or partial function.
    
    Raises:
        RuntimeError: If an invalid convolution type is specified.
    """
    if args.conv_type == 'rgcn':
        conv_fn = RGCNConv
    elif args.conv_type == 'hgt':
        conv_fn = partial(HGTConv, num_types=1, n_heads=8, use_RTE=False)
    elif args.conv_type == 'rgat':
        conv_fn = RGATConv
    elif args.conv_type == 'rgat_simple':
        conv_fn = partial(RGATConvSimple, heads=args.num_heads, concat=args.concat_gat,
                          scaled_attention=args.scaled_attention, attention_type=args.attention_type)
    elif args.conv_type == 'rgat_same_w':
        conv_fn = partial(RGATConv, source_target_same_w=True)
    else:
        raise RuntimeError(f"Not a valid conv type: {args.conv_type}")
    return conv_fn

def set_seed(seed):
    """
    Sets the random seed for reproducibility.
    
    Args:
        seed (int): The seed value to be used for random number generation.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
