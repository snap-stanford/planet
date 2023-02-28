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
    Write logs to checkpoint and console
    """

    args.do_train = True
    args.print_on_screen = True
    if args.do_train:
        log_file = os.path.join(args.log_dir, 'train.log')
    else:
        log_file = os.path.join(args.log_dir, 'test.log')

    logging.basicConfig(
        format='%(asctime)s %(levelname)-8s %(message)s',
        level=logging.INFO,
        datefmt='%Y-%m-%d %H:%M:%S',
        filename=log_file,
        filemode='a'
    )

    if args.print_on_screen:
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s %(levelname)-8s %(message)s')
        console.setFormatter(formatter)
        logging.getLogger('').addHandler(console)


def get_activation(args):
    if args.activation == 'relu':
        return nn.ReLU
    elif args.activation == 'leaky_relu':
        return lambda: nn.LeakyReLU(negative_slope=0.2)
    elif args.activation == 'elu':
        return nn.ELU
    elif args.activation == 'prelu':
        return nn.PReLU


def get_norm_layer(args):
    if args.norm_layer == 'batchnorm':
        return nn.BatchNorm1d
    elif args.norm_layer == 'layernorm':
        return nn.LayerNorm
    elif args.norm_layer == 'none':
        return nn.Identity


def get_conv_layer(args):
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
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
