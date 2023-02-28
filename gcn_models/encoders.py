import sys

import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.utils import dropout_adj

sys.path.insert(0, "../")
# Dont feed in num_nodes, it makes the model not transductive
from gcn_models.conv_layers.residual_rgcn import DeepGCNResidualLayer
from gcn_models.layers import MLP, GeneralMultiLayer
from gcn_models.conv_layers.rgcn_concat import RGCNConcat


class RGCNEncoder(torch.nn.Module):
    def __init__(self, *, conv_fn, conv_aggr, in_channels, num_relations, hidden_size, num_layers, num_bases,
                 activation_fn, norm_fn, nbr_concat: bool = False, nbr_concat_weight: bool,
                 dropout, edge_dropout, res_method, pre_layers, post_layers):
        super().__init__()
        num_layers = num_layers if not nbr_concat else num_layers - 1
        self.nbr_concat = nbr_concat
        self.num_layers = num_layers
        self.dropout = dropout
        self.edge_dropout = edge_dropout
        self.no_last_layer_dropout = False
        self.model_version = 4
        if type(hidden_size) == str:
            hidden_sizes = [hidden_size] * num_layers
        else:
            hidden_sizes = hidden_size
        self.hidden_sizes = hidden_sizes
        self.activation_fn = activation_fn
        self.norm_fn = norm_fn
        self.pre_layer = GeneralMultiLayer('linear', pre_layers,
                                           in_channels, hidden_sizes[0], dim_inner=hidden_sizes[0],
                                           activation_fn=activation_fn,
                                           final_act=True)
        in_channels = hidden_sizes[0]
        convs = []
        for i in range(num_layers):
            hidden_size = hidden_sizes[i]
            conv = conv_fn(in_channels, hidden_size, num_relations, num_bases=num_bases, aggr=conv_aggr)
            norm = act = None
            if i != num_layers - 1 or post_layers > 0:
                norm = norm_fn(hidden_size)
                act = activation_fn()
            convs.append(DeepGCNResidualLayer(conv=conv, norm=norm, act=act, dropout=dropout, block=res_method))
            in_channels = hidden_size
        self.nbr_concat = nbr_concat
        if nbr_concat:
            convs.append(RGCNConcat(num_relations, aggr=conv_aggr, rel_w=nbr_concat_weight))
        self.convs = torch.nn.ModuleList(convs)
        self.post_layer = MLP(dim_in=hidden_sizes[-1], dim_out=hidden_sizes[-1], activation_fn=activation_fn,
                              num_layers=post_layers)

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def to_devices(self, devices):
        model = self
        num_convs = len(self.convs)
        if hasattr(model, 'pre_layer'):
            model.pre_layer = model.pre_layer.to(devices[0])
        for i in range(num_convs):
            model.convs[i] = model.convs[i].to(devices[i])
            if hasattr(model, 'extras'):
                model.extras[i] = model.extras[i].to(devices[i])
        if hasattr(model, 'post_layer'):
            model.post_layer = model.post_layer.to(devices[num_convs])
        return model

    def forward(self, x, adjs, edge_type_all, devices):
        nbr_concat = getattr(self, 'nbr_concat', False)
        model_version = getattr(self, 'model_version', 1)
        if hasattr(self, 'pre_layer'):
            x = self.pre_layer(x.to(devices[0]))
        for i, (edge_index, e_id, size) in enumerate(adjs):
            # print(i, edge_index.size(), size)
            if nbr_concat and i == self.num_layers:
                p = 0
            elif getattr(self, 'no_last_layer_dropout', False):
                p = 0
            else:
                p = self.edge_dropout
            edge_index, edge_type = dropout_adj(edge_index, edge_attr=edge_type_all[e_id], p=p, training=self.training)
            # print(i, edge_index.size(), size)
            x = x.to(devices[i])
            x_target = x[:size[1]]  # Target nodes are always placed first.
            x = self.convs[i]((x, x_target), edge_index.to(devices[i]), edge_type.to(devices[i])) #now x.size(0) should be = x_target.size(0)
            if not hasattr(self, 'extras'):
                if model_version < 3 and i != self.num_layers - 1:
                    x = F.relu(x)
                    x = F.dropout(x, p=self.dropout, training=self.training)
            else:
                x = self.extras[i](x)
        if hasattr(self, 'post_layer'):
            x = self.post_layer(x.to(devices[len(self.convs)]))
        return x

    @staticmethod
    def reg_loss(embedding):
        return torch.mean(embedding.pow(2))


class RGCNEncoderFull(RGCNEncoder):
    def forward(self, x, edge_index, edge_type, devices):
        assert len(devices) >= self.num_layers
        for i in range(self.num_layers):
            x = x.to(devices[i])
            x = self.convs[i](x, edge_index.to(devices[i]), edge_type.to(devices[i]))
            if i != self.num_layers - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x


class RGCNEncoderWithBERT(RGCNEncoder):
    def __init__(self, *, bert_dim, **kwargs):
        super().__init__(**kwargs)

        ie_dim = self.hidden_sizes[0] + bert_dim
        self.ie_layer = nn.Sequential(nn.Linear(ie_dim, ie_dim), self.norm_fn(ie_dim), self.activation_fn(), nn.Dropout(p=self.dropout), nn.Linear(ie_dim, ie_dim), nn.Tanh())

    def to_devices(self, devices):
        model = self
        num_convs = len(self.convs)
        if hasattr(model, 'pre_layer'):
            model.pre_layer = model.pre_layer.to(devices[0])
        for i in range(num_convs):
            model.convs[i] = model.convs[i].to(devices[i])
            if hasattr(model, 'extras'):
                model.extras[i] = model.extras[i].to(devices[i])
        if hasattr(model, 'post_layer'):
            model.post_layer = model.post_layer.to(devices[num_convs])
        model.ie_layer.to(devices[0])
        return model

    def forward(self, x, bert_x, adjs, edge_type_all, devices):
        #
        """
          x: [?, node_emb_dim]
          bert_x:      [bs, bert_dim]
        """
        _bs = bert_x.size(0)

        nbr_concat = getattr(self, 'nbr_concat', False)
        model_version = getattr(self, 'model_version', 1)
        if hasattr(self, 'pre_layer'):
            x = self.pre_layer(x.to(devices[0]))

        assert x.size(1) == self.hidden_sizes[0]
        x, bert_x = x.to(devices[0]), bert_x.to(devices[0])
        combo = torch.cat([x[:_bs], bert_x], dim=1) #[bs, gnn_dim + bert_dim]
        combo = combo + self.ie_layer(combo)        #[bs, gnn_dim + bert_dim]
        x[:_bs], bert_x = combo[:, :x.size(1)], combo[:, x.size(1):]

        for i, (edge_index, e_id, size) in enumerate(adjs):
            # print(i, edge_index.size(), size)
            if nbr_concat and i == self.num_layers:
                p = 0
            elif getattr(self, 'no_last_layer_dropout', False):
                p = 0
            else:
                p = self.edge_dropout
            edge_index, edge_type = dropout_adj(edge_index, edge_attr=edge_type_all[e_id], p=p, training=self.training)
            # print(i, edge_index.size(), size)
            x = x.to(devices[i])
            x_target = x[:size[1]]  # Target nodes are always placed first.
            x = self.convs[i]((x, x_target), edge_index.to(devices[i]), edge_type.to(devices[i])) #now x.size(0) should be = x_target.size(0)
            if not hasattr(self, 'extras'):
                if model_version < 3 and i != self.num_layers - 1:
                    x = F.relu(x)
                    x = F.dropout(x, p=self.dropout, training=self.training)
            else:
                x = self.extras[i](x)

            if nbr_concat and i == self.num_layers:
                pass
            else:
                assert x.size(1) == self.hidden_sizes[0]
                x, bert_x = x.to(devices[0]), bert_x.to(devices[0])
                combo = torch.cat([x[:_bs], bert_x], dim=1) #[bs, gnn_dim + bert_dim]
                combo = combo + self.ie_layer(combo)        #[bs, gnn_dim + bert_dim]
                x[:_bs], bert_x = combo[:, :x.size(1)], combo[:, x.size(1):]

        if hasattr(self, 'post_layer'):
            x = self.post_layer(x.to(devices[len(self.convs)]))

        x, bert_x = x.to(devices[0]), bert_x.to(devices[0])
        assert x.size(0) == _bs
        return x, bert_x
