from typing import Union, Tuple

import torch
from torch import Tensor
from torch_geometric.nn import MessagePassing
from torch_geometric.nn.conv.rgcn_conv import masked_edge_index
from torch_geometric.typing import OptTensor, Adj
from torch_sparse import SparseTensor


class RGCNConcat(MessagePassing):
    def __init__(self, num_relations: int, rel_w: bool= False, aggr: str = 'mean', **kwargs):
        super(RGCNConcat, self).__init__(aggr=aggr, node_dim=0, **kwargs)
        self.num_relations = num_relations
        self.relation_weights = rel_w

        if self.relation_weights:
            self.rel_w = torch.nn.Parameter(torch.ones(self.num_relations, dtype=torch.float32))

    def forward(self, x: Union[OptTensor, Tuple[OptTensor, Tensor]],
                edge_index: Adj, edge_type: OptTensor = None):
        r"""
        Args:
            x: The input node features. Can be either a :obj:`[num_nodes,
                in_channels]` node feature matrix, or an optional
                one-dimensional node index tensor (in which case input features
                are treated as trainable node embeddings).
                Furthermore, :obj:`x` can be of type :obj:`tuple` denoting
                source and destination node features.
            edge_type: The one-dimensional relation type/index for each edge in
                :obj:`edge_index`.
                Should be only :obj:`None` in case :obj:`edge_index` is of type
                :class:`torch_sparse.tensor.SparseTensor`.
                (default: :obj:`None`)
        """

        # Convert input features to a pair of node features or node indices.
        x_l: OptTensor = None
        if isinstance(x, tuple):
            x_l = x[0]
        else:
            x_l = x

        x_r: Tensor = x_l
        if isinstance(x, tuple):
            x_r = x[1]

        size = (x_l.size(0), x_r.size(0))

        if isinstance(edge_index, SparseTensor):
            edge_type = edge_index.storage.value()
        assert edge_type is not None

        out = []

        for i in range(self.num_relations):
            tmp = masked_edge_index(edge_index, edge_type == i)
            if tmp.size(1) == 0:
                continue
            else:

                h = self.propagate(tmp, x=x_l, size=size) #average the neighbor node message of the same relation type
                if self.relation_weights:
                    h = h * self.rel_w[i]
                # out.append((h @ weight[i]))
                out.append(h)

        out.append(x_r)
        if getattr(self, 'relation_weights', False):
            return torch.stack(out, dim=-1).sum(dim=-1)
        else:
            return torch.cat(out, dim=-1)

    def message(self, x_j: Tensor) -> Tensor:
        return x_j #message is simply the hidden state of the node

    def __repr__(self):
        return '{}(num_relations={})'.format(self.__class__.__name__,
                                             self.num_relations)
