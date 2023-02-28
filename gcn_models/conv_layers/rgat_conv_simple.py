import math
from typing import Optional, Union, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Parameter, Linear
from torch.nn import Parameter as Param
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.typing import OptTensor, Adj
from torch_geometric.utils import softmax
from torch_sparse import SparseTensor, masked_select_nnz


def uniform(size, tensor):
    if tensor is not None:
        bound = 1.0 / math.sqrt(size)
        tensor.data.uniform_(-bound, bound)


def zeros(tensor):
    if tensor is not None:
        tensor.data.fill_(0)


def glorot(tensor):
    if tensor is not None:
        stdv = math.sqrt(6.0 / (tensor.size(-2) + tensor.size(-1)))
        tensor.data.uniform_(-stdv, stdv)


@torch.jit._overload
def masked_edge_index(edge_index, edge_mask):
    # type: (Tensor, Tensor) -> Tensor
    pass


@torch.jit._overload
def masked_edge_index(edge_index, edge_mask):
    # type: (SparseTensor, Tensor) -> SparseTensor
    pass


def masked_edge_index(edge_index, edge_mask):
    if isinstance(edge_index, Tensor):
        return edge_index[:, edge_mask]
    else:
        return masked_select_nnz(edge_index, edge_mask, layout='coo')


class RGATConvSimple(MessagePassing):
    r"""The relational graph convolutional operator from the `"Modeling
    Relational Data with Graph Convolutional Networks"
    <https://arxiv.org/abs/1703.06103>`_ paper

    .. math::
        \mathbf{x}^{\prime}_i = \mathbf{\Theta}_{\textrm{root}} \cdot
        \mathbf{x}_i + \sum_{r \in \mathcal{R}} \sum_{j \in \mathcal{N}_r(i)}
        \frac{1}{|\mathcal{N}_r(i)|} \mathbf{\Theta}_r \cdot \mathbf{x}_j,

    where :math:`\mathcal{R}` denotes the set of relations, *i.e.* edge types.
    Edge type needs to be a one-dimensional :obj:`torch.long` tensor which
    stores a relation identifier
    :math:`\in \{ 0, \ldots, |\mathcal{R}| - 1\}` for each edge.

    .. note::
        This implementation is as memory-efficient as possible by iterating
        over each individual relation type.
        Therefore, it may result in low GPU utilization in case the graph has a
        large number of relations.
        As an alternative approach, :class:`FastRGCNConv` does not iterate over
        each individual type, but may consume a large amount of memory to
        compensate.
        We advise to check out both implementations to see which one fits your
        needs.

    Args:
        in_channels (int or tuple): Size of each input sample. A tuple
            corresponds to the sizes of source and target dimensionalities.
            In case no input features are given, this argument should
            correspond to the number of nodes in your graph.
        out_channels (int): Size of each output sample.
        num_relations (int): Number of relations.
        num_bases (int, optional): If set to not :obj:`None`, this layer will
            use the basis-decomposition regularization scheme where
            :obj:`num_bases` denotes the number of bases to use.
            (default: :obj:`None`)
        num_blocks (int, optional): If set to not :obj:`None`, this layer will
            use the block-diagonal-decomposition regularization scheme where
            :obj:`num_blocks` denotes the number of blocks to use.
            (default: :obj:`None`)
        aggr (string, optional): The aggregation scheme to use
            (:obj:`"add"`, :obj:`"mean"`, :obj:`"max"`).
            (default: :obj:`"mean"`)
        root_weight (bool, optional): If set to :obj:`False`, the layer will
            not add transformed root node features to the output.
            (default: :obj:`True`)
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.
    """

    def __init__(self, in_channels: Union[int, Tuple[int, int]],
                 out_channels: int,
                 num_relations: int,
                 num_bases: Optional[int] = None,
                 aggr: str = 'mean',
                 root_weight: bool = True,
                 heads: int = 1,
                 concat: bool = False,
                 scaled_attention: bool = False,
                 attention_type: str = 'dot',
                 bias: bool = True, **kwargs):  # yapf: disable

        super(RGATConvSimple, self).__init__(aggr=aggr, node_dim=0, **kwargs)

        if concat:
            out_channels = out_channels // heads
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_relations = num_relations
        self.num_bases = num_bases
        self.heads = heads
        self.concat = concat
        self.attention_type = attention_type
        self.scaled_attention = scaled_attention

        if isinstance(in_channels, int):
            self.lin_l = Linear(in_channels, heads * out_channels, bias=False)
            self.lin_r = self.lin_l
        else:
            self.lin_l = Linear(in_channels[0], heads * out_channels, False)
            self.lin_r = Linear(in_channels[1], heads * out_channels, False)

        if self.attention_type == 'additive':
            self.negative_slope = 0.2
            self.att_l = Parameter(torch.Tensor(1, heads, out_channels))
            self.att_r = Parameter(torch.Tensor(1, heads, out_channels))
        else:
            self.register_parameter('att_l', None)
            self.register_parameter('att_r', None)

        if isinstance(in_channels, int):
            in_channels = (in_channels, in_channels)
        self.in_channels_l = in_channels[0]

        if num_bases is not None:
            self.weight = Parameter(
                torch.Tensor(num_bases, in_channels[0], heads * out_channels))
            self.comp = Parameter(torch.Tensor(num_relations, num_bases))

        else:
            self.weight = Parameter(
                torch.Tensor(num_relations, in_channels[0], heads * out_channels))
            self.register_parameter('comp', None)

        if root_weight:
            c = out_channels
            if concat:
                c = heads * out_channels
            self.root = Param(torch.Tensor(in_channels[1], c))
        else:
            self.register_parameter('root', None)

        if bias:
            c = out_channels
            if concat:
                c = heads * out_channels
            self.bias = Param(torch.Tensor(c))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.lin_l.weight)
        glorot(self.lin_r.weight)
        glorot(self.att_l)
        glorot(self.att_r)
        glorot(self.weight)
        glorot(self.comp)
        glorot(self.root)
        zeros(self.bias)

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
        H, C = self.heads, self.out_channels
        x_l: OptTensor = None
        if isinstance(x, tuple):
            x_l = x[0]
        else:
            x_l = x
        if x_l is None:
            x_l = torch.arange(self.in_channels_l, device=self.weight.device)

        x_r: Tensor = x_l
        if isinstance(x, tuple):
            x_r = x[1]

        size = (x_l.size(0), x_r.size(0))

        if isinstance(edge_index, SparseTensor):
            edge_type = edge_index.storage.value()
        assert edge_type is not None

        # propagate_type: (x: Tensor)
        out = torch.zeros(x_r.size(0), self.heads, self.out_channels, device=x_r.device)

        alpha_l = self.lin_l(x_l).view(-1, H, C)
        alpha_r = self.lin_r(x_r).view(-1, H, C)

        weight = self.weight
        if self.num_bases is not None:  # Basis-decomposition =================
            weight = (self.comp @ weight.view(self.num_bases, -1)).view(
                self.num_relations, self.in_channels_l, self.heads * self.out_channels)

        # No regularization/Basis-decomposition ========================
        for i in range(self.num_relations):
            tmp = masked_edge_index(edge_index, edge_type == i)
            if x_l.dtype == torch.long:
                x_i = weight[i, x_l]
            else:
                x_i = x_l @ weight[i]
            # print(f"{i}: Before propagate: {tmp.size()}", size, x_i.size(), alpha_r.size(), alpha_l.size())
            if tmp.size(1) == 0:
                continue
            out += self.propagate(tmp, x=x_i.view(-1, H, C), alpha=(alpha_l, alpha_r), size=size)

        if self.concat:
            out = out.view(-1, H * C)
        else:
            out = out.mean(dim=1)

        root = self.root
        if root is not None:
            out += root[x_r] if x_r.dtype == torch.long else x_r @ root

        if self.bias is not None:
            out += self.bias

        return out

    def message(self, x_j: Tensor, alpha_j: Tensor, alpha_i: Tensor, index: Tensor, ptr: OptTensor,
                size_i: Optional[int]) -> Tensor:
        if not hasattr(self, 'attention_type'):
            self.attention_type = 'dot'
        if not hasattr(self, 'scaled_attention'):
            self.scaled_attention = False
        if self.attention_type == 'dot':
            e = (alpha_i * alpha_j).sum(dim=-1)
            if self.scaled_attention:
                e = e / math.sqrt(self.out_channels)
        elif self.attention_type == 'additive':
            e_l = (alpha_i * self.att_l).sum(dim=-1)
            e_r = (alpha_j * self.att_r).sum(dim=-1)
            e = e_l + e_r
            e = F.leaky_relu(e, self.negative_slope)
        alpha = softmax(e, index, ptr, size_i)

        return x_j * alpha.unsqueeze(-1)

    def __repr__(self):
        if not hasattr(self, 'heads'):
            self.heads = 1
        if not hasattr(self, 'concat'):
            self.concat = True
        return '{}({}, {}, heads={}, num_relations={})'.format(self.__class__.__name__,
                                                               self.in_channels,
                                                               self.out_channels,
                                                               self.heads,
                                                               self.num_relations)
