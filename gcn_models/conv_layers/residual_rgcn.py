from torch import nn


class DeepGCNResidualLayer(nn.Module):
    def __init__(self, conv, norm=None, act=None, block='res+',
                 dropout=0.):
        super(DeepGCNResidualLayer, self).__init__()

        self.conv = conv
        self.norm = norm or nn.Identity()
        self.act = act or nn.Identity()
        self.block = block.lower()
        assert self.block in ['res+', 'res', 'dense', 'plain']
        self.dropout = nn.Dropout(p=dropout)

    def reset_parameters(self):
        self.conv.reset_parameters()
        if hasattr(self.norm, 'reset_parameters'):
            self.norm.reset_parameters()
        if hasattr(self.act, 'reset_parameters'):
            self.act.reset_parameters()

    def forward(self, x, edge_index, edge_type=None):
        """"""
        if isinstance(x, tuple):
            x_l, x_r = x
        else:
            x_l, x_r = x, x
        if self.block == 'res+':
            h_l = self.dropout(self.act(self.norm(x_l)))
            h_r = self.dropout(self.act(self.norm(x_r)))
            h = self.conv((h_l, h_r), edge_index, edge_type)

            return x_r + h

        else:
            h = self.conv((x_l, x_r), edge_index, edge_type)
            h = self.dropout(self.norm(h))
            if self.block == 'res':
                h = x_r + h
            elif self.block == 'plain':
                pass
            return self.act(h)

    def __repr__(self):
        if self.block == 'res+':
            return f'{self.__class__.__name__}({self.norm}, {self.act}, {self.conv}, {self.dropout}, Residual)'
        elif self.block == 'res':
            return f'{self.__class__.__name__}({self.conv}, {self.norm}, {self.dropout}, Residual, {self.act})'
        elif self.block == 'plain':
            return f'{self.__class__.__name__}({self.conv}, {self.norm}, {self.dropout}, {self.act})'
