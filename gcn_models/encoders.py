import sys

import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.utils import dropout_adj

sys.path.insert(0, "../")
# Don't feed in num_nodes, it makes the model not transductive
from gcn_models.conv_layers.residual_rgcn import DeepGCNResidualLayer
from gcn_models.layers import MLP, GeneralMultiLayer
from gcn_models.conv_layers.rgcn_concat import RGCNConcat


class RGCNEncoder(torch.nn.Module):
    """
    RGCNEncoder implements a Relational Graph Convolutional Network encoder.

    Args:
        conv_fn (callable): Function to create a convolution layer.
        conv_aggr (str): Aggregation method for the convolution.
        in_channels (int): Number of input features.
        num_relations (int): Number of edge types or relations.
        hidden_size (int or str): Hidden size (or specification) for each layer.
        num_layers (int): Total number of layers.
        num_bases (int): Number of bases to use in the convolution.
        activation_fn (callable): Activation function to use.
        norm_fn (callable): Normalization function constructor.
        nbr_concat (bool, optional): Flag to indicate neighbor concatenation. Defaults to False.
        nbr_concat_weight (bool): Whether to use weights for neighbor concatenation.
        dropout (float): Dropout rate for node features.
        edge_dropout (float): Dropout rate for edges.
        res_method (str): Residual connection method.
        pre_layers (int): Number of pre-processing layers.
        post_layers (int): Number of post-processing layers.
    """
    def __init__(self, *, conv_fn, conv_aggr, in_channels, num_relations, hidden_size, num_layers, num_bases,
                 activation_fn, norm_fn, nbr_concat: bool = False, nbr_concat_weight: bool,
                 dropout, edge_dropout, res_method, pre_layers, post_layers):
        super().__init__()
        # Adjust number of layers if neighbor concatenation is enabled.
        num_layers = num_layers if not nbr_concat else num_layers - 1
        self.nbr_concat = nbr_concat
        self.num_layers = num_layers
        self.dropout = dropout
        self.edge_dropout = edge_dropout
        self.no_last_layer_dropout = False
        self.model_version = 4
        
        # Determine hidden sizes per layer.
        if type(hidden_size) == str:
            hidden_sizes = [hidden_size] * num_layers
        else:
            hidden_sizes = hidden_size
        self.hidden_sizes = hidden_sizes
        self.activation_fn = activation_fn
        self.norm_fn = norm_fn
        
        # Create a pre-processing layer to map input features to first hidden layer dimension.
        self.pre_layer = GeneralMultiLayer('linear', pre_layers,
                                           in_channels, hidden_sizes[0], dim_inner=hidden_sizes[0],
                                           activation_fn=activation_fn,
                                           final_act=True)
        in_channels = hidden_sizes[0]
        convs = []
        # Build convolutional layers.
        for i in range(num_layers):
            hidden_size = hidden_sizes[i]
            # Initialize the convolution layer for current layer.
            conv = conv_fn(in_channels, hidden_size, num_relations, num_bases=num_bases, aggr=conv_aggr)
            norm = act = None
            # For all layers except possibly the last, add normalization and activation.
            if i != num_layers - 1 or post_layers > 0:
                norm = norm_fn(hidden_size)
                act = activation_fn()
            # Wrap the convolution in a residual layer.
            convs.append(DeepGCNResidualLayer(conv=conv, norm=norm, act=act, dropout=dropout, block=res_method))
            in_channels = hidden_size
        self.nbr_concat = nbr_concat
        # If neighbor concatenation is enabled, append the concatenation layer.
        if nbr_concat:
            convs.append(RGCNConcat(num_relations, aggr=conv_aggr, rel_w=nbr_concat_weight))
        self.convs = torch.nn.ModuleList(convs)
        
        # Create a post-processing MLP layer.
        self.post_layer = MLP(dim_in=hidden_sizes[-1], dim_out=hidden_sizes[-1], activation_fn=activation_fn,
                              num_layers=post_layers)

    def reset_parameters(self):
        """
        Resets the parameters of all convolution layers.
        """
        for conv in self.convs:
            conv.reset_parameters()

    def to_devices(self, devices):
        """
        Moves the model's components to the specified devices.

        Args:
            devices (list): List of device objects (e.g., CPU, GPU) to which parts of the model will be assigned.

        Returns:
            The model with its components distributed across the provided devices.
        """
        model = self
        num_convs = len(self.convs)
        # Move pre-layer to the first device.
        if hasattr(model, 'pre_layer'):
            model.pre_layer = model.pre_layer.to(devices[0])
        # Move each convolution layer to its corresponding device.
        for i in range(num_convs):
            model.convs[i] = model.convs[i].to(devices[i])
            if hasattr(model, 'extras'):
                model.extras[i] = model.extras[i].to(devices[i])
        # Move the post-layer to the device after the convolution layers.
        if hasattr(model, 'post_layer'):
            model.post_layer = model.post_layer.to(devices[num_convs])
        return model

    def forward(self, x, adjs, edge_type_all, devices):
        """
        Forward pass of the RGCNEncoder.

        Args:
            x (Tensor): Input node features.
            adjs (list): List of tuples containing edge_index, edge id, and size for each layer.
            edge_type_all (Tensor): Edge type attributes for all edges.
            devices (list): List of devices on which to perform computations.

        Returns:
            Tensor: The output node embeddings.
        """
        # Retrieve neighbor concatenation flag and model version.
        nbr_concat = getattr(self, 'nbr_concat', False)
        model_version = getattr(self, 'model_version', 1)
        # Preprocess input features if pre_layer is defined.
        if hasattr(self, 'pre_layer'):
            x = self.pre_layer(x.to(devices[0]))
        # Iterate over adjacency lists for each layer.
        for i, (edge_index, e_id, size) in enumerate(adjs):
            # Determine edge dropout probability.
            if nbr_concat and i == self.num_layers:
                p = 0
            elif getattr(self, 'no_last_layer_dropout', False):
                p = 0
            else:
                p = self.edge_dropout
            # Apply edge dropout.
            edge_index, edge_type = dropout_adj(edge_index, edge_attr=edge_type_all[e_id], p=p, training=self.training)
            # Move node features to the current device.
            x = x.to(devices[i])
            # Identify target nodes (always placed first).
            x_target = x[:size[1]]
            # Apply the convolutional layer.
            x = self.convs[i]((x, x_target), edge_index.to(devices[i]), edge_type.to(devices[i]))
            # If no extra layers are defined and using an earlier model version, apply activation and dropout.
            if not hasattr(self, 'extras'):
                if model_version < 3 and i != self.num_layers - 1:
                    x = F.relu(x)
                    x = F.dropout(x, p=self.dropout, training=self.training)
            else:
                # If extras are defined, apply them.
                x = self.extras[i](x)
        # Apply post-processing MLP if defined.
        if hasattr(self, 'post_layer'):
            x = self.post_layer(x.to(devices[len(self.convs)]))
        return x

    @staticmethod
    def reg_loss(embedding):
        """
        Computes a regularization loss on the embedding.

        Args:
            embedding (Tensor): Node embedding tensor.

        Returns:
            Tensor: Regularization loss computed as the mean squared value of the embeddings.
        """
        return torch.mean(embedding.pow(2))


class RGCNEncoderFull(RGCNEncoder):
    """
    Full version of the RGCNEncoder that applies all convolution layers in a full forward pass.

    Overrides the forward method to work with a single edge_index and edge_type input.
    """
    def forward(self, x, edge_index, edge_type, devices):
        """
        Forward pass for the full encoder.

        Args:
            x (Tensor): Input node features.
            edge_index (Tensor): Edge indices for the graph.
            edge_type (Tensor): Edge type attributes.
            devices (list): List of devices to use for computation.

        Returns:
            Tensor: Output node embeddings.
        """
        assert len(devices) >= self.num_layers
        for i in range(self.num_layers):
            x = x.to(devices[i])
            x = self.convs[i](x, edge_index.to(devices[i]), edge_type.to(devices[i]))
            # Apply activation and dropout for all but the last layer.
            if i != self.num_layers - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x


class RGCNEncoderWithBERT(RGCNEncoder):
    """
    RGCNEncoderWithBERT extends RGCNEncoder to integrate BERT embeddings into the graph encoder.

    It combines graph node embeddings with BERT representations using an intermediate layer.
    """
    def __init__(self, *, bert_dim, **kwargs):
        """
        Initializes the RGCNEncoderWithBERT.

        Args:
            bert_dim (int): Dimensionality of BERT embeddings.
            **kwargs: Additional keyword arguments for the base RGCNEncoder.
        """
        super().__init__(**kwargs)

        # Create an intermediate embedding layer to combine GNN and BERT features.
        ie_dim = self.hidden_sizes[0] + bert_dim
        self.ie_layer = nn.Sequential(
            nn.Linear(ie_dim, ie_dim),
            self.norm_fn(ie_dim),
            self.activation_fn(),
            nn.Dropout(p=self.dropout),
            nn.Linear(ie_dim, ie_dim),
            nn.Tanh()
        )

    def to_devices(self, devices):
        """
        Moves the model's components, including the intermediate embedding layer, to specified devices.

        Args:
            devices (list): List of devices to move the components to.

        Returns:
            The model with updated device assignments.
        """
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
        # Move the intermediate embedding layer to the first device.
        model.ie_layer.to(devices[0])
        return model

    def forward(self, x, bert_x, adjs, edge_type_all, devices):
        """
        Forward pass for the encoder that combines graph and BERT embeddings.

        Args:
            x (Tensor): Input node features.
            bert_x (Tensor): BERT embeddings.
            adjs (list): List of adjacency information (edge_index, edge id, size) for each layer.
            edge_type_all (Tensor): Edge type attributes for all edges.
            devices (list): List of devices to use for computation.

        Returns:
            Tuple[Tensor, Tensor]: Updated node embeddings and BERT embeddings.
        """
        #
        """
          x: [?, node_emb_dim]
          bert_x:      [bs, bert_dim]
        """
        _bs = bert_x.size(0)  # Batch size based on BERT embeddings.

        # Retrieve flags and version info.
        nbr_concat = getattr(self, 'nbr_concat', False)
        model_version = getattr(self, 'model_version', 1)
        # Preprocess input features.
        if hasattr(self, 'pre_layer'):
            x = self.pre_layer(x.to(devices[0]))

        # Ensure the input dimension matches the expected hidden size.
        assert x.size(1) == self.hidden_sizes[0]
        x, bert_x = x.to(devices[0]), bert_x.to(devices[0])
        # Combine GNN and BERT embeddings for the first _bs nodes.
        combo = torch.cat([x[:_bs], bert_x], dim=1)  # [bs, gnn_dim + bert_dim]
        combo = combo + self.ie_layer(combo)         # Apply the intermediate embedding layer.
        # Split the combined embeddings back into separate tensors.
        x[:_bs], bert_x = combo[:, :x.size(1)], combo[:, x.size(1):]

        # Process each layer defined in the adjacency list.
        for i, (edge_index, e_id, size) in enumerate(adjs):
            # Determine edge dropout probability.
            if nbr_concat and i == self.num_layers:
                p = 0
            elif getattr(self, 'no_last_layer_dropout', False):
                p = 0
            else:
                p = self.edge_dropout
            # Apply edge dropout.
            edge_index, edge_type = dropout_adj(edge_index, edge_attr=edge_type_all[e_id], p=p, training=self.training)
            x = x.to(devices[i])
            # Identify target nodes.
            x_target = x[:size[1]]  # Target nodes are always placed first.
            # Apply the convolutional layer.
            x = self.convs[i]((x, x_target), edge_index.to(devices[i]), edge_type.to(devices[i]))
            if not hasattr(self, 'extras'):
                # Apply activation and dropout for earlier layers in older model versions.
                if model_version < 3 and i != self.num_layers - 1:
                    x = F.relu(x)
                    x = F.dropout(x, p=self.dropout, training=self.training)
            else:
                x = self.extras[i](x)

            # If not using neighbor concatenation at this layer, recombine with BERT embeddings.
            if nbr_concat and i == self.num_layers:
                pass
            else:
                # Ensure the hidden size remains consistent.
                assert x.size(1) == self.hidden_sizes[0]
                x, bert_x = x.to(devices[0]), bert_x.to(devices[0])
                combo = torch.cat([x[:_bs], bert_x], dim=1)  # [bs, gnn_dim + bert_dim]
                combo = combo + self.ie_layer(combo)         # Update the combined representation.
                x[:_bs], bert_x = combo[:, :x.size(1)], combo[:, x.size(1):]

        # Apply post-processing if defined.
        if hasattr(self, 'post_layer'):
            x = self.post_layer(x.to(devices[len(self.convs)]))

        # Ensure the final tensors are on the same device.
        x, bert_x = x.to(devices[0]), bert_x.to(devices[0])
        # Confirm that the output corresponds to the batch size.
        assert x.size(0) == _bs
        return x, bert_x
