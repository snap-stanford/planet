import torch  # PyTorch library for tensor operations
import torch.nn as nn  # Neural network module
import torch.nn.functional as F  # Functional interface for neural network operations


## General classes
class GeneralLayer(nn.Module):
    '''General wrapper for layers'''

    def __init__(self, name, dim_in, dim_out, has_act=True, has_bn=True,
                 has_l2norm=False, **kwargs):
        """
        Initialize the GeneralLayer.

        Parameters:
            name (str): Type of layer to create. Must be 'linear'.
            dim_in (int): Input feature dimension.
            dim_out (int): Output feature dimension.
            has_act (bool): Whether to apply an activation function after the linear layer.
            has_bn (bool): Whether to apply batch normalization.
            has_l2norm (bool): Whether to apply L2 normalization on the output.
            **kwargs: Additional keyword arguments. Expected to contain 'activation_fn'.
        """
        super(GeneralLayer, self).__init__()
        self.has_l2norm = has_l2norm  # Store flag for L2 normalization
        
        # Ensure that the layer type is 'linear'
        assert name == 'linear'
        
        # Create the linear layer. If batch normalization is used, disable bias.
        self.layer = Linear(dim_in, dim_out, bias=not has_bn, **kwargs)
        
        # Build a sequential container for optional batch normalization and activation
        layer_wrapper = []
        if has_bn:
            # Append BatchNorm1d layer if batch normalization is enabled
            layer_wrapper.append(nn.BatchNorm1d(dim_out))
        if has_act:
            # Append activation function, instantiated from the passed activation_fn
            layer_wrapper.append(kwargs['activation_fn']())
        self.post_layer = nn.Sequential(*layer_wrapper)  # Combine the layers

    def forward(self, batch):
        """
        Forward pass through the GeneralLayer.

        Applies a linear transformation, then optionally applies batch normalization,
        an activation function, and L2 normalization if configured.

        Parameters:
            batch (Tensor or custom object with attribute 'node_feature'): The input data.

        Returns:
            The transformed data with the same structure as the input.
        """
        # Apply the linear transformation
        batch = self.layer(batch)
        
        # Check if the batch is a plain tensor
        if isinstance(batch, torch.Tensor):
            # Apply post-processing layers (e.g., BN, activation)
            batch = self.post_layer(batch)
            if self.has_l2norm:
                # Apply L2 normalization if enabled
                batch = F.normalize(batch, p=2, dim=1)
        else:
            # For non-tensor objects, assume data is stored in 'node_feature'
            batch.node_feature = self.post_layer(batch.node_feature)
            if self.has_l2norm:
                # Apply L2 normalization on the node_feature attribute
                batch.node_feature = F.normalize(batch.node_feature, p=2, dim=1)
        return batch


class GeneralMultiLayer(nn.Module):
    '''General wrapper for a stack of layers'''

    def __init__(self, name, num_layers, dim_in, dim_out, dim_inner=None,
                 final_act=True, **kwargs):
        """
        Initialize the GeneralMultiLayer.

        Parameters:
            name (str): Type of layer to create (must be 'linear').
            num_layers (int): Number of layers to stack.
            dim_in (int): Input feature dimension.
            dim_out (int): Output feature dimension.
            dim_inner (int, optional): Inner (hidden) layer dimension. Defaults to dim_in if not provided.
            final_act (bool): Whether to apply the activation function on the final layer.
            **kwargs: Additional keyword arguments passed to each GeneralLayer.
        """
        super(GeneralMultiLayer, self).__init__()
        # Set hidden dimension; if not provided, default to input dimension
        dim_inner = dim_in if dim_inner is None else dim_inner
        # Create and add the specified number of layers to the module
        for i in range(num_layers):
            # Determine input dimension for current layer
            d_in = dim_in if i == 0 else dim_inner
            # Determine output dimension: last layer uses dim_out, others use dim_inner
            d_out = dim_out if i == num_layers - 1 else dim_inner
            # Use final_act flag only for the last layer, otherwise always apply activation
            has_act = final_act if i == num_layers - 1 else True
            # Instantiate a GeneralLayer with the computed dimensions and settings
            layer = GeneralLayer(name, d_in, d_out, has_act, **kwargs)
            # Dynamically add the layer as a submodule with a unique name
            self.add_module('Layer_{}'.format(i), layer)

    def forward(self, batch):
        """
        Forward pass through the GeneralMultiLayer.

        Sequentially passes the input through each layer in the stack.

        Parameters:
            batch (Tensor or custom object with attribute 'node_feature'): The input data.

        Returns:
            The transformed data after passing through all layers.
        """
        # Iterate through all child modules (layers) and update the batch
        for layer in self.children():
            batch = layer(batch)
        return batch


## Core basic layers
# Input: batch; Output: batch
class Linear(nn.Module):
    def __init__(self, dim_in, dim_out, bias=False, **kwargs):
        """
        Initialize the Linear layer.

        Parameters:
            dim_in (int): Input feature dimension.
            dim_out (int): Output feature dimension.
            bias (bool): Whether to include a bias term.
            **kwargs: Additional keyword arguments (currently unused).
        """
        super(Linear, self).__init__()
        # Create a standard linear transformation layer from PyTorch
        self.model = nn.Linear(dim_in, dim_out, bias=bias)

    def forward(self, batch):
        """
        Forward pass through the Linear layer.

        Applies the linear transformation to the input tensor or to its 'node_feature' attribute.

        Parameters:
            batch (Tensor or custom object with attribute 'node_feature'): The input data.

        Returns:
            The transformed data.
        """
        if isinstance(batch, torch.Tensor):
            # If input is a tensor, apply the linear layer directly
            batch = self.model(batch)
        else:
            # Otherwise, apply the linear layer to the 'node_feature' attribute
            batch.node_feature = self.model(batch.node_feature)
        return batch


class BatchNorm1dNode(nn.Module):
    '''General wrapper for batch normalization on node features'''

    def __init__(self, dim_in):
        """
        Initialize the BatchNorm1dNode.

        Parameters:
            dim_in (int): Dimension of the node features for normalization.
        """
        super(BatchNorm1dNode, self).__init__()
        # Create a BatchNorm1d layer for the specified input dimension
        self.bn = nn.BatchNorm1d(dim_in)

    def forward(self, batch):
        """
        Forward pass through the BatchNorm1dNode.

        Applies batch normalization to the 'node_feature' attribute of the input.

        Parameters:
            batch: An object containing a 'node_feature' attribute.

        Returns:
            The input object with normalized 'node_feature'.
        """
        # Apply batch normalization to the node features
        batch.node_feature = self.bn(batch.node_feature)
        return batch


class MLP(nn.Module):
    def __init__(self, dim_in, dim_out, activation_fn, bias=True, dim_inner=None,
                 num_layers=2, **kwargs):
        '''
        Initialize the MLP (Multi-Layer Perceptron).

        Parameters:
            dim_in (int): Input feature dimension.
            dim_out (int): Output feature dimension.
            activation_fn (callable): Activation function class to be used.
            bias (bool): Whether to include bias in the linear layers.
            dim_inner (int, optional): Hidden layer dimension. Defaults to dim_in if not provided.
            num_layers (int): Number of layers. For 0 layers, an identity mapping is used.
            **kwargs: Additional keyword arguments.

        Note:
            MLP works for 0 layers (resulting in an identity mapping).
        '''
        super(MLP, self).__init__()
        # Set hidden dimension to input dimension if not specified
        dim_inner = dim_in if dim_inner is None else dim_inner
        layers = []  # List to hold all layers of the MLP
        
        if num_layers > 1:
            # For more than one layer, use GeneralMultiLayer for the hidden layers
            layers.append(
                GeneralMultiLayer('linear', num_layers - 1, dim_in, dim_inner,
                                  dim_inner, final_act=True, activation_fn=activation_fn))
            # Append a final linear layer to produce the output dimension
            layers.append(Linear(dim_inner, dim_out, bias))
        elif num_layers == 1:
            # For a single layer, directly use a Linear layer
            layers.append(Linear(dim_in, dim_out, bias))
        elif num_layers == 0:
            # For 0 layers, use the identity mapping (no change to input)
            layers.append(nn.Identity())
        
        # Combine all layers into a sequential model
        self.model = nn.Sequential(*layers)

    def forward(self, batch):
        """
        Forward pass through the MLP.

        Applies the MLP model to either the input tensor directly or to the
        'node_feature' attribute of the input object.

        Parameters:
            batch (Tensor or custom object with attribute 'node_feature'): The input data.

        Returns:
            The output after processing through the MLP.
        """
        if isinstance(batch, torch.Tensor):
            # Directly process the tensor through the model
            batch = self.model(batch)
        else:
            # Process the 'node_feature' attribute through the model
            batch.node_feature = self.model(batch.node_feature)
        return batch
