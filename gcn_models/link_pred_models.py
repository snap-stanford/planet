from typing import List

import torch
import torch.nn.functional as F
from torch import nn

from gcn_models.decoders import TransEDecoder, DistMultDecoder
from gcn_models.encoders import RGCNEncoder, RGCNEncoderWithBERT
from gcn_models.node_classification_models.classification_model import MixedEmbedding, ClassificationTaskLoss, \
    RegressionTaskLoss, ClassificationMeanTaskLoss

import pickle as pkl


class LinkPredModel(torch.nn.Module):
    """
    Link prediction model that combines a graph convolutional encoder and a decoder.
    
    This model supports different encoder types (with or without BERT) and decoders (TransE or DistMult).
    It also implements regularization and loss computation for training.
    
    Parameters:
        conv_fn: Convolution function to use in the encoder.
        num_relations (int): Number of relation types.
        in_channels: Number of input channels (features per node).
        hidden_dim (List[int]): List of hidden dimensions for embedding and encoder layers.
        num_enc_layers (int): Number of layers in the encoder.
        emb_df: Dataframe for initializing embeddings.
        num_bases (int): Number of bases for relation-specific parameterization.
        conv_aggr (str): Aggregation method for convolution.
        decoder_type (str): Type of decoder to use ('TransE' or 'DistMult').
        gamma (float): Margin or scaling factor used in the decoder.
        activation_fn: Activation function to use.
        norm_fn: Normalization function to use.
        negative_adversarial_sampling (bool): Whether to use negative adversarial sampling.
        adversarial_temperature (float): Temperature parameter for adversarial sampling.
        reg_param (float): Regularization parameter.
        dropout (float): Dropout rate for encoder layers.
        edge_dropout (float): Dropout rate for edges in the graph.
        add_residual (bool): Whether to add residual connections.
        num_pre_layers (int): Number of pre-processing layers.
        num_post_layers (int): Number of post-processing layers.
        nbr_concat (bool): Whether to concatenate neighbor features.
        nbr_concat_weight (bool): Whether to use weighting when concatenating neighbor features.
        bert_dim (int, optional): Dimensionality of BERT embeddings; if 0, BERT is not used.
    """
    def __init__(self, *,
                 conv_fn,
                 num_relations: int,
                 in_channels,
                 hidden_dim: List[int],
                 num_enc_layers: int,
                 emb_df,
                 num_bases: int,
                 conv_aggr: str,
                 decoder_type: str,
                 gamma: float,
                 activation_fn,
                 norm_fn,
                 negative_adversarial_sampling: bool,
                 adversarial_temperature: float,
                 reg_param: float,
                 dropout: float,
                 edge_dropout: float,
                 add_residual: bool,
                 num_pre_layers: int,
                 num_post_layers: int,
                 nbr_concat: bool,
                 nbr_concat_weight: bool,
                 bert_dim=0,  # 0 means not combining BERT features
                 ):
        super().__init__()
        # Initialize mixed embedding layer
        self.embedding = MixedEmbedding(embedding_dim=hidden_dim[0], num_embeddings=in_channels, emb_df=emb_df)
        self.bert_dim = bert_dim

        # Choose encoder type based on whether BERT embeddings are used
        if bert_dim == 0:
            self.encoder = RGCNEncoder(
                conv_fn=conv_fn,
                conv_aggr=conv_aggr,
                in_channels=hidden_dim[0],
                hidden_size=hidden_dim[1:],
                num_relations=num_relations,
                num_bases=num_bases,
                num_layers=num_enc_layers,
                activation_fn=activation_fn,
                norm_fn=norm_fn,
                dropout=dropout,
                edge_dropout=edge_dropout,
                res_method='res' if add_residual else 'plain',
                pre_layers=num_pre_layers,
                post_layers=num_post_layers,
                nbr_concat=nbr_concat,
                nbr_concat_weight=nbr_concat_weight
            )
        else:
            self.encoder = RGCNEncoderWithBERT(
                conv_fn=conv_fn,
                conv_aggr=conv_aggr,
                in_channels=hidden_dim[0],
                hidden_size=hidden_dim[1:],
                num_relations=num_relations,
                num_bases=num_bases,
                num_layers=num_enc_layers,
                activation_fn=activation_fn,
                norm_fn=norm_fn,
                dropout=dropout,
                edge_dropout=edge_dropout,
                res_method='res' if add_residual else 'plain',
                pre_layers=num_pre_layers,
                post_layers=num_post_layers,
                nbr_concat=nbr_concat,
                nbr_concat_weight=nbr_concat_weight,
                bert_dim=bert_dim
            )
        # Initialize the appropriate decoder based on decoder_type
        if decoder_type == 'TransE':
            self.decoder = TransEDecoder(gamma=gamma, num_rels=num_relations // 2, h_dim=hidden_dim[-1])
        elif decoder_type == 'DistMult':
            self.decoder = DistMultDecoder(num_rels=num_relations // 2, h_dim=hidden_dim[-1])
        # Store regularization and sampling parameters
        self.reg_param = reg_param
        self.negative_adversarial_sampling = negative_adversarial_sampling
        self.adversarial_temperature = adversarial_temperature

    def to_devices(self, devices):
        """
        Moves model components to specified devices.
        
        Parameters:
            devices (List): List of device identifiers.
        
        Returns:
            model: The model with components moved to the designated devices.
        """
        model = self
        model.embedding = model.embedding.to(devices[0])
        model.encoder = model.encoder.to_devices(devices[1:])
        if hasattr(model, 'decoder'):
            model.decoder = model.decoder.to(devices[3])
        return model

    def encode(self, x, adjs, edge_type, devices: List[int]):
        """
        Encodes the input features using the embedding and encoder layers.
        
        Parameters:
            x: Input node features. If BERT is used, x should be a tuple (gcn_x, orig_node_x, bert_x).
            adjs: Adjacency information for the graph.
            edge_type: Types of edges in the graph.
            devices (List[int]): List of devices for different parts of the computation.
        
        Returns:
            Encoded node representations (and optionally BERT features).
        """
        if getattr(self, 'bert_dim', 0) == 0:
            # When not using BERT, embed and encode
            x = self.embedding(x.to(devices[0]))
            # Return encoded representation from the GCN encoder
            return self.encoder(x, adjs, edge_type, devices[1:])
        else:
            # When using BERT, expect a tuple of inputs
            gcn_x, orig_node_x, bert_x = x
            _bs = bert_x.size(0)
            # Ensure consistency in batch sizes
            assert _bs == orig_node_x.size(0)
            assert (gcn_x[:_bs] == orig_node_x).long().sum() == _bs
            gcn_x = self.embedding(gcn_x.to(devices[0]))
            # Encode both GCN and BERT features
            gcn_x, bert_x = self.encoder(gcn_x, bert_x, adjs, edge_type, devices[1:])
            return gcn_x, bert_x

    def forward(self, embs, triplets, devices: List[int], mode: str = 'train'):
        """
        Forward pass for link prediction.
        
        Parameters:
            embs: Node embeddings.
            triplets: A tuple containing positive samples, head negative samples, and tail negative samples.
            devices (List[int]): List of devices for computation.
            mode (str): Mode of operation ('train' or 'eval').
        
        Returns:
            If mode is 'eval': returns embeddings and positive scores.
            Otherwise: returns positive and negative scores.
        """
        embs = embs.to(devices[3])
        pos_samples, head_negative_sample, tail_negative_sample = triplets
        # pos_samples: list[h, r, t] with each of h, r, t being of shape [batch_size]
        # head_negative_sample: shape [batch_size, num_negatives]
        # tail_negative_sample: shape [batch_size, num_negatives]
        positive_score = self.decoder(embs, pos_samples)  # [batch_size, 1]

        if mode == 'eval':
            return embs, positive_score

        # Compute scores for negative samples using head and tail batching
        head_neg_scores = self.decoder(embs, (pos_samples, head_negative_sample), mode='head-batch')
        tail_neg_scores = self.decoder(embs, (pos_samples, tail_negative_sample), mode='tail-batch')

        negative_score = torch.cat([head_neg_scores, tail_neg_scores], dim=-1)  # [batch_size, total_negatives]

        return positive_score, negative_score

    def regularization_loss(self, embedding, devices):
        """
        Computes the regularization loss based on the decoder's regularization method.
        
        Parameters:
            embedding: Node embeddings.
            devices: List of devices to which the regularization loss should be moved.
        
        Returns:
            Regularization loss tensor.
        """
        # Note: Alternative implementations (commented out) show different regularization strategies.
        return self.decoder.reg_loss().to(devices[3])

    def loss(self, embs, scores, devices):
        """
        Computes the overall loss including positive and negative sampling losses and regularization.
        
        Parameters:
            embs: Node embeddings.
            scores: Tuple containing positive and negative scores.
            devices: List of devices for computation.
        
        Returns:
            A tuple of (total loss, positive sample loss, negative sample loss).
        """
        positive_score, negative_score = scores
        if self.negative_adversarial_sampling:
            # Apply self-adversarial sampling; detach the sampling weights to avoid backpropagation
            negative_score = (F.softmax(negative_score * self.adversarial_temperature, dim=1).detach()
                              * F.logsigmoid(-negative_score)).sum(dim=1)
        else:
            negative_score = F.logsigmoid(-negative_score).mean(dim=1)  # [batch_size]

        positive_score = F.logsigmoid(positive_score).squeeze(dim=1)  # [batch_size]

        positive_sample_loss = - positive_score.mean()  # Scalar loss for positive samples
        negative_sample_loss = - negative_score.mean()  # Scalar loss for negative samples

        loss = (positive_sample_loss + negative_sample_loss) / 2 + self.reg_param * self.regularization_loss(embs, devices)

        return loss, positive_sample_loss, negative_sample_loss


class EmbeddingNormalize(nn.Module):
    """
    Module to perform L2 normalization on embeddings.
    
    Depending on the flag 'with_grad', normalization is performed with or without gradient flow.
    """
    def __init__(self, with_grad: bool):
        """
        Initializes the normalization module.
        
        Parameters:
            with_grad (bool): If True, gradients will flow through the normalization; otherwise, normalization is detached.
        """
        super(EmbeddingNormalize, self).__init__()
        self.with_grad = with_grad
        self.epsilon = 1e-12  # Small constant to avoid division by zero

    def forward(self, x):
        """
        Forward pass to normalize the embeddings.
        
        Parameters:
            x: Input embeddings tensor.
        
        Returns:
            L2 normalized embeddings.
        """
        if self.with_grad:
            return F.normalize(x, p=2, dim=1)
        else:
            norm = x.norm(p=2, dim=1, keepdim=True).detach()
            return x / norm

    def __repr__(self):
        return f'{self.__class__.__name__}(with_grad={self.with_grad})'


class RelationAugment(nn.Module):
    """
    Module for augmenting node embeddings with relation embeddings.
    
    This module supports the DistMult decoder and aggregates relation-augmented features.
    """
    def __init__(self, r_embs, decoder_type, aggr='mean'):
        """
        Initializes the RelationAugment module.
        
        Parameters:
            r_embs: Relation embeddings.
            decoder_type (str): Decoder type; currently only supports 'DistMult'.
            aggr (str): Aggregation method for combining features ('mean' or 'sum').
        """
        super(RelationAugment, self).__init__()
        # Register relation embeddings as a buffer so that they are part of the module's state
        self.register_buffer('r_embs', nn.Parameter(r_embs))
        self.decoder_type = decoder_type
        assert decoder_type in ['DistMult']
        self.aggr = aggr
        assert aggr in ['mean', 'sum']

    def forward(self, x):
        """
        Augments the input tensor x with relation-specific information.
        
        For each relation embedding, multiplies it element-wise with x and aggregates the results.
        
        Parameters:
            x: Input tensor representing node features.
        
        Returns:
            Augmented tensor after aggregation.
        """
        if self.decoder_type == 'DistMult':
            xs = [x]
            for r_emb in self.r_embs:
                xs.append(x * r_emb)
            if self.aggr == 'mean':
                return torch.stack(xs, dim=1).mean(dim=1)
            elif self.aggr == 'sum':
                return torch.stack(xs, dim=1).sum(dim=1)


class ClassificationModel(nn.Module):
    """
    Classification model that processes node representations through an MLP and applies task-specific heads.
    
    Supports various classification and regression tasks.
    """
    def __init__(self, *, input_dim: int, hidden_sizes: List[int], num_layers: int,
                 normalize_embeddings: bool, normalize_embeddings_with_grad: bool,
                 normalize_clf_weights: bool,
                 norm_layer, activation_fn,
                 dropout_prob: float, tasks, args):
        """
        Initializes the ClassificationModel.
        
        Parameters:
            input_dim (int): Dimensionality of input features.
            hidden_sizes (List[int]): List of hidden layer sizes.
            num_layers (int): Total number of layers (including the output layer).
            normalize_embeddings (bool): Flag to apply embedding normalization.
            normalize_embeddings_with_grad (bool): Whether normalization should allow gradients.
            normalize_clf_weights (bool): Flag to normalize classifier weights.
            norm_layer: Normalization layer to use.
            activation_fn: Activation function.
            dropout_prob (float): Dropout probability.
            tasks: List of task configurations.
            args: Additional arguments.
        """
        super().__init__()
        self.args = args
        layers = []
        self.normalize_clf_weights = normalize_clf_weights
        # Optionally prepend embedding normalization
        if normalize_embeddings:
            layers.append(EmbeddingNormalize(normalize_embeddings_with_grad))
        # Ensure the number of layers matches the provided hidden sizes
        assert num_layers == len(hidden_sizes) + 1
        for i in range(num_layers - 1):
            p = dropout_prob
            hidden_dim = hidden_sizes[i]
            # Append a linear layer, normalization, activation, and dropout
            layers.extend([
                nn.Linear(input_dim, hidden_dim),
                norm_layer(hidden_dim),
                activation_fn(),
                nn.Dropout(p=p)
            ])
            input_dim = hidden_dim
        # Define the representation network
        self.representation = nn.Sequential(*layers)
        # Build task-specific heads and losses; assuming one task in tasks
        assert len(tasks) == 1
        task_heads = []
        task_loss = []
        task_weights = []
        for task_params in tasks:
            task_name = task_params['name']
            num_subtasks = task_params['num_subtasks']
            print('task_name', task_name, 'num_subtasks', num_subtasks)
            # For certain tasks, the input dimension might be doubled
            task_heads.append(nn.Linear(input_dim * 2 if task_name == 'binary_pair_efficacy' else input_dim,
                                          num_subtasks))
            task_weights.append(task_params['task_weight'])
            # Assign the appropriate loss function based on the task name
            if task_name in ['ae_clf', 'binary', 'ae_clf_or', 'ae_clf_freq', 'binary_or', 'binary_pair_efficacy', 'with_or_without_results']:
                task_loss.append(ClassificationTaskLoss(num_tasks=num_subtasks,
                                                        weight=task_params['subtask_weight'],
                                                        pos_weight=task_params['pos_weight']))
            elif task_name in ['ae_clf_or_l']:
                task_loss.append(
                    ClassificationMeanTaskLoss(num_tasks=num_subtasks,
                                               weight=task_params['subtask_weight'],
                                               pos_weight=task_params['pos_weight']))
            elif task_name == 'ae_regr':
                task_loss.append(RegressionTaskLoss(num_tasks=num_subtasks,
                                                    weight=task_params['subtask_weight']))
            else:
                raise RuntimeError("Invalid task name " + task_name)
        self.task_heads = nn.ModuleList(task_heads)
        self.task_losses = nn.ModuleList(task_loss)
        self.task_weights = task_weights
        self.version = 2

    def to_device(self, device):
        """
        Moves the entire model to the specified device.
        
        Parameters:
            device: Target device.
        
        Returns:
            The model on the target device.
        """
        return self.to(device)

    def forward(self, x, devices):
        """
        Forward pass for the classification model.
        
        Parameters:
            x: Input features.
            devices: List of devices; the last device in the list is used for final computation.
        
        Returns:
            A list of outputs from the task-specific heads.
        """
        version = self.version if hasattr(self, 'version') else 1
        if self.normalize_clf_weights:
            # Normalize classifier weights if required
            self.classifier[-1].weight.data = F.normalize(self.classifier[-1].weight.data, p=2, dim=1)
        if version == 1:
            return self.classifier(x.to(devices[-1]))
        elif version == 2:
            # Obtain representation from the MLP
            x = self.representation(x.to(devices[-1]))  # [batch_size, hidden_dim]
            if self.args.default_task_name == 'binary_pair_efficacy':
                # For binary pair efficacy, combine pairs of representations
                _bs, _dim = x.size()
                _bs = _bs // 2
                x = torch.cat(x.view(2, _bs, _dim).unbind(dim=0), dim=1)  # [batch_size, 2 * hidden_dim]
            # Compute outputs for each task head
            return [module(x) for module in self.task_heads]

    def loss(self, scores, labels):
        """
        Computes the loss for the classification tasks.
        
        Parameters:
            scores: List of outputs from the task heads.
            labels: Ground truth labels.
        
        Returns:
            The overall loss value.
        """
        version = self.version if hasattr(self, 'version') else 1
        if version == 1:
            return self.task_loss(scores, labels.to(scores.device))
        elif version == 2:
            loss = 0
            # Compute weighted loss for each task head
            for i, weight in enumerate(self.task_weights):
                loss += weight * self.task_losses[i](scores[i],
                                                     labels[i].to(scores[i].device),
                                                     labels[i + len(self.task_weights)].to(scores[i].device))
            return loss


class ClassificationModelWithAEEmb(nn.Module):
    """
    Classification model that integrates additional autoencoder embeddings (AE embeddings)
    with the standard MLP-based representation for classification tasks.
    """
    def __init__(self, *, input_dim: int, hidden_sizes: List[int], num_layers: int,
                 normalize_embeddings: bool, normalize_embeddings_with_grad: bool,
                 normalize_clf_weights: bool,
                 norm_layer, activation_fn,
                 dropout_prob: float, tasks, args):
        """
        Initializes the ClassificationModelWithAEEmb.
        
        Parameters:
            input_dim (int): Dimensionality of input features.
            hidden_sizes (List[int]): List of hidden layer sizes.
            num_layers (int): Total number of layers.
            normalize_embeddings (bool): Whether to normalize embeddings.
            normalize_embeddings_with_grad (bool): Whether normalization should allow gradients.
            normalize_clf_weights (bool): Whether to normalize classifier weights.
            norm_layer: Normalization layer to use.
            activation_fn: Activation function.
            dropout_prob (float): Dropout probability.
            tasks: List of task configurations.
            args: Additional arguments.
        """
        super().__init__()
        layers = []
        self.normalize_clf_weights = normalize_clf_weights
        # Optionally add normalization layer for embeddings
        if normalize_embeddings:
            layers.append(EmbeddingNormalize(normalize_embeddings_with_grad))
        assert num_layers == len(hidden_sizes) + 1
        for i in range(num_layers - 1):
            p = dropout_prob
            hidden_dim = hidden_sizes[i]
            # Append linear layer, normalization, activation, and dropout layers
            layers.extend([
                nn.Linear(input_dim, hidden_dim),
                norm_layer(hidden_dim),
                activation_fn(),
                nn.Dropout(p=p)
            ])
            input_dim = hidden_dim
        self.representation = nn.Sequential(*layers)
        # Load precomputed AE embeddings from a pickle file
        AEembdict = pkl.load(open('../data/clf_data/PT/AE_embdict_OR2.pkl', 'rb'))

        task_heads = []
        task_loss = []
        task_weights = []

        # Ensure exactly one task is provided
        assert len(tasks) == 1
        task_params = tasks[0]
        task_name = task_params['name']
        num_subtasks = task_params['num_subtasks']
        idx2aename = task_params['idx2aename']
        print('task_name', task_name, 'num_subtasks', num_subtasks)
        assert task_name in ['ae_clf_or']

        # Create a tensor for AE embeddings based on the provided index mapping
        AEemb = torch.stack([torch.tensor(AEembdict[aename]) for aename in idx2aename])
        self.AEemb = nn.Parameter(AEemb)
        assert num_subtasks == AEemb.size(0)
        # The head takes concatenated features from the representation and AE embeddings
        task_heads.append(nn.Linear(input_dim + AEemb.size(1), num_subtasks))
        task_weights.append(task_params['task_weight'])

        # Set up the appropriate loss function for the task
        if task_name in ['ae_clf', 'binary', 'ae_clf_or', 'ae_clf_freq', 'binary_or']:
            print('task_params[pos_weight]', task_params['pos_weight'])
            task_loss.append(ClassificationTaskLoss(num_tasks=num_subtasks,
                                                     weight=task_params['subtask_weight'],
                                                     pos_weight=task_params['pos_weight']))
        elif task_name in ['ae_clf_or_l']:
            task_loss.append(
                ClassificationMeanTaskLoss(num_tasks=num_subtasks,
                                           weight=task_params['subtask_weight'],
                                           pos_weight=task_params['pos_weight']))
        elif task_name == 'ae_regr':
            task_loss.append(RegressionTaskLoss(num_tasks=num_subtasks,
                                                weight=task_params['subtask_weight']))
        else:
            raise RuntimeError("Invalid task name " + task_name)

        self.task_heads = nn.ModuleList(task_heads)
        self.task_losses = nn.ModuleList(task_loss)
        self.task_weights = task_weights
        self.version = 2

    def to_device(self, device):
        """
        Moves the model and its AE embeddings to the specified device.
        
        Parameters:
            device: Target device.
        
        Returns:
            The model on the target device.
        """
        # Ensure AE embeddings are moved to the target device
        self.AEemb.to(device)
        return self.to(device)

    def forward(self, x, devices):
        """
        Forward pass for the classification model with AE embeddings.
        
        Parameters:
            x: Input features.
            devices: List of devices; the last device is used for the final computation.
        
        Returns:
            A list of outputs computed from the task-specific heads.
        """
        version = self.version if hasattr(self, 'version') else 1
        if self.normalize_clf_weights:
            # Normalize classifier weights if necessary
            self.classifier[-1].weight.data = F.normalize(self.classifier[-1].weight.data, p=2, dim=1)
        if version == 1:
            return self.classifier(x.to(devices[-1]))
        elif version == 2:
            # Obtain representation from the MLP
            x = self.representation(x.to(devices[-1]))  # [batch_size, hidden_dim]
            # Expand and repeat representation to match the number of AE embeddings
            x = x.unsqueeze(1).repeat(1, self.AEemb.size(0), 1)  # [batch_size, num_AE, hidden_dim]
            # Concatenate the AE embeddings with the representation
            x = torch.cat([x, self.AEemb.unsqueeze(0).repeat(x.size(0), 1, 1)], dim=2)  # [batch_size, num_AE, hidden_dim + AE_dim]
            # Compute output scores for each task head using a weighted sum plus bias
            return [(module.weight.unsqueeze(0) * x).sum(dim=2) + module.bias.unsqueeze(0) for module in self.task_heads]

    def loss(self, scores, labels):
        """
        Computes the loss for the classification tasks with AE embeddings.
        
        Parameters:
            scores: List of outputs from the task heads.
            labels: Ground truth labels.
        
        Returns:
            The overall loss value.
        """
        version = self.version if hasattr(self, 'version') else 1
        if version == 1:
            return self.task_loss(scores, labels.to(scores.device))
        elif version == 2:
            loss = 0
            # Compute weighted loss for each task head
            for i, weight in enumerate(self.task_weights):
                loss += weight * self.task_losses[i](scores[i],
                                                     labels[i].to(scores[i].device),
                                                     labels[i + len(self.task_weights)].to(scores[i].device))
            return loss
