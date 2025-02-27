from typing import List
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from gcn_models.decoders import TransEDecoder, DistMultDecoder
from gcn_models.encoders import RGCNEncoderFull


class MixedEmbedding(nn.Module):
    """
    A class to handle mixed embedding transformations based on different types of embeddings.
    """
    def __init__(self, *, embedding_dim, num_embeddings, emb_df=None):
        """
        Initializes the MixedEmbedding layer.

        :param embedding_dim: The dimension of the embedding.
        :param num_embeddings: The number of total embeddings.
        :param emb_df: DataFrame containing the embeddings and additional data (optional).
        """
        super(MixedEmbedding, self).__init__()
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        print ('self.num_embeddings', self.num_embeddings)

        # If no embedding DataFrame is provided, create a simple embedding layer
        if emb_df is None:
            self.transforms = nn.ModuleList(
                [nn.Embedding(embedding_dim=embedding_dim, num_embeddings=num_embeddings)]
            )
            return

        # Calculate the maximum length of embeddings
        max_len = emb_df['emb'].map(len).max()

        # Initialize embeddings for specific node ids
        node_ids = emb_df['id'].values
        need_embedding = np.ones(num_embeddings, dtype=np.bool)
        need_embedding[node_ids] = False
        embedding_node_ids = np.arange(num_embeddings)[need_embedding]
        emb_inverse_index = np.zeros(num_embeddings) - 1
        emb_inverse_index[embedding_node_ids] = np.arange(len(embedding_node_ids))

        self.register_buffer('emb_inverse_index', torch.tensor(emb_inverse_index, dtype=torch.long))
        
        # Define transforms for each embedding type
        self.transforms = nn.ModuleList(
            [nn.Embedding(embedding_dim=embedding_dim, num_embeddings=len(embedding_node_ids))] +
            [nn.Linear(max_len, self.embedding_dim, bias=False) for _ in emb_df.etype.cat.categories]
        )

        # Prepare fixed embeddings and etype for nodes
        emb = emb_df['emb'].map(lambda x: x.mean(axis=0) if len(x.shape) > 1 else x) \
            .map(lambda x: np.pad(np.nan_to_num(x), (0, max_len - len(x))))

        emb = torch.tensor(np.stack(emb.values), dtype=torch.float32)
        etype = torch.tensor(emb_df['etype'].cat.codes.values + 1, dtype=torch.uint8)

        fixed_emb = torch.zeros((self.num_embeddings, max_len), dtype=torch.float32)
        etype_all = torch.zeros(self.num_embeddings, dtype=torch.uint8)

        fixed_emb[node_ids] = emb
        etype_all[node_ids] = etype

        self.register_buffer('fixed_emb', fixed_emb)
        self.register_buffer('etype', etype_all)

    def forward(self, x):
        """
        Forward pass for the MixedEmbedding class.

        :param x: The input tensor representing node indices.
        :return: The transformed embeddings.
        """
        if len(self.transforms) == 1:
            return self.transforms[0](x)
        e_type = self.etype[x]
        out = torch.zeros((len(x), self.embedding_dim), device=x.device)
        for idx, transform in enumerate(self.transforms):
            mask = e_type == idx
            x_idx = self.fixed_emb[x[mask]] if idx != 0 else self.emb_inverse_index[x[mask]]
            out[mask] = transform(x_idx)
        return out


class SetEmbedding(nn.Module):
    """
    A module for embedding sets of data, using either mean or sum aggregation.
    """
    def __init__(self, channels, mode='mean'):
        """
        Initializes the SetEmbedding layer.

        :param channels: The number of input channels.
        :param mode: The aggregation method ('mean' or 'sum').
        """
        super(SetEmbedding, self).__init__()
        assert mode in ['mean', 'sum']
        self.linear = nn.Linear(channels, channels)
        self.mode = mode

    def forward(self, x, seq_len):
        """
        Forward pass for set embedding.

        :param seq_len: Tensor representing sequence lengths.
        :param x: Input tensor of shape B * N * D.
        :return: Aggregated and transformed tensor.
        """
        if self.mode == 'mean':
            x = x.sum(dim=1) / seq_len.unsqueeze(1)
        elif self.mode == 'sum':
            x = x.sum(dim=1)
        else:
            raise RuntimeError("Unexpected set embedding mode")
        return self.linear(x)


class TaskLoss(nn.Module):
    """
    A loss function for multi-task learning with task-specific weighting.
    """
    def __init__(self, *, num_tasks, weight='uniform'):
        """
        Initializes the TaskLoss function.

        :param num_tasks: The number of tasks (outputs).
        :param weight: The type of weighting ('uniform' or 'variance').
        """
        super(TaskLoss, self).__init__()
        assert weight in ['variance', 'uniform']
        self.weight = weight
        self.num_tasks = num_tasks
        self.log_vars = nn.Parameter(torch.zeros(num_tasks), requires_grad=weight == 'variance')

    def forward(self, input, target, sample_weight):
        """
        Computes the task-specific loss.

        :param input: The predicted values.
        :param target: The ground truth values.
        :param sample_weight: The weight for each sample.
        :return: The weighted loss across tasks.
        """
        loss = self.criterion(input, target) * sample_weight
        loss = loss.mean(dim=0)
        if self.weight == 'uniform':
            return loss.sum()

        task_weight = torch.exp(-self.log_vars)
        return (loss * task_weight).sum() + self.log_vars.sum()


class ClassificationTaskLoss(TaskLoss):
    """
    A classification-specific task loss that uses binary cross-entropy with logits.
    """
    def __init__(self, *, pos_weight, **kwargs):
        super(ClassificationTaskLoss, self).__init__(**kwargs)
        self.criterion = nn.BCEWithLogitsLoss(reduction="none", pos_weight=pos_weight)


class ClassificationMeanTaskLoss(TaskLoss):
    """
    A classification-specific task loss using mean binary cross-entropy with logits.
    """
    def __init__(self, *, pos_weight, **kwargs):
        super(ClassificationMeanTaskLoss, self).__init__(**kwargs)
        self.criterion = nn.BCEWithLogitsLoss()

    def forward(self, input, target, sample_weight):
        """
        Computes the loss for classification with mean reduction.

        :param input: The predicted values.
        :param target: The ground truth values.
        :param sample_weight: The weight for each sample.
        :return: The mean loss value.
        """
        loss = self.criterion(input, target)
        return loss


class RegressionTaskLoss(TaskLoss):
    """
    A regression-specific task loss using mean squared error.
    """
    def __init__(self, **kwargs):
        super(RegressionTaskLoss, self).__init__(**kwargs)
        self.criterion = nn.MSELoss(reduction="none")


class CombinationLoss(TaskLoss):
    """
    A combination of multiple losses, weighted by task-specific weights.
    """
    def forward(self, loss, _=None):
        """
        Combines the losses for all tasks with respective weights.

        :param loss: List of individual task losses.
        :return: The combined loss.
        """
        task_weight = torch.exp(-self.log_vars)
        return sum(task_weight[i] * loss[i] for i in range(len(loss))) + self.log_vars.sum()


class ClassificationModel(nn.Module):
    """
    A classification model that uses various embedding and encoding methods for multi-task learning.
    """
    def __init__(self, *, in_channels, num_relations, num_bases, num_layers_conv, hidden_size, columns, out_channels,
                 emb_df, pos_weight, task_weight,
                 dropout=0.5, aggregate='concat', set_embedding_mode='mean',
                 trial_feats_size: int = -1, only_trial_feats: bool = False):
        """
        Initializes the classification model.

        :param in_channels: The number of input channels.
        :param num_relations: The number of relations for the encoder.
        :param num_bases: The number of base relations.
        :param num_layers_conv: The number of layers for the convolutional encoder.
        :param hidden_size: The size of the hidden layer.
        :param columns: List of columns for collecting set embeddings.
        :param out_channels: The number of output channels.
        :param emb_df: DataFrame containing the embeddings.
        :param pos_weight: Positive class weights for loss calculation.
        :param task_weight: Weighting strategy for the tasks.
        :param dropout: Dropout rate for the model.
        :param aggregate: Aggregation method for embeddings.
        :param set_embedding_mode: Mode of set embedding aggregation ('mean' or 'sum').
        :param trial_feats_size: Size of the trial features (optional).
        :param only_trial_feats: Whether to only use trial features.
        """
        super().__init__()

        self.aggregate = aggregate
        self.use_trial_feats = trial_feats_size != -1
        self.only_trial_feats = only_trial_feats

        self.trial_feats_proj = nn.Linear(trial_feats_size, hidden_size)

        if not self.only_trial_feats:
            self.embedding = MixedEmbedding(embedding_dim=hidden_size, num_embeddings=in_channels, emb_df=emb_df)
            self.encoder = RGCNEncoderFull(in_channels=hidden_size, num_relations=num_relations,
                                           hidden_size=hidden_size,
                                           num_bases=num_bases, num_layers=num_layers_conv, dropout=dropout,
                                           edge_dropout=0)

            self.collectors = nn.ModuleDict(
                {column: SetEmbedding(hidden_size, mode=set_embedding_mode) for column in columns})

            if aggregate == 'concat':
                inp_size = hidden_size * len(self.collectors)
            elif aggregate in ['mean', 'sum', 'weighted_sum']:
                inp_size = hidden_size
            else:
                raise RuntimeError("Unexpected aggregate method")

            if aggregate == 'weighted_sum':
                self.agg_layer = nn.Linear(len(columns), 1, bias=False)

            if self.use_trial_feats:
                inp_size += hidden_size
        else:
            inp_size = hidden_size

        # Output module consisting of linear layers, batch normalization, and ReLU activation
        self.output_module = nn.Sequential(
            nn.Linear(inp_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, out_channels)
        )

        self.padding_idx = -1

        # Initialize the task loss function
        self.task_loss = TaskLoss(len(pos_weight), pos_weight=pos_weight, weight=task_weight)

    def encode(self, x, adjs, edge_type, devices: List[int]):
        """
        Encodes the input data and adjacency matrices.

        :param x: Input node embeddings.
        :param adjs: List of adjacency matrices.
        :param edge_type: Edge types for the graph.
        :param devices: List of devices for model distribution.
        :return: The encoded output from the model.
        """
        if self.only_trial_feats:
            return torch.tensor([0])
        x = self.embedding(x.to(devices[0]))
        return self.encoder(x, adjs, edge_type, devices[1:])

    def to_devices(self, devices):
        """
        Distributes the model to specified devices.

        :param devices: List of devices for model parallelism.
        :return: The model moved to the devices.
        """
        model = self
        if not self.only_trial_feats:
            model.embedding = model.embedding.to(devices[0])
            k = 1
            for i in range(len(model.encoder.convs)):
                model.encoder.convs[i] = model.encoder.convs[i].to(devices[k])
                k += 1

            model.collectors = model.collectors.to(devices[k])
            if hasattr(model, 'agg_layer'):
                model.agg_layer = model.agg_layer.to(devices[k])
        else:
            k = -1
        if self.use_trial_feats:
            model.trial_feats_proj.to(devices[k])
        model.output_module = model.output_module.to(devices[k])
        model.task_loss = model.task_loss.to(devices[k])
        return model

    def forward(self, embs, trial_data, devices: List[int]):
        """
        The forward pass for the classification model.

        :param embs: The input embeddings.
        :param trial_data: Data related to trials.
        :param devices: List of devices to be used.
        :return: The output logits from the model.
        """
        if self.only_trial_feats:
            trial_x = self.trial_feats_proj(trial_data['trial_feats'].to(devices[-1]))
            return self.output_module(trial_x)
        xs = []
        for key, data in trial_data.items():
            if key == 'trial_feats':
                continue
            mask = data == self.padding_idx
            data[mask] = 0
            b, l = data.shape
            h = embs[data.reshape(-1)].reshape((b, l, -1))
            h[mask] *= 0
            seq_len = torch.tensor(np.invert(mask).sum(axis=1), dtype=torch.long)
            zero_len = seq_len == 0
            if zero_len.any():
                if key not in ['outcomes', 'excl_criteria', 'incl_criteria']:
                    raise RuntimeError("Zero seq length for ", key)
                else:
                    seq_len[zero_len] = 1
            h = self.collectors[key](h.to(devices[-1]), seq_len.to(devices[-1]))
            xs.append(h)
        if self.aggregate == 'concat':
            x = torch.cat(xs, dim=-1).to(devices[-1])
        elif self.aggregate == 'mean':
            x = torch.stack(xs, dim=0).permute((1, 2, 0))
            x = torch.mean(x, dim=-1)
        elif self.aggregate == 'sum':
            x = torch.stack(xs, dim=0).permute((1, 2, 0))
            x = torch.sum(x, dim=-1)
        elif self.aggregate == 'weighted_sum':
            x = torch.stack(xs, dim=0).permute((1, 2, 0))
            x = self.agg_layer(x)
            x = x.squeeze()
        else:
            raise RuntimeError("Unexpected aggregate method")

        if self.use_trial_feats:
            trial_x = self.trial_feats_proj(trial_data['trial_feats'].to(devices[-1]))
            x = torch.cat([x, trial_x], dim=-1)

        return self.output_module(x)

    def loss(self, embs, scores, labels):
        """
        Computes the loss for the model.

        :param embs: The embeddings.
        :param scores: The predicted scores.
        :param labels: The ground truth labels.
        :return: The computed loss.
        """
        return self.task_loss(scores, labels.to(scores.device))


class PretrainModel(torch.nn.Module):
    """
    A model for pretraining tasks using knowledge graph embeddings.
    """
    def __init__(self, *, clf_model: ClassificationModel, num_relations, hidden_dim,
                 decoder_type: str = 'TransE',
                 gamma: float = 6.0,
                 negative_adversarial_sampling: bool = True,
                 adversarial_temperature: float = 1,
                 reg_param=0):
        """
        Initializes the pretraining model with classification and decoder components.

        :param clf_model: The classification model.
        :param num_relations: The number of relations in the knowledge graph.
        :param hidden_dim: The hidden dimension size.
        :param decoder_type: The decoder type ('TransE' or 'DistMult').
        :param gamma: The margin for TransE decoder.
        :param negative_adversarial_sampling: Whether to use negative adversarial sampling.
        :param adversarial_temperature: Temperature for the adversarial sampling.
        :param reg_param: Regularization parameter.
        """
        super().__init__()
        self.clf_model = clf_model
        if decoder_type == 'TransE':
            self.decoder = TransEDecoder(gamma=gamma, num_rels=num_relations, h_dim=hidden_dim)
        elif decoder_type == 'DistMult':
            self.decoder = DistMultDecoder(num_rels=num_relations, h_dim=hidden_dim)
        self.reg_param = reg_param
        self.negative_adversarial_sampling = negative_adversarial_sampling
        self.adversarial_temperature = adversarial_temperature

    def forward(self, embs, triplets, devices: List[int], mode: str = 'train'):
        """
        Performs the forward pass for the pretraining model.

        :param embs: The input embeddings.
        :param triplets: The triplet samples (head, relation, tail).
        :param devices: The list of devices to use.
        :param mode: The mode ('train' or 'eval').
        :return: The computed scores.
        """
        embs = embs.to(devices[3])
        pos_samples, head_negative_sample, tail_negative_sample = triplets
        positive_score = self.decoder(embs, pos_samples)

        if mode == 'eval':
            return embs, positive_score

        head_neg_scores = self.decoder(embs, (pos_samples, head_negative_sample), mode='head-batch')
        tail_neg_scores = self.decoder(embs, (pos_samples, tail_negative_sample), mode='tail-batch')

        negative_score = torch.cat([head_neg_scores, tail_neg_scores], dim=-1)

        return positive_score, negative_score

    def regularization_loss(self, embedding):
        """
        Computes the regularization loss for embeddings.

        :param embedding: The input embeddings.
        :return: The regularization loss.
        """
        return self.encoder.reg_loss(embedding) + self.decoder.reg_loss()

    def to_devices(self, devices):
        """
        Distributes the pretraining model to devices.

        :param devices: List of devices for model parallelism.
        :return: The pretraining model moved to the devices.
        """
        self.clf_model = self.clf_model.to(devices)
        self.decoder = self.decoder.to(devices[0])
        return self

    def loss(self, embs, scores):
        # triplets is a list of data samples (positive and negative)
        # each row in the triplets is a 3-tuple of (source, relation, destination)
        positive_score, negative_score = scores
        if self.negative_adversarial_sampling:
            # In self-adversarial sampling, we do not apply back-propagation on the sampling weight
            negative_score = (F.softmax(negative_score * self.adversarial_temperature, dim=1).detach()
                              * F.logsigmoid(-negative_score)).sum(dim=1)
        else:
            negative_score = F.logsigmoid(-negative_score).mean(dim=1)

        positive_score = F.logsigmoid(positive_score).squeeze(dim=1)

        positive_sample_loss = - positive_score.mean()
        negative_sample_loss = - negative_score.mean()

        loss = (positive_sample_loss + negative_sample_loss) / 2

        return loss, positive_sample_loss, negative_sample_loss
