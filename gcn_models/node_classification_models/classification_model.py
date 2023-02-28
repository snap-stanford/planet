from typing import List

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from gcn_models.decoders import TransEDecoder, DistMultDecoder
from gcn_models.encoders import RGCNEncoderFull


class MixedEmbedding(nn.Module):
    def __init__(self, *, embedding_dim, num_embeddings, emb_df=None):
        super(MixedEmbedding, self).__init__()
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        print ('self.num_embeddings', self.num_embeddings)

        if emb_df is None:
            self.transforms = nn.ModuleList(
                [nn.Embedding(embedding_dim=embedding_dim, num_embeddings=num_embeddings)]
            )
            return
        max_len = emb_df['emb'].map(len).max()

        node_ids = emb_df['id'].values
        need_embedding = np.ones(num_embeddings, dtype=np.bool)
        need_embedding[node_ids] = False
        embedding_node_ids = np.arange(num_embeddings)[need_embedding]
        emb_inverse_index = np.zeros(num_embeddings) - 1
        emb_inverse_index[embedding_node_ids] = np.arange(len(embedding_node_ids))
        self.register_buffer('emb_inverse_index', torch.tensor(emb_inverse_index, dtype=torch.long))
        self.transforms = nn.ModuleList(
            [nn.Embedding(embedding_dim=embedding_dim, num_embeddings=len(embedding_node_ids))] +
            [nn.Linear(max_len, self.embedding_dim, bias=False) for _ in emb_df.etype.cat.categories]
        )

        # TODO: remove this mean after correcting the proein features thing
        emb = emb_df['emb'].map(lambda x: x.mean(axis=0) if len(x.shape) > 1 else x) \
            .map(lambda x: np.pad(np.nan_to_num(x), (0, max_len - len(x))))

        emb = torch.tensor(np.stack(emb.values), dtype=torch.float32)
        # +1 as 0 is learned embeddings
        etype = torch.tensor(emb_df['etype'].cat.codes.values + 1, dtype=torch.uint8)

        fixed_emb = torch.zeros((self.num_embeddings, max_len), dtype=torch.float32)
        etype_all = torch.zeros(self.num_embeddings, dtype=torch.uint8)

        fixed_emb[node_ids] = emb
        etype_all[node_ids] = etype

        self.register_buffer('fixed_emb', fixed_emb)
        self.register_buffer('etype', etype_all)

    def forward(self, x):
        if len(self.transforms) == 1:
            return self.transforms[0](x)
        e_type = self.etype[x]
        out = torch.zeros((len(x), self.embedding_dim), device=x.device)
        for idx, transform in enumerate(self.transforms):
            mask = e_type == idx
            # print ('idx', idx)
            # print ('mask.size()', mask.size())
            # print ('x.size()', x.size())
            # print ('self.fixed_emb.size()', self.fixed_emb.size())
            # print ('self.emb_inverse_index.size()', self.emb_inverse_index.size())
            x_idx = self.fixed_emb[x[mask]] if idx != 0 else self.emb_inverse_index[x[mask]]
            out[mask] = transform(x_idx)
        return out


class SetEmbedding(nn.Module):
    def __init__(self, channels, mode='mean'):
        super(SetEmbedding, self).__init__()
        assert mode in ['mean', 'sum']
        self.linear = nn.Linear(channels, channels)
        self.mode = mode

    def forward(self, x, seq_len):
        """
        :param seq_len: Tensor of shape B denoting seq lengths
        :param x: Tensor of shape B * N * D
        :return:
        """
        # sum
        # lstm
        # tranformer
        if self.mode == 'mean':
            x = x.sum(dim=1) / seq_len.unsqueeze(1)
        elif self.mode == 'sum':
            x = x.sum(dim=1)
        else:
            raise RuntimeError("Unexpected set embedding mode")
        # print(x.shape)
        return self.linear(x)


class TaskLoss(nn.Module):
    def __init__(self, *, num_tasks, weight='uniform'):
        super(TaskLoss, self).__init__()
        assert weight in ['variance', 'uniform']
        self.weight = weight
        self.num_tasks = num_tasks
        self.log_vars = nn.Parameter(torch.zeros(num_tasks), requires_grad=weight == 'variance')

    def forward(self, input, target, sample_weight):
        loss = self.criterion(input, target) * sample_weight
        loss = loss.mean(dim=0)
        # print(loss.size())
        if self.weight == 'uniform':
            return loss.sum()

        task_weight = torch.exp(-self.log_vars)

        return (loss * task_weight).sum() + self.log_vars.sum()


class ClassificationTaskLoss(TaskLoss):
    def __init__(self, *, pos_weight, **kwargs):
        super(ClassificationTaskLoss, self).__init__(**kwargs)
        self.criterion = nn.BCEWithLogitsLoss(reduction="none", pos_weight=pos_weight)


class ClassificationMeanTaskLoss(TaskLoss):
    def __init__(self, *, pos_weight, **kwargs):
        super(ClassificationMeanTaskLoss, self).__init__(**kwargs)
        self.criterion = nn.BCEWithLogitsLoss()

    def forward(self, input, target, sample_weight):
        loss = self.criterion(input, target)
        return loss


class RegressionTaskLoss(TaskLoss):
    def __init__(self, **kwargs):
        super(RegressionTaskLoss, self).__init__(**kwargs)
        self.criterion = nn.MSELoss(reduction="none")


class CombinationLoss(TaskLoss):
    def forward(self, loss, _=None):
        task_weight = torch.exp(-self.log_vars)
        return sum(task_weight[i] * loss[i] for i in range(len(loss))) + self.log_vars.sum()


class ClassificationModel(nn.Module):
    def __init__(self, *, in_channels, num_relations, num_bases, num_layers_conv, hidden_size, columns, out_channels,
                 emb_df, pos_weight, task_weight,
                 dropout=0.5, aggregate='concat', set_embedding_mode='mean',
                 trial_feats_size: int = -1, only_trial_feats: bool = False):
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

        self.output_module = nn.Sequential(
            nn.Linear(inp_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, out_channels)
        )

        self.padding_idx = -1

        self.task_loss = TaskLoss(len(pos_weight), pos_weight=pos_weight, weight=task_weight)

    def encode(self, x, adjs, edge_type, devices: List[int]):
        if self.only_trial_feats:
            return torch.tensor([0])
        x = self.embedding(x.to(devices[0]))
        return self.encoder(x, adjs, edge_type, devices[1:])

    def to_devices(self, devices):
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
        # print(x.size())

        if self.use_trial_feats:
            trial_x = self.trial_feats_proj(trial_data['trial_feats'].to(devices[-1]))
            x = torch.cat([x, trial_x], dim=-1)

        return self.output_module(x)

    def loss(self, embs, scores, labels):
        return self.task_loss(scores, labels.to(scores.device))


class PretrainModel(torch.nn.Module):
    def __init__(self, *, clf_model: ClassificationModel, num_relations, hidden_dim,
                 decoder_type: str = 'TransE',
                 gamma: float = 6.0,
                 negative_adversarial_sampling: bool = True,
                 adversarial_temperature: float = 1,
                 reg_param=0):
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
        return self.encoder.reg_loss(embedding) + self.decoder.reg_loss()

    def to_devices(self, devices):
        self.clf_model = self.clf_model.to_devices(devices)
        self.decoder = self.decoder.to(devices[-1])
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
