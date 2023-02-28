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
                 bert_dim=0, #0 means not combining bert
                 ):
        super().__init__()
        self.embedding = MixedEmbedding(embedding_dim=hidden_dim[0], num_embeddings=in_channels, emb_df=emb_df)
        self.bert_dim = bert_dim
        if bert_dim == 0:
            self.encoder = RGCNEncoder(conv_fn=conv_fn, conv_aggr=conv_aggr, in_channels=hidden_dim[0],
                                       hidden_size=hidden_dim[1:],
                                       num_relations=num_relations,
                                       num_bases=num_bases, num_layers=num_enc_layers,
                                       activation_fn=activation_fn, norm_fn=norm_fn,
                                       dropout=dropout, edge_dropout=edge_dropout,
                                       res_method='res' if add_residual else 'plain',
                                       pre_layers=num_pre_layers,
                                       post_layers=num_post_layers,
                                       nbr_concat=nbr_concat,
                                       nbr_concat_weight=nbr_concat_weight)
        else:
            self.encoder = RGCNEncoderWithBERT(conv_fn=conv_fn, conv_aggr=conv_aggr, in_channels=hidden_dim[0],
                                       hidden_size=hidden_dim[1:],
                                       num_relations=num_relations,
                                       num_bases=num_bases, num_layers=num_enc_layers,
                                       activation_fn=activation_fn, norm_fn=norm_fn,
                                       dropout=dropout, edge_dropout=edge_dropout,
                                       res_method='res' if add_residual else 'plain',
                                       pre_layers=num_pre_layers,
                                       post_layers=num_post_layers,
                                       nbr_concat=nbr_concat,
                                       nbr_concat_weight=nbr_concat_weight,
                                       bert_dim=bert_dim)
        if decoder_type == 'TransE':
            self.decoder = TransEDecoder(gamma=gamma, num_rels=num_relations // 2, h_dim=hidden_dim[-1])
        elif decoder_type == 'DistMult':
            self.decoder = DistMultDecoder(num_rels=num_relations // 2, h_dim=hidden_dim[-1])
        self.reg_param = reg_param
        self.negative_adversarial_sampling = negative_adversarial_sampling
        self.adversarial_temperature = adversarial_temperature

    def to_devices(self, devices):
        model = self
        model.embedding = model.embedding.to(devices[0])
        model.encoder = model.encoder.to_devices(devices[1:])
        if hasattr(model, 'decoder'):
            model.decoder = model.decoder.to(devices[3])
        return model

    def encode(self, x, adjs, edge_type, devices: List[int]):
        if getattr(self, 'bert_dim', 0) == 0:
            x = self.embedding(x.to(devices[0]))
            # return x
            return self.encoder(x, adjs, edge_type, devices[1:])
        else:
            gcn_x, orig_node_x, bert_x = x
            _bs = bert_x.size(0)
            assert _bs == orig_node_x.size(0)
            assert (gcn_x[:_bs] == orig_node_x).long().sum() == _bs
            gcn_x = self.embedding(gcn_x.to(devices[0]))
            gcn_x, bert_x = self.encoder(gcn_x, bert_x, adjs, edge_type, devices[1:])
            return gcn_x, bert_x

    def forward(self, embs, triplets, devices: List[int], mode: str = 'train'):
        embs = embs.to(devices[3])
        pos_samples, head_negative_sample, tail_negative_sample = triplets
        #pos_samples: list[h, r, t], where each of h, r, t is [bs (= n_triple), ]
        #head_negative_sample: [bs, n_neg]
        #tail_negative_sample: [bs, n_neg]
        positive_score = self.decoder(embs, pos_samples) #[bs, 1]

        if mode == 'eval':
            return embs, positive_score

        head_neg_scores = self.decoder(embs, (pos_samples, head_negative_sample), mode='head-batch')
        tail_neg_scores = self.decoder(embs, (pos_samples, tail_negative_sample), mode='tail-batch')

        negative_score = torch.cat([head_neg_scores, tail_neg_scores], dim=-1) #[bs, total_n_neg]

        return positive_score, negative_score

    def regularization_loss(self, embedding, devices):
        # e1s, rs, e2s = self.compute_codes(mode='train')
        # regularization = tf.reduce_mean(tf.square(e1s))
        # regularization += tf.reduce_mean(tf.square(rs))
        # regularization += tf.reduce_mean(tf.square(e2s))
        #
        # return self.regularization_parameter * regularization
        # return self.encoder.reg_loss(embedding).to(devices[3]) + self.decoder.reg_loss().to(devices[3])
        return self.decoder.reg_loss().to(devices[3])

    def loss(self, embs, scores, devices):
        # triplets is a list of data samples (positive and negative)
        # each row in the triplets is a 3-tuple of (source, relation, destination)
        positive_score, negative_score = scores
        if self.negative_adversarial_sampling:
            # In self-adversarial sampling, we do not apply back-propagation on the sampling weight
            negative_score = (F.softmax(negative_score * self.adversarial_temperature, dim=1).detach()
                              * F.logsigmoid(-negative_score)).sum(dim=1)
        else:
            negative_score = F.logsigmoid(-negative_score).mean(dim=1) #[bs,]

        positive_score = F.logsigmoid(positive_score).squeeze(dim=1) #[bs,]

        positive_sample_loss = - positive_score.mean() #scalar
        negative_sample_loss = - negative_score.mean() #scalar

        loss = (positive_sample_loss + negative_sample_loss) / 2 + self.reg_param * self.regularization_loss(embs,
                                                                                                             devices)

        return loss, positive_sample_loss, negative_sample_loss


class EmbeddingNormalize(nn.Module):
    def __init__(self, with_grad: bool):
        super(EmbeddingNormalize, self).__init__()
        self.with_grad = with_grad
        self.epsilon = 1e-12

    def forward(self, x):
        if self.with_grad:
            return F.normalize(x, p=2, dim=1)
        else:
            norm = x.norm(p=2, dim=1, keepdim=True).detach()
            return x / norm

    def __repr__(self):
        return f'{self.__class__.__name__}(with_grad={self.with_grad})'


class RelationAugment(nn.Module):
    def __init__(self, r_embs, decoder_type, aggr='mean'):
        super(RelationAugment, self).__init__()
        # TODO: write for transe as well
        self.register_buffer('r_embs', nn.Parameter(r_embs))
        self.decoder_type = decoder_type
        assert decoder_type in ['DistMult']
        self.aggr = aggr
        assert aggr in ['mean', 'sum']

    def forward(self, x):
        if self.decoder_type == 'DistMult':
            xs = [x]
            for r_emb in self.r_embs:
                xs.append(x * r_emb)
            if self.aggr == 'mean':
                return torch.stack(xs, dim=1).mean(dim=1)
            elif self.aggr == 'sum':
                return torch.stack(xs, dim=1).sum(dim=1)


class ClassificationModel(nn.Module):
    def __init__(self, *, input_dim: int, hidden_sizes: List[int], num_layers: int,
                 normalize_embeddings: bool, normalize_embeddings_with_grad: bool,
                 normalize_clf_weights: bool,
                 norm_layer, activation_fn,
                 dropout_prob: float, tasks, args):
        super().__init__()
        self.args = args
        layers = []
        self.normalize_clf_weights = normalize_clf_weights
        # if len(relation_embs) > 0:
        #     layers.append(RelationAugment(relation_embs, decoder_type=decoder_type, aggr=rproj_aggr))
        if normalize_embeddings:
            layers.append(EmbeddingNormalize(normalize_embeddings_with_grad))
        assert num_layers == len(hidden_sizes) + 1
        for i in range(num_layers - 1):
            # if i == num_layers - 2:
            #     p = 0
            # else:
            #     p = dropout_prob
            p = dropout_prob
            hidden_dim = hidden_sizes[i]
            layers.extend([
                nn.Linear(input_dim, hidden_dim),
                norm_layer(hidden_dim),
                activation_fn(),
                nn.Dropout(p=p)
            ])
            input_dim = hidden_dim
        self.representation = nn.Sequential(*layers)
        # layers.append(nn.Linear(input_dim, num_tasks))
        assert len(tasks) == 1
        task_heads = []
        task_loss = []
        task_weights = []
        for task_params in tasks:
            task_name = task_params['name']
            num_subtasks = task_params['num_subtasks']
            print ('task_name', task_name, 'num_subtasks', num_subtasks)
            task_heads.append(nn.Linear(input_dim*2 if task_name=='binary_pair_efficacy' else input_dim, num_subtasks))
            task_weights.append(task_params['task_weight'])
            if task_name in ['ae_clf', 'binary', 'ae_clf_or', 'ae_clf_freq', 'binary_or', 'binary_pair_efficacy', 'with_or_without_results']:
                task_loss.append(ClassificationTaskLoss(num_tasks=num_subtasks, weight=task_params['subtask_weight'],
                                                        pos_weight=task_params['pos_weight']))
            elif task_name in ['ae_clf_or_l']:
                task_loss.append(
                    ClassificationMeanTaskLoss(num_tasks=num_subtasks, weight=task_params['subtask_weight'],
                                               pos_weight=task_params['pos_weight']))
            elif task_name == 'ae_regr':
                task_loss.append(RegressionTaskLoss(num_tasks=num_subtasks, weight=task_params['subtask_weight']))
            else:
                raise RuntimeError("Invalid task name " + task_name)
        self.task_heads = nn.ModuleList(task_heads)
        self.task_losses = nn.ModuleList(task_loss)
        self.task_weights = task_weights
        # self.total_loss = CombinationLoss(num_tasks=len(tasks), weight='variance')
        self.version = 2

    def to_device(self, device):
        return self.to(device)
        # version = self.version if hasattr(self, 'version') else 1
        # model = self
        # if version == 1:
        #     model.classifier = model.classifier.to(device)
        #     model.task_loss = model.task_loss.to(device)
        #     return model
        # elif version == 2:
        #     model.representation = model.representation.to(device)
        #     model.task_heads = model.task_heads.to(device)
        #     model.task_losses = model.task_losses.to(device)
        #     return model

    def forward(self, x, devices):
        version = self.version if hasattr(self, 'version') else 1
        if self.normalize_clf_weights:
            self.classifier[-1].weight.data = F.normalize(self.classifier[-1].weight.data, p=2, dim=1)
        if version == 1:
            return self.classifier(x.to(devices[-1]))
        elif version == 2:
            x = self.representation(x.to(devices[-1])) #[batch_size, dim]
            if self.args.default_task_name == 'binary_pair_efficacy':
                #x is [batch_size + batch_size, dim]
                _bs, _dim = x.size()
                _bs = _bs//2
                x = torch.cat(x.view(2,_bs,_dim).unbind(dim=0), dim=1) #[batch_size, dim + dim]
            return [module(x) for module in self.task_heads]

    def loss(self, scores, labels):
        version = self.version if hasattr(self, 'version') else 1
        if version == 1:
            return self.task_loss(scores, labels.to(scores.device))
        elif version == 2:
            # return self.total_loss(
            #     [self.task_losses[i](scores[i], labels[i].to(scores[i].device)) for i in range(len(self.task_losses))])
            loss = 0
            for i, weight in enumerate(self.task_weights):
                loss += weight * self.task_losses[i](scores[i], labels[i].to(scores[i].device),
                                                     labels[i + len(self.task_weights)].to(scores[i].device))
            return loss


class ClassificationModelWithAEEmb(nn.Module):
    def __init__(self, *, input_dim: int, hidden_sizes: List[int], num_layers: int,
                 normalize_embeddings: bool, normalize_embeddings_with_grad: bool,
                 normalize_clf_weights: bool,
                 norm_layer, activation_fn,
                 dropout_prob: float, tasks, args):
        super().__init__()
        layers = []
        self.normalize_clf_weights = normalize_clf_weights
        # if len(relation_embs) > 0:
        #     layers.append(RelationAugment(relation_embs, decoder_type=decoder_type, aggr=rproj_aggr))
        if normalize_embeddings:
            layers.append(EmbeddingNormalize(normalize_embeddings_with_grad))
        assert num_layers == len(hidden_sizes) + 1
        for i in range(num_layers - 1):
            p = dropout_prob
            hidden_dim = hidden_sizes[i]
            layers.extend([
                nn.Linear(input_dim, hidden_dim),
                norm_layer(hidden_dim),
                activation_fn(),
                nn.Dropout(p=p)
            ])
            input_dim = hidden_dim
        self.representation = nn.Sequential(*layers)
        AEembdict = pkl.load(open('../data/clf_data/PT/AE_embdict_OR2.pkl', 'rb'))

        task_heads = []
        task_loss = []
        task_weights = []

        assert len(tasks) == 1
        task_params = tasks[0]
        task_name = task_params['name']
        num_subtasks = task_params['num_subtasks']
        idx2aename = task_params['idx2aename']
        print ('task_name', task_name, 'num_subtasks', num_subtasks)
        assert task_name in ['ae_clf_or']

        AEemb = torch.stack([torch.tensor(AEembdict[aename]) for aename in idx2aename])
        self.AEemb = nn.Parameter(AEemb)
        assert num_subtasks == AEemb.size(0)
        task_heads.append(nn.Linear(input_dim + AEemb.size(1), num_subtasks))
        task_weights.append(task_params['task_weight'])

        if task_name in ['ae_clf', 'binary', 'ae_clf_or', 'ae_clf_freq', 'binary_or']:
            print ('task_params[pos_weight]', task_params['pos_weight'])
            task_loss.append(ClassificationTaskLoss(num_tasks=num_subtasks, weight=task_params['subtask_weight'],
                                                    pos_weight=task_params['pos_weight']))
        elif task_name in ['ae_clf_or_l']:
            task_loss.append(
                ClassificationMeanTaskLoss(num_tasks=num_subtasks, weight=task_params['subtask_weight'],
                                           pos_weight=task_params['pos_weight']))
        elif task_name == 'ae_regr':
            task_loss.append(RegressionTaskLoss(num_tasks=num_subtasks, weight=task_params['subtask_weight']))
        else:
            raise RuntimeError("Invalid task name " + task_name)

        self.task_heads = nn.ModuleList(task_heads)
        self.task_losses = nn.ModuleList(task_loss)
        self.task_weights = task_weights
        self.version = 2

    def to_device(self, device):
        # self.AEemb = self.AEemb.to(device)
        self.AEemb.to(device)
        return self.to(device)

    def forward(self, x, devices):
        version = self.version if hasattr(self, 'version') else 1
        if self.normalize_clf_weights:
            self.classifier[-1].weight.data = F.normalize(self.classifier[-1].weight.data, p=2, dim=1)
        if version == 1:
            return self.classifier(x.to(devices[-1]))
        elif version == 2:
            x = self.representation(x.to(devices[-1])) #[bs, hiddim]
            x = x.unsqueeze(1).repeat(1, self.AEemb.size(0), 1) #[bs, 931, hiddim]
            x = torch.cat([x, self.AEemb.unsqueeze(0).repeat(x.size(0), 1, 1)], dim=2) #[bs, 931, hiddim+768]
            # print ('x.size()', x.size())
            return [(module.weight.unsqueeze(0) * x).sum(dim=2) + module.bias.unsqueeze(0) for module in self.task_heads]

    def loss(self, scores, labels):
        version = self.version if hasattr(self, 'version') else 1
        if version == 1:
            return self.task_loss(scores, labels.to(scores.device))
        elif version == 2:
            loss = 0
            for i, weight in enumerate(self.task_weights):
                loss += weight * self.task_losses[i](scores[i], labels[i].to(scores[i].device),
                                                     labels[i + len(self.task_weights)].to(scores[i].device))
            return loss
