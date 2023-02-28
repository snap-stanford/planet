import torch
from torch import nn


class Decoder(nn.Module):
    def __init__(self, num_rels, h_dim):
        super().__init__()
        self.num_relations = num_rels
        self.embedding_dim = h_dim
        self.register_parameter('w_relation', nn.Parameter(torch.Tensor(num_rels, h_dim)))
        nn.init.xavier_uniform_(self.w_relation,
                                gain=nn.init.calculate_gain('relu'))

    def forward(self, embs, sample, mode='single'):
        """
        Forward function that calculate the score of a batch of triples.
        In the 'single' mode, sample is a batch of triple.
        In the 'head-batch' or 'tail-batch' mode, sample consists two part.
        The first part is usually the positive sample.
        And the second part is the entities in the negative samples.
        Because negative samples and positive samples usually share two elements
        in their triple ((head, relation) or (relation, tail)).
        """

        if mode == 'single':
            batch_size, negative_sample_size = sample[0].shape[0], 1

            head = embs[sample[0]].unsqueeze(1) #[bs, 1, dim]
            relation = self.w_relation[sample[1]].unsqueeze(1) #[bs, 1, dim]
            tail = embs[sample[2]].unsqueeze(1) #[bs, 1, dim]

        elif mode == 'head-batch':
            tail_part, head_part = sample
            batch_size, negative_sample_size = head_part.shape

            head = embs[head_part.reshape(-1)].view(batch_size, negative_sample_size, -1) #[bs, n_neg, dim]
            relation = self.w_relation[tail_part[1]].unsqueeze(1) #[bs, 1, dim]
            tail = embs[tail_part[2]].unsqueeze(1) #[bs, 1, dim]

        elif mode == 'tail-batch':
            head_part, tail_part = sample
            batch_size, negative_sample_size = tail_part.shape

            head = embs[head_part[0]].unsqueeze(1)
            relation = self.w_relation[head_part[1]].unsqueeze(1)

            tail = embs[tail_part.reshape(-1)].view(batch_size, negative_sample_size, -1)

        else:
            raise ValueError('mode %s not supported' % mode)

        score = self.score(head, relation, tail, mode) #[bs, 1 or n_neg]

        return score

    def score(self, h, r, t, mode):
        raise NotImplementedError

    def reg_loss(self):
        return torch.mean(self.w_relation.pow(2))
        # return torch.tensor(0)


class TransEDecoder(Decoder):
    """TransE score function
    Paper link: https://papers.nips.cc/paper/5071-translating-embeddings-for-modeling-multi-relational-data
    """

    def __init__(self, gamma, num_rels, h_dim, dist_func='l2'):
        super().__init__(num_rels, h_dim)
        # TODO: store gamma and dist ord into parametrs so tha they are perssted in modue dict

        self.gamma = gamma
        if dist_func == 'l1':
            dist_ord = 1
        else:  # default use l2
            dist_ord = 2
        self.dist_ord = dist_ord

    def score(self, head, relation, tail, mode):
        if mode == 'head-batch':
            score = head + (relation - tail)
        else:
            score = (head + relation) - tail

        score = self.gamma - torch.norm(score, p=self.dist_ord, dim=2)
        return score

    def __repr__(self):
        return '{}(embedding_size={}, num_relations={}, gamma={}, dist_ord={})'.format(self.__class__.__name__,
                                                                                       self.embedding_dim,
                                                                                       self.num_relations,
                                                                                       self.gamma,
                                                                                       self.dist_ord)


class DistMultDecoder(Decoder):
    """DistMult score function
        Paper link: https://arxiv.org/abs/1412.6575
    """

    def score(self, head, relation, tail, mode):
        if mode == 'head-batch':
            score = head * (relation * tail)
        else:
            score = (head * relation) * tail

        score = score.sum(dim=2)
        return score

    def __repr__(self):
        return '{}(embedding_size={}, num_relations={})'.format(self.__class__.__name__,
                                                                self.embedding_dim,
                                                                self.num_relations)
