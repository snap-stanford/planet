import math
import sys

import numpy as np
import torch
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve, auc, recall_score, \
    precision_score, f1_score, mean_squared_error, accuracy_score
from torch_geometric.data import NeighborSampler

sys.path.insert(0, "../")
from gcn_models.data_loader import Dataset
from tqdm import tqdm


class Evaluator:
    def __init__(self, dataset: Dataset):
        self.dataset = dataset
        self.graph_samplers = {}
        for key, d in dataset.datasets.items():
            self.graph_samplers[key] = NeighborSampler(d.data.edge_index, node_idx=None,
                                                       sizes=[-1, -1], batch_size=1024,
                                                       shuffle=True, num_workers=0)

    @staticmethod
    def _eval_metric(scores, labels, probs, rng, num_repeats):
        # if probs is not None:
        #     scores = probs
        #     scores = -scores
        #     temp = scores.argsort()
        #     ranks = np.empty_like(temp)
        #     ranks[temp] = np.arange(len(scores))
        # else:
        #     ranks = np.random.permutation(len(labels))
        #
        # ranks = ranks + 1
        # pos_ranks = ranks[labels == 1]
        #
        # ranking = min(pos_ranks)
        #
        # def hitk(k=1):
        #     return sum(1.0 if rank <= k else 0.0 for rank in pos_ranks) / k
        #
        # def rk(k=10):
        #     return sum(1.0 if rank <= k else 0.0 for rank in pos_ranks) / len(pos_ranks)
        #
        # def actual_apk(k=-1):
        #     score = 0
        #     if k < 0:
        #         k = max(pos_ranks) + 1
        #     for i, rank in enumerate(sorted(pos_ranks)):
        #         if rank > k:
        #             break
        #         score += (i + 1) / rank
        #     return score / min(len(pos_ranks), k)

        def auprc(*, y_true, y_score):
            precision, recall, _ = precision_recall_curve(y_true=y_true, probas_pred=y_score)
            return auc(recall, precision)

        def balanced_auprc_(*, y_true, y_score, ratio=1):
            n = y_true.size(0)
            mask = np.zeros_like(y_true).astype(np.bool)
            mask[y_true == 1] = True
            num_pos = int(y_true.sum().item())
            num_negs = n - num_pos
            sampled_num_negs = min(ratio * num_pos, num_negs)
            sampled_negs = rng.choice(np.arange(n)[y_true == 0], sampled_num_negs)
            # mask[y_true == 0] = np.random.randn(y_true.size(0) - num_pos) > 0.5
            mask[sampled_negs] = True
            return auprc(y_true=y_true[mask], y_score=y_score[mask])

        def balanced_auprc(*, y_true, y_score, ratio=1, n=100):
            val = []
            for i in range(n):
                val.append(balanced_auprc_(y_true=y_true, y_score=y_score, ratio=ratio))
            return np.mean(val)

        return {
            # 'MRR': 1.0 / ranking,
            # 'MR': float(ranking),
            # 'HITS@1': hitk(1),
            # 'HITS@3': hitk(3),
            # 'HITS@10': hitk(10),
            # 'HITS@20': hitk(20),
            # 'HITS@50': hitk(50),
            'AVG_POS': labels.sum().item(), #number of examples positive for this task (e.g. one of AE tasks)?
            'avg_aes': labels.size(0), #number of examples (e.g. test set size)
            # "AR@5": rk(5),
            # "AR@10": rk(10),
            # "AR@20": rk(20),
            # "AR@50": rk(50),
            # "AR@100": rk(100),
            # "AP": actual_apk(),
            # "AP50": actual_apk(k=50),
            # "AP20": actual_apk(k=20),
            "roc_auc": roc_auc_score(y_true=labels, y_score=probs),
            # 'avg_precision_score': average_precision_score(y_true=labels, y_score=probs),
            'auprc': auprc(y_true=labels, y_score=probs),
            'balanced_auprc': balanced_auprc(y_true=labels, y_score=probs, n=num_repeats),
            # 'balanced_auprc_2': balanced_auprc(y_true=labels, y_score=probs, ratio=2),
            # 'balanced_auprc_5': balanced_auprc(y_true=labels, y_score=probs, ratio=5),
            # 'balanced_auprc_10': balanced_auprc(y_true=labels, y_score=probs, ratio=10),
            # 'precision': precision_score(y_true=labels, y_pred=probs > 0.5, zero_division=0),
            # 'recall': recall_score(y_true=labels, y_pred=probs > 0.5, zero_division=0),
            # 'f1_score': f1_score(y_true=labels, y_pred=probs > 0.5, zero_division=0)
        }

    @staticmethod
    def _eval_metric_freq(scores, labels, probs, rng):
        def get_class_label(p):
            y = np.zeros_like(p)
            for i in range(len(p)):
                p_i = p[i] * 100
                if p_i < 0.001:
                    y[i] = 0
                if p_i < 0.01:
                    y[i] = 1
                elif p_i < 0.1:
                    y[i] = 2
                elif p_i < 1:
                    y[i] = 3
                elif p_i < 10:
                    y[i] = 4
                else:
                    y[i] = 5
            return y

        y_true = get_class_label(labels)
        y_pred = get_class_label(probs)
        return {
            'rmse': mean_squared_error(y_true=labels, y_pred=probs),
            'AVG_POS': labels.sum().item(),
            'avg_aes': labels.shape[0],
            'precision': precision_score(y_true=y_true, y_pred=y_pred, zero_division=0, average='weighted'),
            'recall': recall_score(y_true=y_true, y_pred=y_pred, zero_division=0, average='weighted'),
            'f1_score': f1_score(y_true=y_true, y_pred=y_pred, zero_division=0, average='weighted'),
            'accuracy': accuracy_score(y_true=y_true, y_pred=y_pred),
            'precision_micro': precision_score(y_true=y_true, y_pred=y_pred, zero_division=0, average='micro'),
            'recall_micro': recall_score(y_true=y_true, y_pred=y_pred, zero_division=0, average='micro'),
            'f1_score_micro': f1_score(y_true=y_true, y_pred=y_pred, zero_division=0, average='micro'),
            'precision_macro': precision_score(y_true=y_true, y_pred=y_pred, zero_division=0, average='macro'),
            'recall_macro': recall_score(y_true=y_true, y_pred=y_pred, zero_division=0, average='macro'),
            'f1_score_macro': f1_score(y_true=y_true, y_pred=y_pred, zero_division=0, average='macro'),
        }

    @staticmethod
    def rank_evaluate(*, y_score, labels, num_repeats=100):
        rng = np.random.RandomState(42)

        def calc(prefix, y_score, y_true):
            logs = []
            logs_full = []
            for i in tqdm(range(len(y_score))): #for each AE
                nae = sum(y_true[i])
                if nae == 0 or nae == len(y_true[i]):
                    logs_full.append(None)
                    continue
                eval_out = Evaluator._eval_metric(y_score[i], y_true[i], y_score[i], rng, num_repeats)
                logs.append(eval_out)
                logs_full.append(eval_out)
            metrics = {}
            if len(logs) > 0:
                metrics[f'{prefix}_n'] = len(logs)
                for metric in logs[0].keys():
                    metrics[f"{prefix}_{metric}"] = sum([log[metric] for log in logs]) / len(logs)
                # weights = [(log['avg_aes'] - log['AVG_POS']) / log['AVG_POS'] for log in logs]
                weights = [log['AVG_POS'] for log in logs]
                metrics[f"{prefix}_auprc_logweighted"] = sum(
                    [log['auprc'] * math.log(w) for log, w in zip(logs, weights)]) / sum([math.log(w) for w in weights])
                metrics[f"{prefix}_auprc_weighted"] = sum(
                    [log['auprc'] * w for log, w in zip(logs, weights)]) / sum(weights)
                for metric in ['AVG_POS', 'avg_aes', 'roc_auc', 'auprc', 'balanced_auprc']:
                    #neg_pos_ratio <= 10
                    filtered_logs = [log[metric] for log in logs if (log['avg_aes'] - log['AVG_POS']) / log['AVG_POS'] <= 10]
                    metrics[f"{prefix}_{metric}_neg_pos<=10"] = sum(filtered_logs)/len(filtered_logs) if len(filtered_logs) > 0 else 0
                    metrics[f"{prefix}_n_neg_pos<=10"] = len(filtered_logs)  if len(filtered_logs) > 0 else 0
                    #pos_count >= 10
                    filtered_logs = [log for log in logs if log['AVG_POS'] >= 10]
                    if len(filtered_logs) > 0:
                        metrics[f"{prefix}_{metric}_pos>=10"] = sum([log[metric] for log in filtered_logs])/len(filtered_logs)
                        metrics[f"{prefix}_n_pos>=10"] = len(filtered_logs)
                        weights = [log['AVG_POS'] for log in filtered_logs]
                        metrics[f"{prefix}_{metric}_pos>=10_logweighted"] = sum([log[metric] * math.log(w) for log, w in zip(filtered_logs, weights)]) / sum([math.log(w) for w in weights])
                        metrics[f"{prefix}_{metric}_pos>=10_weighted"] = sum([log[metric] * w for log, w in zip(filtered_logs, weights)]) / sum(weights)
                    else:
                        metrics[f"{prefix}_{metric}_pos>=10"] = 0
                        metrics[f"{prefix}_n_pos>=10"] = 0
                        metrics[f"{prefix}_{metric}_pos>=10_logweighted"] = 0
                        metrics[f"{prefix}_{metric}_pos>=10_weighted"] = 0
                    #pos_count >= 15
                    filtered_logs = [log for log in logs if log['AVG_POS'] >= 15]
                    if len(filtered_logs) > 0:
                        metrics[f"{prefix}_{metric}_pos>=15"] = sum([log[metric] for log in filtered_logs])/len(filtered_logs)
                        metrics[f"{prefix}_n_pos>=15"] = len(filtered_logs)
                        weights = [log['AVG_POS'] for log in filtered_logs]
                        metrics[f"{prefix}_{metric}_pos>=15_logweighted"] = sum([log[metric] * math.log(w) for log, w in zip(filtered_logs, weights)]) / sum([math.log(w) for w in weights])
                        metrics[f"{prefix}_{metric}_pos>=15_weighted"] = sum([log[metric] * w for log, w in zip(filtered_logs, weights)]) / sum(weights)
                    else:
                        metrics[f"{prefix}_{metric}_pos>=15"] = 0
                        metrics[f"{prefix}_n_pos>=15"] = 0
                        metrics[f"{prefix}_{metric}_pos>=15_logweighted"] = 0
                        metrics[f"{prefix}_{metric}_pos>=15_weighted"] = 0
                    #pos_count >= 20
                    filtered_logs = [log for log in logs if log['AVG_POS'] >= 20]
                    if len(filtered_logs) > 0:
                        metrics[f"{prefix}_{metric}_pos>=20"] = sum([log[metric] for log in filtered_logs])/len(filtered_logs)
                        metrics[f"{prefix}_n_pos>=20"] = len(filtered_logs)
                        weights = [log['AVG_POS'] for log in filtered_logs]
                        metrics[f"{prefix}_{metric}_pos>=20_logweighted"] = sum([log[metric] * math.log(w) for log, w in zip(filtered_logs, weights)]) / sum([math.log(w) for w in weights])
                        metrics[f"{prefix}_{metric}_pos>=20_weighted"] = sum([log[metric] * w for log, w in zip(filtered_logs, weights)]) / sum(weights)
                    else:
                        metrics[f"{prefix}_{metric}_pos>=20"] = 0
                        metrics[f"{prefix}_n_pos>=20"] = 0
                        metrics[f"{prefix}_{metric}_pos>=20_logweighted"] = 0
                        metrics[f"{prefix}_{metric}_pos>=20_weighted"] = 0

                metrics[f"{prefix}_logs"] = logs
            metrics[f"{prefix}_logs_full"] = logs_full
            return metrics

        metrics_all = {}
        # metrics_all.update(calc('trial', y_score, labels))
        metrics_all.update(calc('ae', y_score.T, labels.T))
        return metrics_all

    @staticmethod
    def freq_evaluate(*, y_score, labels):
        rng = np.random.RandomState(42)
        y_score = y_score.cpu().numpy()
        labels = labels.cpu().numpy()

        def calc(prefix, *, y_score, y_true):
            logs = []
            for i in range(len(y_score)):
                nae = sum(y_true[i])
                if nae == 0 or nae == len(y_true[i]):
                    continue
                logs.append(Evaluator._eval_metric_freq(y_score[i], y_true[i], y_score[i], rng))
            metrics = {}
            if len(logs) > 0:
                metrics[f'{prefix}_n'] = len(logs)
                for metric in logs[0].keys():
                    metrics[f"{prefix}_{metric}"] = sum([log[metric] for log in logs]) / len(logs)

                metrics[f"{prefix}_logs"] = logs
            return metrics

        metrics_all = {}
        # metrics_all.update(calc('trial', y_score, labels))
        metrics_all.update(calc('ae', y_score=y_score.T, y_true=labels.T))
        return metrics_all

    # @torch.no_grad()
    # def embeddings(self, trainer, batch_size, step, split='valid'):
    #     model = trainer.put_model(trainer.devices['train'], trainer.devices['valid'])
    #     trials = self.dataset.all_trials
    #     aes = self.dataset.all_aes
    #     nodes = torch.cat([trials, aes])
    #     graph_sampler = trainer.samplers[split].graph_sampler
    #     data = trainer.data[split]
    #     model.eval()
    #     writer: SummaryWriter = trainer.writer
    #     for nodes in [trials, aes]:
    #         _, n_id, adjs = graph_sampler.sample(nodes)
    #         emb0 = model.embedding(torch.tensor(nodes, dtype=torch.long, device=trainer.devices['train']))
    #         emb1 = model.encode(data.x[n_id], adjs, data.edge_type, trainer.devices['train'])

    @torch.no_grad()
    def evaluate(self, trainer, batch_size, split='valid'):
        model = trainer.put_model()
        pos_pairs, neg_pairs = self.dataset.trial_ae_triples(split)
        h = np.concatenate([pos_pairs[0], neg_pairs[0]])
        t = np.concatenate([pos_pairs[1], neg_pairs[1]])
        pbar = trainer.pbar
        labels = np.concatenate([np.ones_like(pos_pairs[1]), np.zeros_like(neg_pairs[1])])
        graph_sampler = self.graph_samplers[split]
        data = trainer.data[split]
        scores = []
        probs = []
        model.eval()
        pbar.reset(total=len(h))
        for i in range(0, len(h), batch_size):
            h_i, t_i = h[i:i + batch_size], t[i:i + batch_size]
            r_i = np.ones_like(h_i) * self.dataset.dataset.relation2id[self.dataset.rel]
            nodes, nodes_idx = np.unique(np.concatenate([h_i, t_i]), return_inverse=True)
            _, n_id, adjs = graph_sampler.sample(nodes)
            embs = model.encode(data.x[n_id], adjs, data.edge_type, trainer.devices)
            _, score = model(embs, [(nodes_idx[:len(h_i)], r_i, nodes_idx[len(h_i):]), None, None],
                             trainer.devices, mode='eval')
            score = score.squeeze()
            scores.append(score.detach().cpu().numpy())
            probs.append(torch.sigmoid(score).detach().cpu().numpy())
            pbar.update(batch_size)
        scores = np.concatenate(scores)
        probs = np.concatenate(probs)

        logs_t = []
        trials = self.dataset.trials_by_split[split]
        pbar.reset(total=len(trials))
        cnt_t = 0
        for trial in trials:
            mask = h == trial
            if labels[mask].sum() == 0:
                cnt_t += 1
                continue
            logs_t.append(self._eval_metric(scores[mask], labels[mask], probs[mask]))
            pbar.update()

        logs_ae = []
        aes = self.dataset.all_aes
        pbar.reset(total=len(aes))
        cnt_ae = 0
        for ae in aes:
            mask = t == ae
            if labels[mask].sum() == 0:
                cnt_ae += 1
                continue
            logs_ae.append(self._eval_metric(scores[mask], labels[mask], probs[mask]))
            pbar.update()

        metrics = {}
        if len(logs_t) > 0:
            for prefix, p_logs in [('trial', logs_t), ('ae', logs_ae)]:
                metrics[f'{prefix}_n'] = len(p_logs)
                for metric in p_logs[0].keys():
                    metrics[f"{prefix}_{metric}"] = sum([log[metric] for log in p_logs]) / len(p_logs)

        # metrics['roc_auc'] = roc_auc_score(labels, probs)
        metrics['labels_count'] = np.unique(labels, return_counts=True)
        print(split, cnt_t, cnt_ae)
        return (logs_t, logs_ae), metrics
