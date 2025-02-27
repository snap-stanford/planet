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
    """
    Evaluator class for computing various evaluation metrics for models using graph data.
    
    This class provides methods to evaluate ranking and frequency-based performance, as well as
    an end-to-end evaluation method that computes model scores and aggregates metrics over trials
    and AE (association extraction) tasks.
    """

    def __init__(self, dataset: Dataset):
        """
        Initialize the Evaluator with a dataset and prepare neighbor samplers for each dataset split.
        
        Args:
            dataset (Dataset): A Dataset object containing the dataset splits and associated graph data.
        """
        self.dataset = dataset
        self.graph_samplers = {}
        # Create a NeighborSampler for each dataset split
        for key, d in dataset.datasets.items():
            self.graph_samplers[key] = NeighborSampler(
                d.data.edge_index,
                node_idx=None,
                sizes=[-1, -1],
                batch_size=1024,
                shuffle=True,
                num_workers=0
            )

    @staticmethod
    def _eval_metric(scores, labels, probs, rng, num_repeats):
        """
        Compute evaluation metrics for given scores, labels, and predicted probabilities.
        
        This method calculates metrics such as ROC AUC, AUPRC, and balanced AUPRC.
        Additional ranking metrics (e.g., MRR, Hits@K, AP) are provided in commented-out code.
        
        Args:
            scores (np.array): Array of prediction scores.
            labels (torch.Tensor): Tensor containing ground truth binary labels.
            probs (np.array): Array of predicted probabilities.
            rng (np.random.RandomState): Random number generator for reproducibility.
            num_repeats (int): Number of repeats for balanced AUPRC computation.
        
        Returns:
            dict: A dictionary with computed evaluation metrics.
        """

        # Function to compute Area Under the Precision-Recall Curve (AUPRC)
        def auprc(*, y_true, y_score):
            """
            Compute the AUPRC given true labels and prediction scores.
            
            Args:
                y_true (np.array): True binary labels.
                y_score (np.array): Predicted scores or probabilities.
            
            Returns:
                float: The computed AUPRC value.
            """
            precision, recall, _ = precision_recall_curve(y_true=y_true, probas_pred=y_score)
            return auc(recall, precision)

        # Function to compute balanced AUPRC by sampling negatives according to a specified ratio
        def balanced_auprc_(*, y_true, y_score, ratio=1):
            """
            Compute a balanced AUPRC by sampling a subset of negative examples.
            
            Args:
                y_true (np.array): True binary labels.
                y_score (np.array): Predicted scores or probabilities.
                ratio (int, optional): Ratio of negatives to positives to sample. Defaults to 1.
            
            Returns:
                float: The computed balanced AUPRC value.
            """
            n = y_true.size(0)
            # Create a boolean mask with True for positive labels
            mask = np.zeros_like(y_true).astype(np.bool)
            mask[y_true == 1] = True
            num_pos = int(y_true.sum().item())
            num_negs = n - num_pos
            # Sample a limited number of negative examples based on the ratio
            sampled_num_negs = min(ratio * num_pos, num_negs)
            sampled_negs = rng.choice(np.arange(n)[y_true == 0], sampled_num_negs)
            mask[sampled_negs] = True
            return auprc(y_true=y_true[mask], y_score=y_score[mask])

        # Function to average the balanced AUPRC over multiple repeats for stability
        def balanced_auprc(*, y_true, y_score, ratio=1, n=100):
            """
            Compute the average balanced AUPRC over multiple iterations.
            
            Args:
                y_true (np.array): True binary labels.
                y_score (np.array): Predicted scores or probabilities.
                ratio (int, optional): Ratio of negatives to positives to sample. Defaults to 1.
                n (int, optional): Number of repeats for sampling. Defaults to 100.
            
            Returns:
                float: The averaged balanced AUPRC value.
            """
            val = []
            for i in range(n):
                val.append(balanced_auprc_(y_true=y_true, y_score=y_score, ratio=ratio))
            return np.mean(val)

        # Return a dictionary of evaluation metrics
        return {
            # Uncomment the following lines to compute additional ranking metrics:
            # 'MRR': 1.0 / ranking,
            # 'MR': float(ranking),
            # 'HITS@1': hitk(1),
            # 'HITS@3': hitk(3),
            # 'HITS@10': hitk(10),
            # 'HITS@20': hitk(20),
            # 'HITS@50': hitk(50),
            'AVG_POS': labels.sum().item(),  # Total number of positive examples
            'avg_aes': labels.size(0),         # Total number of examples (e.g., test set size)
            # Uncomment the following lines for additional metrics:
            # "AR@5": rk(5),
            # "AR@10": rk(10),
            # "AR@20": rk(20),
            # "AR@50": rk(50),
            # "AR@100": rk(100),
            # "AP": actual_apk(),
            # "AP50": actual_apk(k=50),
            # "AP20": actual_apk(k=20),
            "roc_auc": roc_auc_score(y_true=labels, y_score=probs),  # ROC AUC metric
            # 'avg_precision_score': average_precision_score(y_true=labels, y_score=probs),
            'auprc': auprc(y_true=labels, y_score=probs),              # AUPRC metric
            'balanced_auprc': balanced_auprc(y_true=labels, y_score=probs, n=num_repeats),  # Averaged balanced AUPRC
            # Additional balanced AUPRC metrics with varying ratios can be uncommented if needed:
            # 'balanced_auprc_2': balanced_auprc(y_true=labels, y_score=probs, ratio=2),
            # 'balanced_auprc_5': balanced_auprc(y_true=labels, y_score=probs, ratio=5),
            # 'balanced_auprc_10': balanced_auprc(y_true=labels, y_score=probs, ratio=10),
            # Other classification metrics (precision, recall, f1_score) can be computed as needed:
            # 'precision': precision_score(y_true=labels, y_pred=probs > 0.5, zero_division=0),
            # 'recall': recall_score(y_true=labels, y_pred=probs > 0.5, zero_division=0),
            # 'f1_score': f1_score(y_true=labels, y_pred=probs > 0.5, zero_division=0)
        }

    @staticmethod
    def _eval_metric_freq(scores, labels, probs, rng):
        """
        Compute frequency-based evaluation metrics.
        
        This method maps continuous probability values to discrete class labels based on predefined thresholds,
        and computes metrics such as RMSE, precision, recall, F1 score, and accuracy.
        
        Args:
            scores (np.array): Array of prediction scores (unused in current implementation).
            labels (np.array): Array of true continuous label values.
            probs (np.array): Array of predicted probabilities.
            rng (np.random.RandomState): Random number generator for reproducibility.
        
        Returns:
            dict: A dictionary containing computed frequency-based metrics.
        """

        def get_class_label(p):
            """
            Map continuous probability values to discrete class labels based on thresholds.
            
            Args:
                p (np.array): Array of probability values.
            
            Returns:
                np.array: Array of integer class labels ranging from 0 to 5.
            """
            y = np.zeros_like(p)
            for i in range(len(p)):
                p_i = p[i] * 100  # Scale probability for thresholding
                if p_i < 0.001:
                    y[i] = 0  # Very low probability
                if p_i < 0.01:
                    y[i] = 1  # Low probability
                elif p_i < 0.1:
                    y[i] = 2  # Slightly higher probability
                elif p_i < 1:
                    y[i] = 3  # Moderate probability
                elif p_i < 10:
                    y[i] = 4  # High probability
                else:
                    y[i] = 5  # Very high probability
            return y

        # Map raw labels and probabilities to discrete class labels
        y_true = get_class_label(labels)
        y_pred = get_class_label(probs)
        # Return a dictionary of frequency-based evaluation metrics computed using sklearn functions
        return {
            'rmse': mean_squared_error(y_true=labels, y_pred=probs),  # Root Mean Squared Error
            'AVG_POS': labels.sum().item(),  # Total positive count
            'avg_aes': labels.shape[0],      # Total count of examples
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
        """
        Evaluate ranking performance metrics for AE tasks.
        
        This method iterates over each example, computes evaluation metrics using _eval_metric,
        and aggregates the results. It also computes weighted averages of the AUPRC metric based on 
        the number of positive examples.
        
        Args:
            y_score (np.array): 2D array of prediction scores (each row corresponds to an example).
            labels (np.array): 2D array of ground truth labels (each row corresponds to an example).
            num_repeats (int, optional): Number of repeats for computing balanced AUPRC. Defaults to 100.
        
        Returns:
            dict: A dictionary containing aggregated ranking evaluation metrics.
        """
        # Initialize a random number generator with a fixed seed for reproducibility
        rng = np.random.RandomState(42)

        def calc(prefix, y_score, y_true):
            """
            Calculate evaluation metrics for each example and aggregate the results.
            
            Args:
                prefix (str): Prefix label for the metrics (e.g., 'ae' or 'trial').
                y_score (np.array): Array of prediction scores for each example.
                y_true (np.array): Array of true labels for each example.
            
            Returns:
                dict: A dictionary containing aggregated metrics and evaluation logs.
            """
            logs = []
            logs_full = []
            # Iterate over each example in the score array
            for i in tqdm(range(len(y_score))):  # for each AE
                nae = sum(y_true[i])
                # Skip evaluation if the example is trivial (all positive or no positive labels)
                if nae == 0 or nae == len(y_true[i]):
                    logs_full.append(None)
                    continue
                # Compute evaluation metrics for the current example
                eval_out = Evaluator._eval_metric(y_score[i], y_true[i], y_score[i], rng, num_repeats)
                logs.append(eval_out)
                logs_full.append(eval_out)
            metrics = {}
            if len(logs) > 0:
                # Record the number of evaluated examples
                metrics[f'{prefix}_n'] = len(logs)
                # Average each metric over all examples
                for metric in logs[0].keys():
                    metrics[f"{prefix}_{metric}"] = sum([log[metric] for log in logs]) / len(logs)
                # Compute weighted averages of AUPRC using the number of positive examples as weights
                weights = [log['AVG_POS'] for log in logs]
                metrics[f"{prefix}_auprc_logweighted"] = sum(
                    [log['auprc'] * math.log(w) for log, w in zip(logs, weights)]
                ) / sum([math.log(w) for w in weights])
                metrics[f"{prefix}_auprc_weighted"] = sum(
                    [log['auprc'] * w for log, w in zip(logs, weights)]
                ) / sum(weights)
                # For each metric, compute additional aggregated metrics based on filtering conditions
                for metric in ['AVG_POS', 'avg_aes', 'roc_auc', 'auprc', 'balanced_auprc']:
                    # Filter examples with a negative-to-positive ratio less than or equal to 10
                    filtered_logs = [log[metric] for log in logs if (log['avg_aes'] - log['AVG_POS']) / log['AVG_POS'] <= 10]
                    metrics[f"{prefix}_{metric}_neg_pos<=10"] = sum(filtered_logs) / len(filtered_logs) if len(filtered_logs) > 0 else 0
                    metrics[f"{prefix}_n_neg_pos<=10"] = len(filtered_logs) if len(filtered_logs) > 0 else 0
                    # Filter examples with at least 10 positive labels
                    filtered_logs = [log for log in logs if log['AVG_POS'] >= 10]
                    if len(filtered_logs) > 0:
                        metrics[f"{prefix}_{metric}_pos>=10"] = sum([log[metric] for log in filtered_logs]) / len(filtered_logs)
                        metrics[f"{prefix}_n_pos>=10"] = len(filtered_logs)
                        weights = [log['AVG_POS'] for log in filtered_logs]
                        metrics[f"{prefix}_{metric}_pos>=10_logweighted"] = sum(
                            [log[metric] * math.log(w) for log, w in zip(filtered_logs, weights)]
                        ) / sum([math.log(w) for w in weights])
                        metrics[f"{prefix}_{metric}_pos>=10_weighted"] = sum(
                            [log[metric] * w for log, w in zip(filtered_logs, weights)]
                        ) / sum(weights)
                    else:
                        metrics[f"{prefix}_{metric}_pos>=10"] = 0
                        metrics[f"{prefix}_n_pos>=10"] = 0
                        metrics[f"{prefix}_{metric}_pos>=10_logweighted"] = 0
                        metrics[f"{prefix}_{metric}_pos>=10_weighted"] = 0
                    # Filter examples with at least 15 positive labels
                    filtered_logs = [log for log in logs if log['AVG_POS'] >= 15]
                    if len(filtered_logs) > 0:
                        metrics[f"{prefix}_{metric}_pos>=15"] = sum([log[metric] for log in filtered_logs]) / len(filtered_logs)
                        metrics[f"{prefix}_n_pos>=15"] = len(filtered_logs)
                        weights = [log['AVG_POS'] for log in filtered_logs]
                        metrics[f"{prefix}_{metric}_pos>=15_logweighted"] = sum(
                            [log[metric] * math.log(w) for log, w in zip(filtered_logs, weights)]
                        ) / sum([math.log(w) for w in weights])
                        metrics[f"{prefix}_{metric}_pos>=15_weighted"] = sum(
                            [log[metric] * w for log, w in zip(filtered_logs, weights)]
                        ) / sum(weights)
                    else:
                        metrics[f"{prefix}_{metric}_pos>=15"] = 0
                        metrics[f"{prefix}_n_pos>=15"] = 0
                        metrics[f"{prefix}_{metric}_pos>=15_logweighted"] = 0
                        metrics[f"{prefix}_{metric}_pos>=15_weighted"] = 0
                    # Filter examples with at least 20 positive labels
                    filtered_logs = [log for log in logs if log['AVG_POS'] >= 20]
                    if len(filtered_logs) > 0:
                        metrics[f"{prefix}_{metric}_pos>=20"] = sum([log[metric] for log in filtered_logs]) / len(filtered_logs)
                        metrics[f"{prefix}_n_pos>=20"] = len(filtered_logs)
                        weights = [log['AVG_POS'] for log in filtered_logs]
                        metrics[f"{prefix}_{metric}_pos>=20_logweighted"] = sum(
                            [log[metric] * math.log(w) for log, w in zip(filtered_logs, weights)]
                        ) / sum([math.log(w) for w in weights])
                        metrics[f"{prefix}_{metric}_pos>=20_weighted"] = sum(
                            [log[metric] * w for log, w in zip(filtered_logs, weights)]
                        ) / sum(weights)
                    else:
                        metrics[f"{prefix}_{metric}_pos>=20"] = 0
                        metrics[f"{prefix}_n_pos>=20"] = 0
                        metrics[f"{prefix}_{metric}_pos>=20_logweighted"] = 0
                        metrics[f"{prefix}_{metric}_pos>=20_weighted"] = 0

                # Save the full logs for reference
                metrics[f"{prefix}_logs"] = logs
            # Save the full logs including skipped examples
            metrics[f"{prefix}_logs_full"] = logs_full
            return metrics

        metrics_all = {}
        # Evaluate on the AE (association extraction) task by transposing the score and label matrices
        metrics_all.update(calc('ae', y_score.T, labels.T))
        return metrics_all

    @staticmethod
    def freq_evaluate(*, y_score, labels):
        """
        Evaluate frequency-based performance metrics for AE tasks.
        
        This method converts torch tensors to numpy arrays and computes frequency-based evaluation
        metrics using the _eval_metric_freq function.
        
        Args:
            y_score (torch.Tensor): Tensor of prediction scores.
            labels (torch.Tensor): Tensor of ground truth labels.
        
        Returns:
            dict: A dictionary containing aggregated frequency-based evaluation metrics.
        """
        # Convert tensors to numpy arrays for evaluation
        rng = np.random.RandomState(42)
        y_score = y_score.cpu().numpy()
        labels = labels.cpu().numpy()

        def calc(prefix, *, y_score, y_true):
            """
            Calculate frequency-based evaluation metrics for each example.
            
            Args:
                prefix (str): Prefix label for the metrics (e.g., 'ae' or 'trial').
                y_score (np.array): Array of prediction scores for each example.
                y_true (np.array): Array of true labels for each example.
            
            Returns:
                dict: A dictionary containing aggregated frequency-based metrics.
            """
            logs = []
            for i in range(len(y_score)):
                nae = sum(y_true[i])
                # Skip examples with no positive labels or all positives
                if nae == 0 or nae == len(y_true[i]):
                    continue
                logs.append(Evaluator._eval_metric_freq(y_score[i], y_true[i], y_score[i], rng))
            metrics = {}
            if len(logs) > 0:
                metrics[f'{prefix}_n'] = len(logs)
                # Average each metric over all evaluated examples
                for metric in logs[0].keys():
                    metrics[f"{prefix}_{metric}"] = sum([log[metric] for log in logs]) / len(logs)
                metrics[f"{prefix}_logs"] = logs
            return metrics

        metrics_all = {}
        # Evaluate frequency metrics on the AE task by transposing the score and label matrices
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
        """
        Evaluate the model on trial and AE pairs for a given data split.
        
        This method performs the following steps:
          1. Retrieves positive and negative trial-AE pairs.
          2. Computes prediction scores and probabilities in batches.
          3. Aggregates evaluation metrics separately for trials and AEs.
          4. Returns both detailed logs and aggregated metrics.
        
        Args:
            trainer: Trainer object that holds the model, data, and utility functions.
            batch_size (int): Batch size for evaluation.
            split (str, optional): Data split to evaluate on (default is 'valid').
        
        Returns:
            tuple: A tuple containing:
                - (logs_t, logs_ae): Lists of evaluation logs for trials and AEs.
                - metrics (dict): A dictionary of aggregated evaluation metrics.
        """
        # Retrieve the model configured for evaluation
        model = trainer.put_model()
        # Get positive and negative trial-AE pairs for the given split
        pos_pairs, neg_pairs = self.dataset.trial_ae_triples(split)
        # Concatenate head and tail pairs for evaluation
        h = np.concatenate([pos_pairs[0], neg_pairs[0]])
        t = np.concatenate([pos_pairs[1], neg_pairs[1]])
        pbar = trainer.pbar
        # Create binary labels: 1 for positive pairs, 0 for negative pairs
        labels = np.concatenate([np.ones_like(pos_pairs[1]), np.zeros_like(neg_pairs[1])])
        # Get the appropriate graph sampler and data for the split
        graph_sampler = self.graph_samplers[split]
        data = trainer.data[split]
        scores = []
        probs = []
        model.eval()
        pbar.reset(total=len(h))
        # Process evaluation in batches
        for i in range(0, len(h), batch_size):
            h_i, t_i = h[i:i + batch_size], t[i:i + batch_size]
            # Get the relation id for the current batch
            r_i = np.ones_like(h_i) * self.dataset.dataset.relation2id[self.dataset.rel]
            # Get unique nodes and their indices for the batch
            nodes, nodes_idx = np.unique(np.concatenate([h_i, t_i]), return_inverse=True)
            # Sample the neighborhood for the unique nodes
            _, n_id, adjs = graph_sampler.sample(nodes)
            # Compute node embeddings using the model
            embs = model.encode(data.x[n_id], adjs, data.edge_type, trainer.devices)
            # Get the prediction score for the batch
            _, score = model(embs, [(nodes_idx[:len(h_i)], r_i, nodes_idx[len(h_i):]), None, None],
                             trainer.devices, mode='eval')
            score = score.squeeze()
            scores.append(score.detach().cpu().numpy())
            # Apply sigmoid to get probabilities
            probs.append(torch.sigmoid(score).detach().cpu().numpy())
            pbar.update(batch_size)
        # Concatenate scores and probabilities from all batches
        scores = np.concatenate(scores)
        probs = np.concatenate(probs)

        logs_t = []
        trials = self.dataset.trials_by_split[split]
        pbar.reset(total=len(trials))
        cnt_t = 0
        # Evaluate metrics for each trial
        for trial in trials:
            mask = h == trial
            # Skip trials with no positive examples
            if labels[mask].sum() == 0:
                cnt_t += 1
                continue
            logs_t.append(self._eval_metric(scores[mask], labels[mask], probs[mask]))
            pbar.update()

        logs_ae = []
        aes = self.dataset.all_aes
        pbar.reset(total=len(aes))
        cnt_ae = 0
        # Evaluate metrics for each AE (association extraction)
        for ae in aes:
            mask = t == ae
            # Skip AEs with no positive examples
            if labels[mask].sum() == 0:
                cnt_ae += 1
                continue
            logs_ae.append(self._eval_metric(scores[mask], labels[mask], probs[mask]))
            pbar.update()

        metrics = {}
        if len(logs_t) > 0:
            # Aggregate metrics separately for trials and AEs
            for prefix, p_logs in [('trial', logs_t), ('ae', logs_ae)]:
                metrics[f'{prefix}_n'] = len(p_logs)
                for metric in p_logs[0].keys():
                    metrics[f"{prefix}_{metric}"] = sum([log[metric] for log in p_logs]) / len(p_logs)

        # Record label distribution for reference
        metrics['labels_count'] = np.unique(labels, return_counts=True)
        # Print count of skipped trials and AEs
        print(split, cnt_t, cnt_ae)
        return (logs_t, logs_ae), metrics
