import logging
import os
import sys
from collections import defaultdict
from typing import Dict, List

import torch
from ogb.linkproppred import Evaluator
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from gcn_models.data_loader import SplitDataset

sys.path.insert(0, "../")
from gcn_models.link_pred_models import LinkPredModel
from gcn_models.sampler import MultiProcessSampler
import wandb


class LinkPredTrainer:
    """
    Trainer class for link prediction models.
    
    This class encapsulates the training, evaluation, and logging of a link prediction model,
    utilizing negative sampling and multi-processing for efficient training.
    """

    def __init__(self, *, model: LinkPredModel, datasets: Dict[str, SplitDataset],
                 devices, val_devices, lr: float, warm_up_steps: int, num_negs: int, batch_size: Dict[str, int],
                 layer_sizes: List[int], log_dir: str, do_valid: bool, max_grad_norm: float, weight_decay: float,
                 warm_up_factor: float,
                 masked_full_graph_sampler: bool, mask_once_every: int,
                 num_workers=1):
        """
        Initialize the LinkPredTrainer.

        Args:
            model (LinkPredModel): The link prediction model to be trained.
            datasets (Dict[str, SplitDataset]): Dictionary containing training, validation, and test datasets.
            devices: Devices to run the training on.
            val_devices: Devices to run validation/testing on.
            lr (float): Initial learning rate.
            warm_up_steps (int): Number of steps for learning rate warm-up.
            num_negs (int): Number of negative samples per positive sample.
            batch_size (Dict[str, int]): Batch sizes for different dataset splits.
            layer_sizes (List[int]): List defining sizes for each layer in the model.
            log_dir (str): Directory for logging checkpoints and TensorBoard logs.
            do_valid (bool): Whether to perform validation during training.
            max_grad_norm (float): Maximum gradient norm for clipping.
            weight_decay (float): Weight decay factor for the optimizer.
            warm_up_factor (float): Factor to reduce the learning rate after warm-up.
            masked_full_graph_sampler (bool): Whether to use a masked sampler for the full graph.
            mask_once_every (int): Frequency (in steps) to apply the mask.
            num_workers (int, optional): Number of worker processes for data sampling. Defaults to 1.
        """
        self.model = model
        # Initialize the optimizer with model parameters, learning rate, and weight decay.
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        self.current_learning_rate = lr
        self.weight_decay = weight_decay
        self.warm_up_steps = warm_up_steps
        self.warm_up_factor = warm_up_factor
        # Initialize a tqdm progress bar for training iterations.
        self.pbar = tqdm(total=None)
        self.samplers = {}  # Dictionary to hold data samplers for each dataset split.
        self.data = {}      # Dictionary to hold raw data for each dataset split.
        self.devices = devices
        self.val_devices = val_devices
        self.datasets = datasets
        # Initialize the evaluator from the ogb package for validation.
        self.val_evaluator = Evaluator("ogbl-biokg")
        self.batch_size = batch_size
        # Create a sampler for each dataset split (e.g., train, valid, test).
        for key, d in datasets.items():
            self.samplers[key] = MultiProcessSampler(
                dataset=d,
                neg_sample_size=num_negs,
                batch_size=batch_size[key],
                masked_sampler=masked_full_graph_sampler if key == 'train' else False,
                mask_ratio=mask_once_every,
                mask=None,  # Optionally, a mask from d.data.pop_edges_mask could be used.
                layer_sizes=layer_sizes if key == 'train' else [-1] * len(layer_sizes),
                num_procs=num_workers,
                num_epochs=1 if key != 'train' else -1  # Unlimited epochs for training, one epoch for others.
            )
            self.data[key] = d.data  # Store the raw data for each split.
        self.log_dir = log_dir
        # Set up TensorBoard summary writer.
        self.writer = SummaryWriter(self.log_dir)
        self.do_valid = do_valid
        self.max_grad_norm = max_grad_norm

    def optimizer_to_devices(self, optimizer, model):
        """
        Move optimizer state tensors to the same device as their corresponding model parameters.

        This ensures that all optimizer states are on the correct device for computation.

        Args:
            optimizer: The optimizer whose states need to be moved.
            model: The model providing the target device for each parameter.

        Returns:
            The optimizer with its state tensors moved to the appropriate devices.
        """
        for group in optimizer.param_groups:
            for p in group['params']:
                if p.requires_grad:
                    state = optimizer.state[p]
                    if len(state) == 0:
                        # If the state is empty, no need to move tensors.
                        # (Commented out code shows how state initialization might be done.)
                        continue
                    for elm in state.values():
                        if isinstance(elm, torch.Tensor):
                            # Move the tensor data to the device of the parameter.
                            elm.data = elm.data.to(p.device)
                            # If the tensor has a gradient, move it as well.
                            if elm._grad is not None:
                                elm._grad.data = elm._grad.data.to(p.device)
        return optimizer

    def train(self, steps: int):
        """
        Run the training loop for a given number of steps.

        Args:
            steps (int): Number of training steps to perform.

        Returns:
            A tuple (average_loss, average_grad_norm) over the training steps.
        """
        # Move the model to the specified devices and update optimizer state tensors.
        model, optimizer = self.model.to_devices(self.devices), self.optimizer
        optimizer = self.optimizer_to_devices(optimizer, model)
        pbar = self.pbar
        train_sampler = self.samplers['train']
        data = self.data['train']
        model.train()  # Set the model to training mode.

        loss_total = 0
        pos_loss_total = 0
        neg_loss_total = 0
        grad_norm_total = 0
        # Reset the progress bar with the total number of steps.
        pbar.reset(total=steps)
        for i in range(steps):
            optimizer.zero_grad()  # Clear gradients from the previous step.
            # Get the next batch of training data.
            triples, nodes, (_, n_id, adjs) = next(train_sampler)
            # Encode node features using the model.
            embs = model.encode(data.x[n_id], adjs, data.edge_type, self.devices)
            # Compute prediction scores for the batch.
            scores = model(embs, triples, self.devices)
            # Compute the loss (total, positive, and negative components).
            loss, pos_loss, neg_loss = model.loss(embs, scores, self.devices)

            loss.backward()  # Backpropagate the loss.
            # Clip the gradient norms to avoid exploding gradients.
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), self.max_grad_norm)
            optimizer.step()  # Update the model parameters.

            # Accumulate loss and gradient norm metrics.
            grad_norm_total += grad_norm.mean().item()
            loss_total += loss.item()
            pos_loss_total += pos_loss.item()
            neg_loss_total += neg_loss.item()
            pbar.update()  # Update the progress bar.
            # Set postfix on the progress bar to display current training metrics.
            pbar.set_postfix({
                'loss': loss.item(),
                'pos_loss': pos_loss.item(),
                'neg_loss': neg_loss.item(),
                "grad_norm": grad_norm.mean().item(),
                'loss_running': loss_total / (i + 1),
                'pos_loss_running': pos_loss_total / (i + 1),
                'neg_loss_running': neg_loss_total / (i + 1)
            }, refresh=False)

        # Return the average loss and average gradient norm over the steps.
        return loss_total / steps, grad_norm_total / steps

    @torch.no_grad()
    def test(self, mode='valid'):
        """
        Evaluate the model on a specified dataset split without computing gradients.

        Args:
            mode (str, optional): The dataset split to evaluate ('valid' or 'test'). Defaults to 'valid'.

        Returns:
            metrics (dict): A dictionary containing evaluation metrics.
        """
        devices = self.val_devices
        # Move the model to the validation devices.
        model = self.model.to_devices(devices=devices)
        pbar = self.pbar
        pbar.set_postfix({}, refresh=False)
        model.eval()  # Set the model to evaluation mode.
        val_sampler = self.samplers[mode]
        data = self.data[mode]
        test_logs = defaultdict(list)  # Dictionary to collect logs for each metric.
        # Reset the progress bar with the number of validation batches.
        pbar.reset(total=len(val_sampler))
        for i, batch in enumerate(val_sampler):
            triples, nodes, (_, n_id, adjs) = batch
            # Encode node features for the batch.
            embs = model.encode(data.x[n_id], adjs, data.edge_type, devices)
            # Compute prediction scores.
            scores = model(embs, triples, devices)
            # Evaluate predictions using the provided evaluator.
            batch_results = self.val_evaluator.eval({
                'y_pred_pos': scores[0].squeeze(1),
                'y_pred_neg': scores[1]
            })
            # Log each metric's result.
            for metric in batch_results:
                test_logs[metric].append(batch_results[metric])
            pbar.update()
        # Renew the sampler for the next evaluation round.
        self.samplers[mode] = val_sampler.renew()
        metrics = {}
        # Aggregate and average metrics over all batches.
        for metric in test_logs:
            metrics[metric] = torch.cat(test_logs[metric]).mean().item()
        return metrics

    def log_metrics(self, mode, step, metrics):
        """
        Log and record evaluation metrics.

        Args:
            mode (str): The phase of evaluation ('train', 'valid', or 'test').
            step (int): The current training step or epoch.
            metrics (dict): Dictionary containing metric names and their corresponding values.
        """
        for metric in metrics:
            # Skip metrics that are lists or tuples.
            if type(metrics[metric]) in [list, tuple]:
                continue
            # Log metric information using the standard logging module.
            logging.info('%s %s at step %d: %f' % (mode, metric, step, metrics[metric]))
            # Record metric in TensorBoard.
            self.writer.add_scalar("_".join([mode, metric]), metrics[metric], step)
            # Log metric to Weights & Biases.
            wandb.log({"/".join([mode, metric]): metrics[metric]}, step=step)

    def train_loop(self, num_steps, old_steps, log_every=100, val_every=1000):
        """
        Run the main training loop over multiple epochs.

        Args:
            num_steps (int): Total number of training steps.
            old_steps (int): Number of steps already completed (used to determine the starting epoch).
            log_every (int): Frequency (in steps) at which training metrics are logged.
            val_every (int): Frequency (in steps) at which validation is performed and checkpoints are saved.
        """
        best_val_perf = 0  # Variable to track the best validation performance.
        # Determine epoch range based on steps already completed and total steps.
        for epoch in range(old_steps // log_every + 1, num_steps // log_every + 1):
            # Update the progress bar description with the current epoch.
            self.pbar.set_description(f'Epoch: {epoch}', refresh=False)
            # Log current learning rate.
            self.log_metrics('train', epoch * log_every, {'lr': self.current_learning_rate})
            # Perform training for the specified number of steps.
            train_loss, grad_norm = self.train(log_every)
            # Log training loss and gradient norm.
            self.log_metrics('train', epoch * log_every, {'train_loss': train_loss, 'grad_norm': grad_norm})

            # Adjust learning rate after warm-up period.
            if epoch >= self.warm_up_steps:
                self.current_learning_rate = self.current_learning_rate / self.warm_up_factor
                logging.info('Change learning_rate to %f at step %d' % (self.current_learning_rate, epoch))
                # Reinitialize the optimizer with the new learning rate.
                self.optimizer = torch.optim.Adam(
                    self.model.parameters(),
                    lr=self.current_learning_rate,
                    weight_decay=self.weight_decay
                )
                self.warm_up_steps = self.warm_up_steps * 2  # Double the warm-up steps for the next adjustment.

            # Save checkpoints and perform validation/testing periodically.
            if epoch == 1 or (epoch * log_every) % val_every == 0:
                # Save model checkpoint.
                torch.save(
                    {
                        'model': self.model.state_dict(),
                        'optimizer': self.optimizer.state_dict(),
                        'steps': epoch,
                        'warm_up_steps': self.warm_up_steps,
                        'lr': self.current_learning_rate
                    },
                    os.path.join(self.log_dir, f'ckpt-{epoch}.pt')
                )
                if self.do_valid:
                    # Log validation and test metrics.
                    self.log_metrics('valid', epoch * log_every, self.test(mode='valid'))
                    self.log_metrics('test', epoch * log_every, self.test(mode='test'))

        # Delete samplers to free resources.
        del self.samplers
