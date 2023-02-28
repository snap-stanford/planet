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
    def __init__(self, *, model: LinkPredModel, datasets: Dict[str, SplitDataset],
                 devices, val_devices, lr: float, warm_up_steps: int, num_negs: int, batch_size: Dict[str, int],
                 layer_sizes: List[int], log_dir: str, do_valid: bool, max_grad_norm: float, weight_decay: float,
                 warm_up_factor: float,
                 masked_full_graph_sampler: bool, mask_once_every: int,
                 num_workers=1):
        self.model = model
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        self.current_learning_rate = lr
        self.weight_decay = weight_decay
        self.warm_up_steps = warm_up_steps
        self.warm_up_factor = warm_up_factor
        self.pbar = tqdm(total=None)
        self.samplers = {}
        self.data = {}
        self.devices = devices
        self.val_devices = val_devices
        self.datasets = datasets
        self.val_evaluator = Evaluator("ogbl-biokg")
        self.batch_size = batch_size
        for key, d in datasets.items():
            self.samplers[key] = MultiProcessSampler(dataset=d, neg_sample_size=num_negs, batch_size=batch_size[key],
                                                     masked_sampler=masked_full_graph_sampler if key == 'train' else False,
                                                     mask_ratio=mask_once_every,
                                                     mask=None, #d.data.pop_edges_mask,
                                                     layer_sizes=layer_sizes if key == 'train' else [-1] * len(
                                                         layer_sizes),
                                                     num_procs=num_workers,
                                                     num_epochs=1 if key != 'train' else -1)
            self.data[key] = d.data
        self.log_dir = log_dir
        self.writer = SummaryWriter(self.log_dir)
        self.do_valid = do_valid
        self.max_grad_norm = max_grad_norm

    def optimizer_to_devices(self, optimizer, model):
        for group in optimizer.param_groups:
            for p in group['params']:
                if p.requires_grad:
                    state = optimizer.state[p]
                    if len(state) == 0:
                        # state['step'] = 0
                        # # Exponential moving average of gradient values
                        # state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                        # # Exponential moving average of squared gradient values
                        # state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                        # if group['amsgrad']:
                        #     # Maintains max of all exp. moving avg. of sq. grad. values
                        #     state['max_exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                        continue
                    for elm in state.values():
                        if isinstance(elm, torch.Tensor):
                            elm.data = elm.data.to(p.device)
                            if elm._grad is not None:
                                elm._grad.data = elm._grad.data.to(p.device)
        return optimizer

    def train(self, steps: int):
        model, optimizer = self.model.to_devices(self.devices), self.optimizer
        optimizer = self.optimizer_to_devices(optimizer, model)
        pbar = self.pbar
        train_sampler = self.samplers['train']
        data = self.data['train']
        model.train()

        loss_total = 0
        pos_loss_total = 0
        neg_loss_total = 0
        grad_norm_total = 0
        pbar.reset(total=steps)
        for i in range(steps):
            optimizer.zero_grad()
            triples, nodes, (_, n_id, adjs) = next(train_sampler)
            embs = model.encode(data.x[n_id], adjs, data.edge_type, self.devices)
            scores = model(embs, triples, self.devices)
            loss, pos_loss, neg_loss = model.loss(embs, scores, self.devices)

            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), self.max_grad_norm)
            optimizer.step()

            grad_norm_total += grad_norm.mean().item()
            loss_total += loss.item()
            pos_loss_total += pos_loss.item()
            neg_loss_total += neg_loss.item()
            pbar.update()
            pbar.set_postfix({
                'loss': loss.item(),
                'pos_loss': pos_loss.item(),
                'neg_loss': neg_loss.item(),
                "grad_norm": grad_norm.mean().item(),
                'loss_running': loss_total / (i + 1),
                'pos_loss_running': pos_loss_total / (i + 1),
                'neg_loss_running': neg_loss_total / (i + 1)
            }, refresh=False)

        return loss_total / steps, grad_norm_total / steps

    @torch.no_grad()
    def test(self, mode='valid'):
        devices = self.val_devices
        model = self.model.to_devices(devices=devices)
        pbar = self.pbar
        pbar.set_postfix({}, refresh=False)
        model.eval()
        val_sampler = self.samplers[mode]
        data = self.data[mode]
        test_logs = defaultdict(list)
        pbar.reset(total=len(val_sampler))
        for i, batch in enumerate(val_sampler):
            triples, nodes, (_, n_id, adjs) = batch
            embs = model.encode(data.x[n_id], adjs, data.edge_type, devices)
            scores = model(embs, triples, devices)
            # print(scores[0].squeeze(1).size())
            batch_results = self.val_evaluator.eval({'y_pred_pos': scores[0].squeeze(1),
                                                     'y_pred_neg': scores[1]})
            for metric in batch_results:
                test_logs[metric].append(batch_results[metric])
            pbar.update()
        self.samplers[mode] = val_sampler.renew()
        metrics = {}
        for metric in test_logs:
            metrics[metric] = torch.cat(test_logs[metric]).mean().item()
        return metrics

    def log_metrics(self, mode, step, metrics):
        """
        Print the evaluation logs
        """
        for metric in metrics:
            if type(metrics[metric]) in [list, tuple]:
                continue
            logging.info('%s %s at step %d: %f' % (mode, metric, step, metrics[metric]))
            self.writer.add_scalar("_".join([mode, metric]), metrics[metric], step)
            wandb.log({"/".join([mode, metric]): metrics[metric]}, step=step)

    def train_loop(self, num_steps, old_steps, log_every=100, val_every=1000):
        best_val_perf = 0
        for epoch in range(old_steps // log_every + 1, num_steps // log_every + 1):
            self.pbar.set_description(f'Epoch: {epoch}', refresh=False)
            self.log_metrics('train', epoch * log_every, {'lr': self.current_learning_rate})
            train_loss, grad_norm = self.train(log_every)
            self.log_metrics('train', epoch * log_every, {'train_loss': train_loss, 'grad_norm': grad_norm})

            if epoch >= self.warm_up_steps:
                self.current_learning_rate = self.current_learning_rate / self.warm_up_factor
                logging.info('Change learning_rate to %f at step %d' % (self.current_learning_rate, epoch))
                self.optimizer = torch.optim.Adam(
                    self.model.parameters(),
                    lr=self.current_learning_rate,
                    weight_decay=self.weight_decay
                )
                self.warm_up_steps = self.warm_up_steps * 2

            if epoch == 1 or (epoch * log_every) % val_every == 0:
                torch.save(
                    {'model': self.model.state_dict(), 'optimizer': self.optimizer.state_dict(), 'steps': epoch,
                     'warm_up_steps': self.warm_up_steps, 'lr': self.current_learning_rate},
                    os.path.join(self.log_dir, f'ckpt-{epoch}.pt'))
                if self.do_valid:
                    self.log_metrics('valid', epoch * log_every, self.test(mode='valid'))
                    self.log_metrics('test', epoch * log_every, self.test(mode='test'))

        del self.samplers
