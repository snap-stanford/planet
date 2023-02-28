import logging
import os
import pickle
import sys
from typing import List

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

sys.path.insert(0, "../")
from gcn_models.link_pred_models import ClassificationModel, LinkPredModel
from gcn_models.evaluator import Evaluator as Evaluator
from torch_geometric.data import NeighborSampler
import matplotlib.pyplot as plt

import wandb

class ClassifierTrainer:
    def __init__(self, *, model,
                 encoder: LinkPredModel,
                 bert_encoder,
                 combine_bert,
                 evaluate_fn,
                 fixed_encoder: bool,
                 lr, encoder_lr,
                 warm_up_steps, devices, val_devices,
                 datasets, batch_size, gradient_accumulation_steps, graph, layer_sizes,
                 log_dir, do_valid, max_grad_norm, weight_decay, trial_features,
                 concat_trial_features: bool,
                 encoder_layers_finetune: List[str],
                 fp16: bool,
                 summary_prefix: str = '', args):
        self.args = args
        self.model = model
        self.summary_prefix = summary_prefix + "/" if len(summary_prefix) > 0 else ""
        self.encoder = encoder
        self.bert_encoder = bert_encoder
        self.combine_bert = combine_bert
        if combine_bert:
            assert bert_encoder is not None
            print ('ClassifierTrainer, self.combine_bert =', self.combine_bert)
        self.evaluate_fn = evaluate_fn
        self.fixed_encoder = fixed_encoder
        print ('self.fixed_encoder', self.fixed_encoder)
        print ('self.args.bert_unfreeze_epoch', self.args.bert_unfreeze_epoch)
        self.encoder_layers_finetune = encoder_layers_finetune
        self.weight_decay = weight_decay
        bert_encoder_lr = self.args.bert_encoder_lr
        self._prepare_optimizer(lr=lr, encoder_lr=encoder_lr, bert_encoder_lr=bert_encoder_lr)
        self.peak_learning_rate = lr
        self.peak_encoder_lr = encoder_lr
        self.peak_bert_encoder_lr = bert_encoder_lr
        self.current_learning_rate = lr
        self.encoder_lr = encoder_lr
        self.bert_encoder_lr = bert_encoder_lr
        self.warm_up_steps = warm_up_steps
        if self.args.warm_up_with_linear_decay >= 0:
            self.warm_up_steps = 1e16
            print ('use self.args.warm_up_with_linear_decay', self.args.warm_up_with_linear_decay, 'and ignore warm_up_steps')
        self.pbar = tqdm(total=None)
        self.graph_samplers = {}
        self.data = {}
        self.samplers = {}
        self.devices = devices
        self.val_devices = val_devices
        self.datasets = datasets
        self.batch_size = batch_size
        self.gradient_accumulation_steps = gradient_accumulation_steps
        print ('batch_size:', batch_size)
        print ('gradient_accumulation_steps (for training):', gradient_accumulation_steps)
        print ('effective batch_size (for training):', batch_size * gradient_accumulation_steps)

        self.data = graph.data
        print ('layer_sizes', layer_sizes)
        self.graph_sampler = NeighborSampler(graph.data.edge_index, node_idx=None,
                                             sizes=layer_sizes, batch_size=batch_size, #1024,
                                             shuffle=True, num_workers=0)
        for key, d in datasets.items():
            self.samplers[key] = DataLoader(d, batch_size=batch_size if key=='train' else int(batch_size*2), shuffle=(key=='train'), drop_last=(key=='train'))
        self.log_dir = log_dir + (f"/{summary_prefix}" if len(summary_prefix) > 0 else "")
        self.writer = SummaryWriter(self.log_dir)
        self.do_valid = do_valid
        self.max_grad_norm = max_grad_norm
        self.trial_features = trial_features
        self.concat_trial_features = concat_trial_features
        # if self.trial_features:
        #     assert self.fixed_encoder, "Encoder should be fixed for trial features"

        self.fp16 = fp16
        if self.fp16:
            print ('Using fp16 training')
            self.scaler = torch.cuda.amp.GradScaler() #assume torch>=1.6.0


    def put_model(self, devices):
        if self.encoder:
            encoder = self.encoder.to_devices(devices) #this uses 4
        else:
            encoder = self.encoder
        model = self.model.to_device(devices[-1])
        #Added
        if self.bert_encoder:
            bert_encoder = self.bert_encoder.to(devices[-2])
        else:
            bert_encoder = self.bert_encoder
        return encoder, bert_encoder, model

    def freeze_net(self, module):
        if module is None:
            return
        for p in module.parameters():
            p.requires_grad = False

    def unfreeze_net(self, module):
        if module is None:
            return
        for p in module.parameters():
            p.requires_grad = True

    def _prepare_optimizer(self, *, lr, encoder_lr, bert_encoder_lr):
        if self.fixed_encoder:
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=self.weight_decay)
        else:
            encoder_params = []
            if self.combine_bert != -1:
                encoder_layers = {
                    "embedding": self.encoder.embedding,
                }
                for i in range(len(self.encoder.encoder.convs)):
                    encoder_layers['conv' + str(i)] = self.encoder.encoder.convs[i]
                for layer in self.encoder_layers_finetune:
                    encoder_params.append({
                        'params': encoder_layers[layer].parameters(), 'lr': encoder_lr
                    })
                if getattr(self.encoder.encoder, 'ie_layer', False):
                    print ('Optimze ie_layer params')
                    encoder_params.append({
                        'params': self.encoder.encoder.ie_layer.parameters(), 'lr': encoder_lr
                    })
            if self.combine_bert:
                print ('Optimze bert params')
                encoder_params.append({
                    'params': self.bert_encoder.parameters(), 'lr': bert_encoder_lr
                })
            logging.info(f"Optimizer params: {encoder_params}")
            to_optimze = [
                *encoder_params,
                {'params': self.model.parameters(), 'lr': lr},
            ]
            print ('to_optimze', to_optimze)
            self.optimizer = torch.optim.Adam(to_optimze, weight_decay=self.weight_decay)
            # self.optimizer = torch.optim.Adam([
            #     *encoder_params,
            #     {'params': self.model.parameters()},
            # ], lr=lr, weight_decay=self.weight_decay)


    def _encode(self, x, devices, mode='train'):
        def gcn_encode(x_gcn):
            _, n_id, adjs = self.graph_sampler.sample(x_gcn) #n_id includes all k-hop neighbors from the nodes in original batch. adjs is a list of length k
            if self.fixed_encoder:
                encoder = self.encoder.eval()
                with torch.no_grad():
                    return encoder.encode(self.data.x[n_id], adjs, self.data.edge_type, self.devices)
            else:
                if mode == 'train':
                    encoder = self.encoder.train()
                else:
                    encoder = self.encoder.eval()
                return encoder.encode(self.data.x[n_id], adjs, self.data.edge_type, self.devices)

        def gcn_encode_with_bert(x_gcn, bert_x):
            orig_node_x = x_gcn
            _, n_id, adjs = self.graph_sampler.sample(x_gcn) #n_id includes all k-hop neighbors from the nodes in original batch. adjs is a list of length k
            if self.fixed_encoder:
                encoder = self.encoder.eval()
                with torch.no_grad():
                    return encoder.encode((self.data.x[n_id], orig_node_x, bert_x), adjs, self.data.edge_type, self.devices)
            else:
                if mode == 'train':
                    encoder = self.encoder.train()
                else:
                    encoder = self.encoder.eval()
                return encoder.encode((self.data.x[n_id], orig_node_x, bert_x), adjs, self.data.edge_type, self.devices)

        def bert_encode(x_input_ids, x_attention_mask, x_token_type_ids):
            if mode == 'train':
                bert_encoder = self.bert_encoder.train()
            else:
                bert_encoder = self.bert_encoder.eval()
            outputs = bert_encoder(input_ids=x_input_ids, attention_mask=x_attention_mask, token_type_ids=x_token_type_ids) #last_hidden_state[bs, seqlen, dim], pooler_output[bs, dim], ...
            if self.args.bert_ignore_pooler:
                outputs.pooler_output = outputs[0][:, 0]
            return outputs

        if self.combine_bert == -1: #Added
            x_input_ids, x_attention_mask, x_token_type_ids, x_gcn = x
            if self.args.default_task_name == 'binary_pair_efficacy':
                assert x_input_ids.size(-1) == 2 and x_gcn.size(-1) == 2
                x_gcn = torch.cat(x_gcn.unbind(dim=-1), dim=0)
                x_input_ids = torch.cat(x_input_ids.unbind(dim=-1), dim=0)
                x_attention_mask = torch.cat(x_attention_mask.unbind(dim=-1), dim=0)
                x_token_type_ids = torch.cat(x_token_type_ids.unbind(dim=-1), dim=0)
            x_input_ids = x_input_ids.to(devices[-2])
            x_attention_mask = x_attention_mask.to(devices[-2])
            x_token_type_ids = x_token_type_ids.to(devices[-2])
            if mode == 'train' and self.fp16:
                with torch.cuda.amp.autocast():
                    out = bert_encode(x_input_ids, x_attention_mask, x_token_type_ids)[1].to(devices[-1])
            else:
                out = bert_encode(x_input_ids, x_attention_mask, x_token_type_ids)[1].to(devices[-1])
            # print ('out.dtype', out.dtype)
            return out
        elif self.combine_bert in [1, 6]: #Added
            x_input_ids, x_attention_mask, x_token_type_ids, x_gcn = x
            if self.args.default_task_name == 'binary_pair_efficacy':
                assert x_input_ids.size(-1) == 2 and x_gcn.size(-1) == 2
                x_gcn = torch.cat(x_gcn.unbind(dim=-1), dim=0)
                x_input_ids = torch.cat(x_input_ids.unbind(dim=-1), dim=0)
                x_attention_mask = torch.cat(x_attention_mask.unbind(dim=-1), dim=0)
                x_token_type_ids = torch.cat(x_token_type_ids.unbind(dim=-1), dim=0)
            x_input_ids = x_input_ids.to(devices[-2])
            x_attention_mask = x_attention_mask.to(devices[-2])
            x_token_type_ids = x_token_type_ids.to(devices[-2])
            bert_out = bert_encode(x_input_ids, x_attention_mask, x_token_type_ids)[1].to(devices[-1])
            gcn_out = gcn_encode(x_gcn).to(devices[-1])
            return torch.cat((bert_out, gcn_out), dim=-1)
        elif self.combine_bert == 2: #Added
            x_input_ids, x_attention_mask, x_token_type_ids, x_gcn = x
            if self.args.default_task_name == 'binary_pair_efficacy':
                assert x_input_ids.size(-1) == 2 and x_gcn.size(-1) == 2
                x_gcn = torch.cat(x_gcn.unbind(dim=-1), dim=0)
                x_input_ids = torch.cat(x_input_ids.unbind(dim=-1), dim=0)
                x_attention_mask = torch.cat(x_attention_mask.unbind(dim=-1), dim=0)
                x_token_type_ids = torch.cat(x_token_type_ids.unbind(dim=-1), dim=0)
            x_input_ids = x_input_ids.to(devices[-2])
            x_attention_mask = x_attention_mask.to(devices[-2])
            x_token_type_ids = x_token_type_ids.to(devices[-2])
            bert_out = bert_encode(x_input_ids, x_attention_mask, x_token_type_ids)[0] #[batch_size, seqlen, dim]
            gcn_out = gcn_encode(x_gcn) #[batch_size, dim*6]
            gcn_out = gcn_out.view(gcn_out.size(0), 6, -1) #[batch_size, 6, dim]
            assert gcn_out.size(1) == 6
            return [bert_out.to(devices[-1]), x_attention_mask.to(devices[-1]), gcn_out.to(devices[-1])]
        elif self.combine_bert == 3: #Added
            x_input_ids, x_attention_mask, x_token_type_ids, x_gcn = x
            if self.args.default_task_name == 'binary_pair_efficacy':
                assert x_input_ids.size(-1) == 2 and x_gcn.size(-1) == 2
                x_gcn = torch.cat(x_gcn.unbind(dim=-1), dim=0)
                x_input_ids = torch.cat(x_input_ids.unbind(dim=-1), dim=0)
                x_attention_mask = torch.cat(x_attention_mask.unbind(dim=-1), dim=0)
                x_token_type_ids = torch.cat(x_token_type_ids.unbind(dim=-1), dim=0)
            x_input_ids = x_input_ids.to(devices[-2])
            x_attention_mask = x_attention_mask.to(devices[-2])
            x_token_type_ids = x_token_type_ids.to(devices[-2])
            bert_out = bert_encode(x_input_ids, x_attention_mask, x_token_type_ids)[1] #[batch_size, dim]
            gcn_out = gcn_encode(x_gcn) #[batch_size, 6*dim]
            gcn_out = gcn_out.view(gcn_out.size(0), 6, -1) #[batch_size, 6, dim]
            assert gcn_out.size(1) == 6
            return [bert_out.to(devices[-1]), gcn_out.to(devices[-1])]
        elif self.combine_bert == 4: #Added
            x_input_ids, x_attention_mask, x_token_type_ids, x_gcn = x
            if self.args.default_task_name == 'binary_pair_efficacy':
                assert x_input_ids.size(-1) == 2 and x_gcn.size(-1) == 2
                x_gcn = torch.cat(x_gcn.unbind(dim=-1), dim=0)
                x_input_ids = torch.cat(x_input_ids.unbind(dim=-1), dim=0)
                x_attention_mask = torch.cat(x_attention_mask.unbind(dim=-1), dim=0)
                x_token_type_ids = torch.cat(x_token_type_ids.unbind(dim=-1), dim=0)
            x_input_ids = x_input_ids.to(devices[-2])
            x_attention_mask = x_attention_mask.to(devices[-2])
            x_token_type_ids = x_token_type_ids.to(devices[-2])
            bert_out = bert_encode(x_input_ids, x_attention_mask, x_token_type_ids)[1] #[batch_size, dim]
            gcn_out, bert_out2 = gcn_encode_with_bert(x_gcn, bert_out) #gcn_out: [batch_size, 6*dim], bert_out: [batch_size, bert_dim]
            return torch.cat((bert_out2.to(devices[-1]), gcn_out.to(devices[-1])), dim=-1)
        elif self.combine_bert == 5: #Added
            x_input_ids, x_attention_mask, x_token_type_ids, x_gcn = x
            if self.args.default_task_name == 'binary_pair_efficacy':
                assert x_input_ids.size(-1) == 2 and x_gcn.size(-1) == 2
                x_gcn = torch.cat(x_gcn.unbind(dim=-1), dim=0)
                x_input_ids = torch.cat(x_input_ids.unbind(dim=-1), dim=0)
                x_attention_mask = torch.cat(x_attention_mask.unbind(dim=-1), dim=0)
                x_token_type_ids = torch.cat(x_token_type_ids.unbind(dim=-1), dim=0)
            x_input_ids = x_input_ids.to(devices[-2])
            x_attention_mask = x_attention_mask.to(devices[-2])
            x_token_type_ids = x_token_type_ids.to(devices[-2])
            bert_out = bert_encode(x_input_ids, x_attention_mask, x_token_type_ids)[1] #[batch_size, dim]
            gcn_out, bert_out2 = gcn_encode_with_bert(x_gcn, bert_out) #gcn_out: [batch_size, 6*dim], bert_out: [batch_size, bert_dim]
            return torch.cat((bert_out.to(devices[-1]), bert_out2.to(devices[-1]), gcn_out.to(devices[-1])), dim=-1)
        elif self.concat_trial_features:
            x_trial, x_gcn = x
            if self.args.default_task_name == 'binary_pair_efficacy':
                assert x_trial.size(-1) == 2 and x_gcn.size(-1) == 2
                x_gcn = torch.cat(x_gcn.unbind(dim=-1), dim=0)
                x_trial = torch.cat(x_trial.unbind(dim=-1), dim=0)
            return torch.cat((x_trial.to(devices[0]), gcn_encode(x_gcn)), dim=-1)
        elif self.trial_features:
            if self.args.default_task_name == 'binary_pair_efficacy':
                assert x.size(-1) == 2
                x = torch.cat(x.unbind(dim=-1), dim=0)
            return x
        else:
            if self.args.default_task_name == 'binary_pair_efficacy':
                assert x.size(-1) == 2
                x = torch.cat(x.unbind(dim=-1), dim=0)
            return gcn_encode(x)

    def _batch_xy(self, batch):
        if self.combine_bert: #Added
            return batch[0:4], batch[4:]
        elif not self.concat_trial_features:
            return batch[0], batch[1:]
        else:
            return batch[0:2], batch[2:]

    def train(self):
        (encoder, bert_encoder, model), optimizer = self.put_model(self.devices), self.optimizer
        pbar = self.pbar
        clf_sampler = self.samplers['train']
        model.train()

        if hasattr(model, 'combo_weight'):
            print ('\nmodel.combo_weight', model.combo_weight)
            print ('torch.sigmoid(model.combo_weight*100)', torch.sigmoid(model.combo_weight*100))

        loss_total = 0
        grad_norm_total = 0
        pbar.reset(total=len(clf_sampler)//self.gradient_accumulation_steps)
        optimizer.zero_grad()
        for i, batch in enumerate(clf_sampler):
            # if i == 2: break
            x, y = self._batch_xy(batch)
            if self.fp16:
                embs = self._encode(x, self.devices)
                with torch.cuda.amp.autocast():
                    scores = model(embs, self.devices)
                    loss = model.loss(scores, y)
            else:
                embs = self._encode(x, self.devices)
                scores = model(embs, self.devices)
                loss = model.loss(scores, y)

            if self.gradient_accumulation_steps > 1:
                loss = loss / self.gradient_accumulation_steps

            if self.fp16:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
            loss_total += loss.item()

            if (i + 1) % self.gradient_accumulation_steps == 0:
                # all_model_params  = [p for p in model.parameters()]
                # all_model_params += [p for p in encoder.parameters()] if encoder is not None else []
                # all_model_params += [p for p in bert_encoder.parameters()] if bert_encoder is not None else []
                if self.fp16:
                    self.scaler.unscale_(optimizer)
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), self.max_grad_norm)
                    if grad_norm.isnan():
                        print ('clf_model grad norm NaN!')
                    elif grad_norm.isinf():
                        print ('clf_model grad norm inf!')
                    if bert_encoder is not None:
                        _grad_norm = torch.nn.utils.clip_grad_norm_(bert_encoder.parameters(), self.max_grad_norm)
                        if _grad_norm.isnan():
                            print ('bert_encoder grad norm NaN!')
                        elif _grad_norm.isinf():
                            print ('bert_encoder grad norm inf!')
                        grad_norm += _grad_norm
                else:
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), self.max_grad_norm)
                    if bert_encoder is not None:
                        grad_norm += torch.nn.utils.clip_grad_norm_(bert_encoder.parameters(), self.max_grad_norm)

                if self.fp16:
                    self.scaler.step(optimizer)
                    self.scaler.update()
                else:
                    optimizer.step()

                optimizer.zero_grad()
                # model.zero_grad()

                assert grad_norm.dim() == 0 #should be scalar
                grad_norm_total += grad_norm.mean().item()
                pbar.update()
                pbar.set_postfix({
                    'loss': loss.item() * self.gradient_accumulation_steps,
                    "grad_norm": grad_norm.mean().item(),
                    'loss_running': loss_total * self.gradient_accumulation_steps / (i + 1)
                }, refresh=False)

        optimizer.zero_grad()
        return loss_total * self.gradient_accumulation_steps / len(clf_sampler),  grad_norm_total / (len(clf_sampler)// self.gradient_accumulation_steps)

    @torch.no_grad()
    def test(self, mode='valid'):
        encoder, bert_encoder, model = self.put_model(self.val_devices)
        pbar = self.pbar
        pbar.set_postfix({}, refresh=False)
        model.eval()
        clf_sampler = self.samplers[mode]
        pbar.reset(total=len(clf_sampler))
        y_pred = []
        y_true = []
        # xs = []
        for i, batch in enumerate(clf_sampler):
            x, y = self._batch_xy(batch)
            embs = self._encode(x, self.val_devices, mode='eval')
            scores = model(embs, self.val_devices)
            y_pred.append(scores[0].cpu())
            y_true.append(y[0].cpu())
            # xs.append(x.cpu())
            # print(scores[0].squeeze(1).size())
            pbar.update()
        y_true = torch.cat(y_true, dim=0)
        # xs = torch.cat(xs, dim=0)
        y_scores = torch.sigmoid(torch.cat(y_pred, dim=0))
        rank_metrics = self.evaluate_fn(y_score=y_scores, labels=y_true, num_repeats=10)
        return rank_metrics, y_true, y_scores #, xs

    def log_metrics(self, mode, step, metrics):
        """
        Print the evaluation logs
        """
        for metric in metrics:
            if self.summary_prefix == '' and metric == 'ae_logs' and 'auprc' in metrics[metric][0]:
                x = [(log['avg_aes'] - log['AVG_POS']) / log['AVG_POS'] for log in metrics[metric]]
                y = [log['auprc'] for log in metrics[metric]]
                plt.scatter(x, y)
                plt.xscale('log')

                self.writer.add_figure(f'{self.summary_prefix}{mode}_auprc_vs_numpos', plt.gcf(), step)
            if type(metrics[metric]) in [list, tuple]:
                continue
            logging.info('%s %s at step %d: %f' % (mode, metric, step, metrics[metric]))
            self.writer.add_scalar(self.summary_prefix + "/".join([mode, metric]), metrics[metric], step) #Changed to /
            wandb.log({"/".join([mode, metric]): metrics[metric]}, step=step)

    def valid(self, epoch):
        results = {}
        for split in ['valid', 'test']: #['train', 'valid', 'test']:
            results[split], y_true, y_score = self.test(mode=split)
            self.log_metrics(split, epoch, results[split])
            torch.save({'y_true': y_true, 'y_score': y_score}, os.path.join(self.log_dir, f'preds-{epoch}-{split}.pt'))
        return results

    def train_loop(self, *, num_epochs, old_epochs, val_every=20):
        print ('Bert param freezed at the beginning of training')
        self.freeze_net(self.bert_encoder)

        for epoch in tqdm(range(old_epochs + 1, num_epochs + 1)):
            if epoch == self.args.bert_unfreeze_epoch:
                print ('Bert param unfreezed at the beginning of epoch', epoch)
                self.unfreeze_net(self.bert_encoder)

            if self.args.warm_up_with_linear_decay >= 0:
                peak_epoch = int(self.args.warm_up_with_linear_decay * num_epochs)
                if epoch <= peak_epoch:
                    lr_scale = max(0,min(1,epoch/peak_epoch))
                else:
                    lr_scale = max(0,min(1,(num_epochs - epoch)/(num_epochs - peak_epoch)))
                self.current_learning_rate = self.peak_learning_rate * lr_scale
                self.encoder_lr = self.peak_encoder_lr * lr_scale
                self.bert_encoder_lr = self.peak_bert_encoder_lr * lr_scale
                self._prepare_optimizer(lr=self.current_learning_rate, encoder_lr=self.encoder_lr, bert_encoder_lr=self.bert_encoder_lr)


            self.pbar.set_description(f'Epoch: {epoch}', refresh=False)
            self.log_metrics('train', epoch, {'encoder_lr': self.encoder_lr, 'bert_encoder_lr': self.bert_encoder_lr, 'lr': self.current_learning_rate}) #log before self.train() so that previous epoch is uploaded to wandb ASAP
            train_loss, grad_norm = self.train()
            self.log_metrics('train', epoch, {'train_loss': train_loss, 'grad_norm': grad_norm})

            if epoch >= self.warm_up_steps:
                self.current_learning_rate /= 10
                self.encoder_lr /= 10
                self.bert_encoder_lr /= 10
                logging.info('Change learning_rate to %f at step %d' % (self.current_learning_rate, epoch))
                self._prepare_optimizer(lr=self.current_learning_rate, encoder_lr=self.encoder_lr, bert_encoder_lr=self.bert_encoder_lr)
                self.warm_up_steps = self.warm_up_steps * 2

            if epoch == 1 or epoch % val_every == 0:
                # torch.save(
                #     {'model': self.model, 'optimizer': self.optimizer, 'steps': epoch,
                #      'warm_up_steps': self.warm_up_steps, 'lr': self.current_learning_rate,
                #      'encoder': self.encoder if not self.fixed_encoder else None},
                #     os.path.join(self.log_dir, f'ckpt-{epoch}.pt'))
                if epoch % (val_every*5 ) == 0 and self.args.save_model:
                    encoder_save = self.encoder.state_dict() if (self.encoder is not None) and (not self.fixed_encoder) else None
                    bert_encoder_save = self.bert_encoder.state_dict() if (self.bert_encoder is not None) and (not self.fixed_encoder) else None
                    torch.save(
                        {'model': self.model.state_dict(), 'optimizer': self.optimizer.state_dict(), 'steps': epoch,
                         'warm_up_steps': self.warm_up_steps, 'lr': self.current_learning_rate, 'encoder_lr': self.encoder_lr, 'bert_encoder_lr': self.bert_encoder_lr,
                         'encoder': encoder_save, 'bert_encoder': bert_encoder_save},
                        os.path.join(self.log_dir, f'ckpt-{epoch}.pt'))
                if self.do_valid:
                    results = self.valid(epoch)
                    with open(os.path.join(self.log_dir, f'results-{epoch}.pkl'), 'wb') as f:
                        pickle.dump(results, f)
                # print(torch.exp(-self.model.total_loss.log_vars))
