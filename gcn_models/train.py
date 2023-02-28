import argparse
import json
import logging
import os
import sys

import torch
import wandb
import datetime

sys.path.insert(0, "../")

from gcn_models.data_loader import Dataset
from gcn_models.link_pred_models import LinkPredModel
from gcn_models.trainers import LinkPredTrainer
from gcn_models.utils import set_logger, get_activation, get_norm_layer, get_conv_layer, set_seed

import socket, os, subprocess, datetime
print(socket.gethostname())
print ("pid:", os.getpid())
print ("screen: %s" % subprocess.check_output('echo $STY', shell=True).decode('utf'))


def boolean_string(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'


def main(args):
    # Write logs to checkpoint and console
    dt = datetime.datetime.today().strftime('%Y%m%d_%H%M')
    wandb_id = wandb.util.generate_id()
    wandb_name = f"{dt}--{wandb_id}"
    wandb.init(config=args, project=args.wandb_project, name=wandb_name, id=wandb_id, dir='../')
    assert wandb.run.id == wandb_id
    args.log_dir = os.path.join(args.log_dir, wandb_name)
    os.makedirs(args.log_dir)
    print(args.log_dir)

    set_logger(args)
    set_seed(args.random_seed)

    data_path = args.data_path

    dataset = args.dataset
    data_files = ['train.tsv', 'valid.tsv', 'test.tsv']

    dataset = Dataset(data_path=data_path, dataset=dataset, data_files=data_files,
                      bidirectional=args.use_bidirectional, use_population=args.use_population,
                      remove_function=args.remove_function, remove_outcomes=args.remove_outcomes,
                      node_feats_df_path=args.node_feats_path, edge_disjoint=args.edge_disjoint,
                      dont_include_trial_node_features=args.dont_include_trial_node_features,
                      add_extra_dis_gene_edges=args.add_extra_dis_gene_edges)
    torch.save(dataset, os.path.join(args.log_dir, 'dataset.pt'))

    train_data = dataset.datasets['train'].data

    model = LinkPredModel(num_relations=train_data.num_relations, in_channels=train_data.num_nodes,
                          hidden_dim=args.embedding_size, emb_df=dataset.node_feats,
                          num_enc_layers=len(args.layer_sizes),
                          nbr_concat=False,
                          nbr_concat_weight=False,
                          num_bases=args.rgcn_bases, decoder_type=args.decoder_type,
                          activation_fn=get_activation(args),
                          norm_fn=get_norm_layer(args),
                          gamma=args.gamma, dropout=args.dropout_prob, edge_dropout=args.edge_dropout,
                          reg_param=args.regularization_param, adversarial_temperature=args.adversarial_temperature,
                          negative_adversarial_sampling=args.negative_adversarial_sampling,
                          add_residual=args.add_residual,
                          conv_fn=get_conv_layer(args),
                          conv_aggr=args.conv_aggr,
                          num_pre_layers=args.num_pre_layers,
                          num_post_layers=args.num_post_layers)

    print(model)
    logging.info('Model Parameter Configuration:')
    for name, param in model.named_parameters():
        logging.info('Parameter %s: %s, require_grad = %s' % (name, str(param.size()), str(param.requires_grad)))

    if args.init_ckpt > 0:
        logging.info(f'Loading checkpoint {args.init_ckpt}, {args.ckpt_path}...')
        ckpt = torch.load(args.ckpt_path, map_location='cpu')
        # model = ckpt['model']
        model.load_state_dict(ckpt['model'])
        current_steps = ckpt['steps'] * 100
        # optimizer = ckpt['optimizer']
        warm_up_steps = ckpt['warm_up_steps']
        lr = ckpt['lr']
        # args.log_dir = os.path.dirname(args.ckpt_path)
        print('resuming from', current_steps)
        with open(os.path.join(args.log_dir, 'config.json'), 'w') as f:
            json.dump({'config': args.__dict__}, f)
        wandb.watch(model)
    else:
        logging.info('Randomly Initializing Model...')
        current_steps = 0
        warm_up_steps = args.warm_up_steps
        lr = args.lr
        with open(os.path.join(args.log_dir, 'config.json'), 'w') as f:
            json.dump({'config': args.__dict__}, f)
        wandb.watch(model)
    trainer = LinkPredTrainer(model=model, datasets=dataset.datasets,
                              devices=args.gpus,
                              val_devices=args.val_gpus,
                              lr=lr,
                              warm_up_steps=warm_up_steps,
                              warm_up_factor=args.warm_up_factor,
                              num_negs=args.num_negs,
                              batch_size={"train": args.batch_size_multiple * args.batch_size,
                                          "test": args.batch_size_multiple * args.batch_size // 2,
                                          "valid": args.batch_size_multiple * args.batch_size // 2},
                              log_dir=args.log_dir,
                              layer_sizes=args.layer_sizes,
                              do_valid=args.do_valid,
                              max_grad_norm=args.max_grad_norm,
                              weight_decay=args.weight_decay,
                              masked_full_graph_sampler=args.masked_full_graph_sampler,
                              mask_once_every=args.mask_once_every,
                              num_workers=1)
    if args.init_ckpt > 0:
        # trainer.optimizer = optimizer
        trainer.optimizer.load_state_dict(ckpt['optimizer'])


    num_steps = 100 * args.num_epochs
    try:
        trainer.train_loop(num_steps, current_steps, log_every=args.log_every, val_every=args.valid_every)

        torch.save({'model': trainer.model.state_dict(), 'optimizer': trainer.optimizer.state_dict(), 'steps': num_steps},
                   os.path.join(args.log_dir, 'final.pt'))
    except Exception as err:
        del trainer
        raise err

def add_encoder_args():
    parser.add_argument('--embedding-size', type=int, default=[256, 256, 256], nargs='+')
    parser.add_argument('--rgcn-bases', type=int, default=30)
    parser.add_argument('--layer-sizes', type=int, default=[-1, -1], nargs='+')
    parser.add_argument('--masked-full-graph-sampler', action='store_true')
    parser.add_argument('--mask-once-every', type=int, default=2)
    parser.add_argument('--decoder-type', type=str, default='TransE')
    parser.add_argument('--activation', type=str, default='relu')
    parser.add_argument('--norm-layer', type=str, default='batchnorm')
    parser.add_argument('--num_pre_layers', type=int, required=True)
    parser.add_argument('--num_post_layers', type=int, required=True)
    parser.add_argument('--gamma', default=6.0, type=float)
    parser.add_argument("--dropout-prob", type=float, default=0)
    parser.add_argument("--edge-dropout", type=float, default=0.5)
    parser.add_argument("--regularization-param", type=float, default=0.01)
    parser.add_argument("--add-residual", type=boolean_string, default=False)
    parser.add_argument("--edge-disjoint", action='store_true')
    parser.add_argument("--dont-include-trial-node-features", action='store_true')
    parser.add_argument("--conv-type", default='rgcn', type=str)
    parser.add_argument('--conv-aggr', type=str, default='mean', choices=['mean', 'add', 'max'])
    parser.add_argument("--num-heads", default=1, type=int)
    parser.add_argument("--concat-gat", action='store_true')
    parser.add_argument("--scaled-attention", action='store_true')
    parser.add_argument("--attention-type", type=str, default='dot')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path', type=str, help='Path of the directory containing the data')
    parser.add_argument('--dataset', type=str, help='Name of the dataset')
    parser.add_argument("--node-feats-path", type=str, help='Path to the feature df')
    parser.add_argument("--use-population", action='store_true')
    parser.add_argument("--remove-outcomes", action='store_true')
    parser.add_argument('--use-bidirectional', action='store_true')
    parser.add_argument("--remove-function", action='store_true')
    parser.add_argument("--add-extra-dis-gene-edges", action='store_true')
    add_encoder_args()
    parser.add_argument("--random-seed", type=int, default=24)

    # Ckpt & logging
    parser.add_argument('--init-ckpt', type=int, default=0)
    parser.add_argument('--ckpt-path', type=str)
    parser.add_argument('--log-dir', type=str)
    parser.add_argument('--log-every', type=int, default=100)
    parser.add_argument('--valid-every', type=int, default=1000)
    parser.add_argument('--print-on-screen', action='store_true')
    parser.add_argument('--do-valid', action='store_true')

    # training
    parser.add_argument('--gpus', type=int, default=[-1], nargs='+', help='A list of 3 gpu ids, e.g. 0 1 2')
    parser.add_argument('--val-gpus', type=int, default=[-1], nargs='+', help='A list of 3 gpu ids, e.g. 0 1 2')
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--batch-size-multiple', type=int, default=1024)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--warm-up-steps', type=int, default=50)
    parser.add_argument('--warm-up-factor', type=int, default=2)
    parser.add_argument('--num-epochs', type=int, default=30)
    parser.add_argument('--num-negs', type=int, default=256, help='Number of negatives per positive sample')
    parser.add_argument('--negative-adversarial-sampling', action='store_true')
    parser.add_argument('--adversarial-temperature', default=1.0, type=float)
    parser.add_argument('--max-grad-norm', default=1.0, type=float)
    parser.add_argument('--weight-decay', default=1E-6, type=float)

    parser.add_argument('--wandb_project', default='clinical_trials', type=str)


    main(parser.parse_args())
