import argparse
import json
import logging
import os, re
import sys

import torch
import wandb
from torch.utils.data import TensorDataset

sys.path.insert(0, "../")

from gcn_models.data_loader import ClassificationDataset, MultiTaskDataset
from gcn_models.link_pred_models import ClassificationModel, ClassificationModelWithAEEmb, LinkPredModel
from gcn_models.clf_trainer import ClassifierTrainer
from gcn_models.utils import set_logger, get_activation, get_norm_layer, set_seed, get_conv_layer
from gcn_models.train import boolean_string

from gcn_models.evaluator import Evaluator

from gcn_models.conv_layers.rgcn_concat import RGCNConcat

from datasets import Dataset, DatasetDict
from transformers import BertForSequenceClassification, AutoTokenizer, AutoModel, BertModel, Trainer, TrainingArguments

import socket, os, subprocess, datetime
print(socket.gethostname())
print ("pid:", os.getpid())
print ("screen: %s" % subprocess.check_output('echo $STY', shell=True).decode('utf'))



def generate_data_path(args):
    base_data_path = args.clf_base_data_path
    column_name = 'ae_freq_percentage'
    if args.binary_classification == 'total-based' or args.binary_classification == True:
        data_path = os.path.join(base_data_path, 'binary_data', f'trial_edges_False_RNone')
    else:
        data_path = os.path.join(base_data_path, f'event_type_{args.event_type}',
                                 f'{column_name}_{args.freq_threshold}',
                                 f'trial_thrs_{args.trial_threshold}',
                                 f'trial_edges_False_RNone')
    return data_path


def populate_encoder_config(args):
    if not args.encoder_path:
        return
    with open(os.path.join(os.path.dirname(args.encoder_path), 'config.json'), 'r') as f:
        config = json.load(f)['config']
    if type(config['embedding_size']) == int:
        config['embedding_size'] = [config['embedding_size']] * 3
    for key in ['data_path', 'dataset', 'use_bidirectional', 'use_population', 'remove_function', 'remove_outcomes',
                'node_feats_path', 'embedding_size', 'dont_include_trial_node_features', 'rgcn_bases', 'activation', 'enc_dropout_prob', 'edge_dropout']:
        if key in config:
            setattr(args, key, config[key])
    setattr(args, 'layer_sizes', [-1, -1])
    setattr(args, 'remove_function', False)
    if not hasattr(args, 'remove_outcomes'):
        setattr(args, 'remove_outcomes', False)
    if not hasattr(args, 'dont_include_trial_node_features'):
        setattr(args, 'dont_include_trial_node_features', False)
    if not hasattr(args, 'use_population'):
        setattr(args, 'use_population', False)


def get_dataset(args, bert_data=False):
    data_path = args.data_path

    dataset = args.dataset
    data_files = ['train.tsv', 'valid.tsv', 'test.tsv']

    if args.dataset_version == 1:
        clf_data_path = generate_data_path(args)
        dataset = ClassificationDataset(data_path=data_path, dataset=dataset, data_files=data_files,
                                        bidirectional=args.use_bidirectional, use_population=args.use_population,
                                        remove_function=args.remove_function,
                                        add_extra_dis_gene_edges=args.add_extra_dis_gene_edges,
                                        test_infer_path=None, #args.test_infer_dataset_path,
                                        node_feats_df_path=args.node_feats_path,
                                        label_column_name=args.label_column_name,
                                        is_binary_classification=args.binary_classification,
                                        binary_threshold=args.binary_clf_threshold,
                                        clf_df_path=clf_data_path,
                                        trial_feats_df_path=args.trial_feats_path,
                                        concat_trial_features=args.concat_trial_features,
                                        split=args.split_strategy,
                                        balanced_sample=args.balanced_sample,
                                        enrollment_filter=args.enrollment_filter,
                                        dont_include_trial_node_features=args.dont_include_trial_node_features,
                                        no_relations=args.no_relations,
                                        single_ae=args.single_ae_name,
                                        dont_include_bert_embs_trial=args.dont_include_bert_embs_trial,
                                        only_trial_embeddings=args.only_trial_embeddings,
                                        bert_data=bert_data,
                                        args=args)
        tasks = [{
            'name': 'ae_clf',
            'subtask_weight': args.task_weight,
            'num_subtasks': dataset.num_tasks,
            'task_weight': 1
        }]
    else:
        if args.binary_classification == '': #this is the default
            tasks = [{
                'name': args.default_task_name,
                'merge_strategy': 'max' if args.event_type == 'all' else args.event_type,
                'task_weight': 1,
                'threshold': args.freq_threshold,
                'example_threshold': args.trial_threshold,
                # 'test_example_threshold': args.test_trial_threshold,
                'subtask_weight': args.task_weight,
            }]
        elif args.binary_classification == 'total-based':
            tasks = [{
                'name': args.default_task_name,
                'task_weight': 1,
                'threshold': args.binary_clf_threshold,
                'subtask_weight': args.task_weight
            }]
        else:
            raise RuntimeError("Unsupported value: ", args.binary_classification)
        for idx, task in enumerate(args.extra_tasks):
            tasks.append({
                'name': task,
                'merge_strategy': 'max' if args.event_type == 'all' else 'serious',
                'task_weight': args.extra_task_weight[idx],
                'threshold': args.binary_clf_threshold,
                'subtask_weight': args.task_weight
            })
        clf_data_path = args.clf_base_data_path
        #this is used
        dataset = MultiTaskDataset(data_path=data_path, dataset=dataset, data_files=data_files,
                                   bidirectional=args.use_bidirectional, use_population=args.use_population,
                                   remove_function=args.remove_function,
                                   add_extra_dis_gene_edges=args.add_extra_dis_gene_edges,
                                   test_infer_path=None, #args.test_infer_dataset_path,
                                   remove_outcomes=args.remove_outcomes,
                                   node_feats_df_path=args.node_feats_path, clf_df_path=clf_data_path,
                                   trial_feats_df_path=args.trial_feats_path,
                                   concat_trial_features=args.concat_trial_features,
                                   combine_bert=args.combine_bert, #Added
                                   bert_model_path=args.bert_model_path, #Added
                                   bert_max_seq_len=args.bert_max_seq_len, #Added
                                   tasks=tasks,
                                   split=args.split_strategy,
                                   enrollment_filter=args.enrollment_filter,
                                   no_relations=args.no_relations,
                                   dont_include_bert_embs_trial=args.dont_include_bert_embs_trial,
                                   only_trial_embeddings=args.only_trial_embeddings,
                                   bert_data=bert_data,
                                   args=args)
    return dataset, tasks


def untrained_encoder(args, dataset):
    train_data = dataset.graph.data
    return LinkPredModel(num_relations=train_data.num_relations, in_channels=train_data.num_nodes,
                         hidden_dim=args.embedding_size, emb_df=dataset.node_feats,
                         num_enc_layers=len(args.layer_sizes),
                         nbr_concat=args.nbr_concat,
                         nbr_concat_weight=args.nbr_concat_weight,
                         num_bases=min(args.rgcn_bases, train_data.num_relations), decoder_type=None,
                         activation_fn=get_activation(args),
                         norm_fn=get_norm_layer(args),
                         gamma=None, dropout=args.enc_dropout_prob, edge_dropout=args.edge_dropout,
                         reg_param=None, adversarial_temperature=None,
                         negative_adversarial_sampling=None,
                         add_residual=args.add_residual,
                         conv_fn=get_conv_layer(args),
                         conv_aggr=args.conv_aggr,
                         num_pre_layers=0,
                         num_post_layers=0,
                         bert_dim=args.bert_dim,
                         )


def train_and_save(args, dataset, tasks, summary_prefix):
    print ('dataset.graph.data.num_nodes', dataset.graph.data.num_nodes)
    #Added
    if args.bert_model_path:
        print (f'loading BertModel: {args.bert_model_path} ...')
        bert_encoder = AutoModel.from_pretrained(args.bert_model_path)
        print ('loading BertModel done')
    else:
        bert_encoder = None

    if args.combine_bert == -1: #only use bert, no GCN
        encoder = None
        encoder_lr = 0
    else:
        if args.no_pretraining: #Not used
            print("Initializing new encoder model")
            print(args.layer_sizes)
            encoder = untrained_encoder(args, dataset)
            encoder_lr = args.lr
        else:
            print ('args.encoder_path', args.encoder_path)
            encoder_ckpt = torch.load(args.encoder_path, map_location='cpu') if args.encoder_path else None
            if True:
                args.bert_dim = 0
                _nbr_concat = args.nbr_concat
                args.nbr_concat = False
                encoder = untrained_encoder(args, dataset)
                if encoder_ckpt:
                    enc_state_dict = encoder_ckpt['model'] if isinstance(encoder_ckpt['model'], dict) else encoder_ckpt['model'].state_dict()
                    missing_keys, unexpected_keys = encoder.load_state_dict(enc_state_dict, strict=False)
                    print ('missing_keys', missing_keys)
                    print ('unexpected_keys', unexpected_keys)
                else:
                    print("Initializing new encoder model")
                args.nbr_concat = _nbr_concat

            if args.no_encoder_dropout_trial:
                encoder.encoder.no_last_layer_dropout = True
            if args.nbr_concat:
                encoder.encoder.convs.append(
                    RGCNConcat(num_relations=dataset.graph.data.num_relations, rel_w=args.nbr_concat_weight,))
                encoder.encoder.nbr_concat = True
                args.layer_sizes.append(-1) #Important
            encoder_lr = encoder_ckpt['lr'] / args.encoder_lr_factor if encoder_ckpt else args.lr
    print("Encoder lr is ", encoder_lr)


    inp_dim = args.embedding_size[-1] if not args.nbr_concat or args.nbr_concat_weight else args.embedding_size[-1] * 6
    if bert_encoder:
        args.bert_dim = bert_encoder.config.hidden_size
    model_class = ClassificationModel
    if args.AE_emb:
        model_class = ClassificationModelWithAEEmb
    if dataset.combine_bert == -1:
        inp_dim = bert_encoder.config.hidden_size
    elif dataset.combine_bert == 1:
        inp_dim += bert_encoder.config.hidden_size
    elif dataset.concat_trial_features:
        inp_dim += args.trial_feats_size
    elif dataset.trial_features:
        inp_dim = args.trial_feats_size
    clf_model = model_class(
        input_dim=inp_dim,
        normalize_embeddings=args.normalize_embeddings,
        normalize_embeddings_with_grad=args.normalize_embeddings_with_grad,
        normalize_clf_weights=args.normalize_clf_weights,
        hidden_sizes=args.hidden_size,
        num_layers=args.num_layers,
        activation_fn=get_activation(args),
        norm_layer=get_norm_layer(args),
        dropout_prob=args.dropout_prob,
        tasks=tasks,
        args=args,
    )

    if hasattr(encoder, 'decoder'):
        del encoder.decoder
    print(encoder)

    if args.init_ckpt > 0:
        logging.info(f'Loading checkpoint {args.init_ckpt}, {args.ckpt_path}...')
        ckpt = torch.load(args.ckpt_path, map_location='cpu')
        clf_model = ckpt['model']
        encoder = ckpt['encoder']
        current_steps = ckpt['steps']
        optimizer = ckpt['optimizer']
        warm_up_steps = ckpt['warm_up_steps']
        lr = ckpt['lr']
        encoder_lr = ckpt['encoder_lr']
        print(current_steps)
    elif args.load_trained_model_path:
        print (f'Loading model from {args.load_trained_model_path} ...')
        ckpt = torch.load(args.load_trained_model_path, map_location='cpu')
        if encoder is not None and ckpt['encoder'] is not None:
            encoder.load_state_dict(ckpt['encoder'])
        if bert_encoder is not None and ckpt['bert_encoder'] is not None:
            bert_encoder.load_state_dict(ckpt['bert_encoder'])
        try:
            clf_model.load_state_dict(ckpt['model'])
        except:
            state_name_to_ignore = {r"task_heads[.]*", r"task_losses[.]*", r"AEemb[.]*"}
            for name, param in ckpt['model'].items():
                ignore = False
                for pttn_to_ignore in state_name_to_ignore:
                    if re.match(pttn_to_ignore, name):
                        ignore = True
                        break
                if not ignore:
                    print ('params: keeping', name)
                    param = param.data
                    clf_model.state_dict()[name].copy_(param)
                else:
                    print ('params: not keeping', name)
        current_steps = 0
        warm_up_steps = args.warm_up_steps
        lr = args.lr
    elif args.load_trained_model_bert_path or args.load_trained_model_rgcn_path:
        if args.load_trained_model_bert_path:
            print (f'Loading bert model from {args.load_trained_model_bert_path} ...')
            ckpt = torch.load(args.load_trained_model_bert_path, map_location='cpu')
            bert_encoder.load_state_dict(ckpt['bert_encoder'])
        if args.load_trained_model_rgcn_path:
            print (f'Loading rgcn model from {args.load_trained_model_rgcn_path} ...')
            ckpt = torch.load(args.load_trained_model_rgcn_path, map_location='cpu')
            missing_keys, unexpected_keys = encoder.load_state_dict(ckpt['encoder'], strict=False)
            print ('load_trained_model_rgcn_path: missing_keys', missing_keys)
            print ('load_trained_model_rgcn_path: unexpected_keys', unexpected_keys)
        current_steps = 0
        warm_up_steps = args.warm_up_steps
        lr = args.lr
    else:
        logging.info('Randomly Initializing Model...'); print ('Randomly Initializing Model...')
        current_steps = 0
        warm_up_steps = args.warm_up_steps
        lr = args.lr
    with open(os.path.join(args.log_dir, 'config.json'), 'w') as f:
        json.dump({'config': args.__dict__}, f)

    trainer = ClassifierTrainer(model=clf_model,
                                encoder=encoder,
                                bert_encoder=bert_encoder,
                                combine_bert=dataset.combine_bert,
                                evaluate_fn=Evaluator.rank_evaluate if args.default_task_name != 'ae_clf_freq' else Evaluator.freq_evaluate,
                                lr=lr,
                                encoder_lr=encoder_lr,
                                fixed_encoder=args.fixed_encoder,
                                datasets=dataset.datasets,
                                graph=dataset.graph,
                                devices=args.gpus,
                                val_devices=args.val_gpus,
                                warm_up_steps=warm_up_steps,
                                batch_size=args.batch_size,
                                gradient_accumulation_steps=args.gradient_accumulation_steps,
                                log_dir=args.log_dir,
                                layer_sizes=args.layer_sizes,
                                do_valid=args.do_valid,
                                max_grad_norm=args.max_grad_norm,
                                weight_decay=args.weight_decay,
                                trial_features=dataset.trial_features,
                                concat_trial_features=dataset.concat_trial_features,
                                encoder_layers_finetune=args.encoder_layers_finetune,
                                fp16=args.fp16,
                                summary_prefix=summary_prefix, args=args)

    print ('Model Parameter Configuration:')
    for name, param in clf_model.named_parameters():
        print ('Parameter %s: %s, require_grad = %s' % (name, str(param.size()), str(param.requires_grad)))
    if encoder is not None:
        for name, param in encoder.named_parameters():
            print ('Parameter %s: %s, require_grad = %s' % (name, str(param.size()), str(param.requires_grad)))
    if bert_encoder is not None:
        for name, param in bert_encoder.named_parameters():
            print ('Parameter %s: %s, require_grad = %s' % (name, str(param.size()), str(param.requires_grad)))

    wandb.watch(clf_model)
    if bert_encoder is not None:
        wandb.watch(bert_encoder)
    # if encoder is not None:
    #     wandb.watch(encoder)

    if args.init_ckpt > 0:
        trainer.optimizer = optimizer

    num_steps = args.num_epochs
    try:
        trainer.train_loop(num_epochs=args.num_epochs, old_epochs=current_steps, val_every=args.valid_every)

        encoder_save = trainer.encoder.state_dict() if trainer.encoder is not None else None
        bert_encoder_save = trainer.bert_encoder.state_dict() if trainer.bert_encoder is not None else None
        torch.save({'model': trainer.model.state_dict(), 'optimizer': trainer.optimizer.state_dict(), 'steps': num_steps,
                    "encoder": encoder_save, 'bert_encoder': bert_encoder_save},
                   os.path.join(args.log_dir, 'final.pt'))
        if args.task_weight == 'variance':
            print(torch.exp(-clf_model.task_loss.log_vars))
    except Exception as err:
        del trainer
        raise err


def main(args: argparse.Namespace):
    # Write logs to checkpoint and console
    if not args.no_pretraining:
        populate_encoder_config(args)

    dt = datetime.datetime.today().strftime('%Y%m%d_%H%M')
    wandb_id = wandb.util.generate_id()
    wandb_name = f"{dt}--{wandb_id}--{args.exp_name}"
    wandb.init(config=args, project=args.wandb_project, name=wandb_name, id=wandb_id, dir='../')
    assert wandb.run.id == wandb_id
    args.log_dir = os.path.join(args.log_dir, wandb_name)
    os.makedirs(args.log_dir)
    print(args.log_dir)
    set_logger(args)
    set_seed(args.random_seed)

    if args.combine_bert:
        assert args.bert_model_path

    dataset, tasks = get_dataset(args)
    print ('dataset loaded')
    # torch.save(dataset, os.path.join(args.log_dir, 'clf-dataset.pt'))
    # print ('dataset saved')

    if args.weighted_bce:
        pos_weight = torch.tensor(dataset.pos_weights)
        print ('Using weighted_bce', len(pos_weight), pos_weight)
    else:
        pos_weight = torch.ones(tasks[0]['num_subtasks'])
    tasks[0]['pos_weight'] = pos_weight

    if args.separate_classifier:
        num_aes = tasks[0]['num_subtasks']
        tasks[0]['num_subtasks'] = 1
        orig_datasets = dataset.datasets.copy()
        orig_pos_weights = tasks[0]['pos_weight']
        for i in range(num_aes):
            for key in orig_datasets:
                dataset.datasets[key] = TensorDataset(orig_datasets[key].tensors[0],
                                                      orig_datasets[key].tensors[1][:, i:i + 1])
            tasks[0]['pos_weight'] = orig_pos_weights[i]
            train_and_save(args, dataset, tasks, f'ae_{i}')
    else:
        train_and_save(args, dataset, tasks, '')


def add_dataset_args(parser):
    # trial features
    parser.add_argument("--trial-feats-path", type=str, help='Path to the feature df of trial nodes')
    parser.add_argument("--trial-feats-size", type=int, help='Size of trial node embeddings')

    # Clf datapath
    parser.add_argument('--dataset-version', type=int)
    parser.add_argument('--clf-base-data-path', type=str, help='Path of the directory containing the data')
    parser.add_argument('--split-strategy', type=str, default='drug-disease')
    parser.add_argument('--enrollment-filter', type=int, default=-1)
    parser.add_argument('--balanced-sample', action='store_true')
    parser.add_argument('--dont_include_bert_embs_trial', action='store_true')
    parser.add_argument('--only_trial_embeddings', action='store_true')
    parser.add_argument('--no-relations', action='store_true')
    # Args describing the dataset

    # multi-task
    parser.add_argument('--default-task-name', type=str, required=True)
    parser.add_argument('--extra-tasks', type=str, default=[], nargs='+')
    parser.add_argument('--extra-task-weight', type=float, default=[], nargs='+')

    parser.add_argument("--task-weight", type=str, default='uniform')

    parser.add_argument("--event-type", type=str)
    parser.add_argument("--freq-threshold", type=int)
    parser.add_argument("--trial-threshold", type=int)
    # parser.add_argument("--test-trial-threshold", type=int)
    parser.add_argument("--label-column-name", type=str)
    parser.add_argument("--binary-classification", type=str, default='')
    parser.add_argument("--binary-clf-threshold", type=float, default=None)


def add_encoder_args():
    parser.add_argument('--data-path', type=str, help='Path of the directory containing the data')
    parser.add_argument('--dataset', type=str, help='Name of the dataset')
    parser.add_argument("--node-feats-path", type=str, help='Path to the feature df')
    parser.add_argument("--use-population", action='store_true')
    parser.add_argument('--use-bidirectional', action='store_true')
    parser.add_argument("--remove-function", action='store_true')
    parser.add_argument("--remove-outcomes", action='store_true')
    parser.add_argument("--add-extra-dis-gene-edges", action='store_true')

    parser.add_argument('--embedding-size', type=int, default=[256, 256, 256], nargs='+')
    parser.add_argument('--rgcn-bases', type=int, default=30)
    parser.add_argument('--layer-sizes', type=int, default=[-1, -1], nargs='+')
    parser.add_argument("--enc-dropout-prob", type=float, default=0)
    parser.add_argument("--edge-dropout", type=float, default=0.5)
    parser.add_argument("--add-residual", type=boolean_string, default=False)
    parser.add_argument("--dont-include-trial-node-features", action='store_true')
    parser.add_argument("--conv-type", default='rgcn', type=str)
    parser.add_argument('--conv-aggr', type=str, default='mean', choices=['mean', 'add', 'max'])
    parser.add_argument('--nbr-concat', action='store_true')
    parser.add_argument('--nbr-concat-weight', action='store_true')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--hidden-size', type=int, nargs='+', default=[])
    parser.add_argument('--num-layers', type=int, default=2)
    parser.add_argument('--activation', type=str, default='relu')
    parser.add_argument('--norm-layer', type=str, default='batchnorm')
    # parser.add_argument('--layer-sizes', type=int, default=[-1, -1], nargs='+')
    parser.add_argument("--weighted-bce", action='store_true', default=False)
    parser.add_argument("--dropout-prob", type=float, default=0.5)
    parser.add_argument("--single-ae-name", type=str, default=None)
    parser.add_argument("--random-seed", type=int, default=24)
    parser.add_argument('--concat-trial-features', action='store_true')
    parser.add_argument('--no-encoder-dropout-trial', action='store_true')

    add_dataset_args(parser)
    add_encoder_args()

    # Encoder
    parser.add_argument('--encoder-path', type=str)
    parser.add_argument('--fixed-encoder', action='store_true')
    parser.add_argument('--encoder-lr-factor', type=float)
    parser.add_argument('--encoder-layers-finetune', type=str, default=[], nargs='+')
    parser.add_argument('--normalize-embeddings', action='store_true')
    parser.add_argument('--normalize-embeddings-with-grad', action='store_true')
    parser.add_argument('--normalize-clf-weights', action='store_true')

    parser.add_argument('--bert-model-path', type=str, default='')
    parser.add_argument('--bert-ignore-pooler', action='store_true')
    parser.add_argument('--combine-bert', type=int, default=0)
    parser.add_argument('--bert_text_path', type=str, default='')
    parser.add_argument('--AE_emb', type=int, default=0)
    parser.add_argument('--bert_unfreeze_epoch', type=int, default=1)
    parser.add_argument('--bert-max-seq-len', type=int, default=512)
    parser.add_argument('--bert-dim', type=int, default=0)
    parser.add_argument('--cross-attn-num-layers', type=int, default=1)
    parser.add_argument('--bert_encoder_lr', type=float, default=5e-5)

    parser.add_argument('--load_trained_model_path', type=str, default='')
    parser.add_argument('--load_trained_model_bert_path', type=str, default='')
    parser.add_argument('--load_trained_model_rgcn_path', type=str, default='')
    parser.add_argument('--subsample_train', type=int, default=0)
    parser.add_argument('--combo_weight', type=float, default=0)
    parser.add_argument('--save_model', type=int, default=1)



    # Ckpt & logging
    parser.add_argument('--init-ckpt', type=int, default=0)
    parser.add_argument('--ckpt-path', type=str)
    parser.add_argument('--log-dir', type=str)
    parser.add_argument('--valid-every', type=int, default=1000)
    parser.add_argument('--print-on-screen', action='store_true')
    parser.add_argument('--do-valid', action='store_true')


    # training
    parser.add_argument('--gpus', type=int, default=[-1], nargs='+', help='A list of 3 gpu ids, e.g. 0 1 2')
    parser.add_argument('--val-gpus', type=int, default=[-1], nargs='+', help='A list of 3 gpu ids, e.g. 0 1 2')
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--gradient-accumulation-steps', default=1, type=int)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--warm-up-steps', type=int, default=50)
    parser.add_argument('--warm-up-factor', type=int, default=2)
    parser.add_argument('--warm_up_with_linear_decay', type=float, default=-1)
    parser.add_argument('--num-epochs', type=int, default=30)
    parser.add_argument('--max-grad-norm', default=1.0, type=float)
    parser.add_argument('--weight-decay', default=1E-6, type=float)

    parser.add_argument('--separate-classifier', action='store_true')
    parser.add_argument('--no-pretraining', action='store_true')

    parser.add_argument('--wandb_project', type=str, default='clinical_trials')
    parser.add_argument('--exp_name', type=str, default='')
    parser.add_argument('--fp16', action='store_true') #Not working currently

    main(parser.parse_args())
