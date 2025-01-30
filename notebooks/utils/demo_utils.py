import os
import sys
import json
import pickle
import itertools

import networkx as nx
import numpy as np
import pandas as pd
import pickle as pkl
import matplotlib.pyplot as plt
from tqdm import tqdm
from argparse import Namespace
from collections import Counter
from transformers import AutoModel

import torch
from torch.utils.data import TensorDataset
from torch_geometric.data import NeighborSampler

sys.path.insert(0, "../")

from gcn_models.data_loader import ClassificationDataset, MultiTaskDataset
from gcn_models.clf_trainer import ClassifierTrainer
from gcn_models.clf_train import generate_data_path, get_dataset, untrained_encoder, populate_encoder_config
from gcn_models.evaluator import Evaluator

from gcn_models.link_pred_models import LinkPredModel, ClassificationModel, ClassificationModelWithAEEmb
from gcn_models.conv_layers.rgcn_concat import RGCNConcat
from gcn_models.utils import set_logger, get_activation, get_norm_layer, set_seed


def load_model(trained_model_path, _dataset=None, _verbose=False):
    with open(os.path.join(os.path.dirname(trained_model_path), 'config.json'), 'r') as f:
        config = json.load(f)['config']
    args = Namespace(**config)

    if _verbose:
        print(args)

    populate_encoder_config(args)

    args.test_infer_dataset_path = None

    if _dataset is None:
        _dataset = get_dataset(args, bert_data=False)

    dataset, tasks = _dataset
    if args.weighted_bce:
        pos_weight = torch.tensor(dataset.pos_weights)
    else:
        pos_weight = torch.ones(tasks[0]['num_subtasks'])
    tasks[0]['pos_weight'] = pos_weight

    if args.bert_model_path:
        bert_encoder = AutoModel.from_pretrained(args.bert_model_path)
    else:
        bert_encoder = None

    encoder_ckpt = torch.load(args.encoder_path, map_location='cpu')
    _nbr_concat = args.nbr_concat
    args.nbr_concat = False
    encoder = untrained_encoder(args, dataset)
    enc_state_dict = encoder_ckpt['model'] if isinstance(encoder_ckpt['model'], dict) else encoder_ckpt['model'].state_dict()
    missing_keys, unexpected_keys = encoder.load_state_dict(enc_state_dict, strict=False)
    # print ('missing_keys', missing_keys)
    # print ('unexpected_keys', unexpected_keys)
    args.nbr_concat = _nbr_concat
    if args.no_encoder_dropout_trial:
        encoder.encoder.no_last_layer_dropout = True
    if args.nbr_concat:
        encoder.encoder.convs.append(
            RGCNConcat(num_relations=dataset.graph.data.num_relations, rel_w=args.nbr_concat_weight,))
        encoder.encoder.nbr_concat = True
        args.layer_sizes.append(-1) #Important

    inp_dim = args.embedding_size[-1] if not args.nbr_concat or args.nbr_concat_weight else args.embedding_size[-1] * 6
    model_class = ClassificationModel
    if args.AE_emb:
        model_class = ClassificationModelWithAEEmb
    if args.combine_bert == -1:
        inp_dim = bert_encoder.config.hidden_size
    elif args.combine_bert == 1:
        inp_dim += bert_encoder.config.hidden_size
    model = model_class(
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
    ckpt = torch.load(trained_model_path, map_location='cpu')
    if ckpt['encoder']:
        missing_keys, unexpected_keys = encoder.load_state_dict(ckpt['encoder'], strict=False)
        # print ('missing_keys', missing_keys)
        # print ('unexpected_keys', unexpected_keys)
    if bert_encoder:
        bert_encoder.load_state_dict(ckpt['bert_encoder'])
    model.load_state_dict(ckpt['model'])
    return (dataset, tasks), encoder, bert_encoder, model, args


def prepare_dataset(dataset):
    df = dataset.df
    x = dataset._get_data_x(df)
    tensors = [x]
    tensors.extend([task_y for task_y in dataset.task_ys])
    tensors.extend([sample_weight_mask for sample_weight_mask in dataset.sample_weight_masks])
    dataset.datasets['all'] = TensorDataset(*tensors)
    return dataset


def prepare_runner(args, dataset, encoder, bert_encoder, model, device=0):
    args.bert_encoder_lr = 0
    runner = ClassifierTrainer(model=model,
                                    encoder=encoder,
                                    bert_encoder=bert_encoder,
                                    combine_bert=args.combine_bert,
                                    evaluate_fn=Evaluator.rank_evaluate if args.default_task_name != 'ae_clf_freq' else Evaluator.freq_evaluate,
                                    lr=0,
                                    encoder_lr=0,
                                    fixed_encoder=args.fixed_encoder,
                                    datasets=dataset.datasets,
                                    graph=dataset.graph,
                                    devices=[device]*5,
                                    val_devices=[device]*5,
                                    warm_up_steps=-1,
                                    batch_size=128,
                                    gradient_accumulation_steps=1,
                                    log_dir="/tmp",
                                    layer_sizes=args.layer_sizes,
                                    do_valid=args.do_valid,
                                    max_grad_norm=args.max_grad_norm,
                                    weight_decay=args.weight_decay,
                                    trial_features=dataset.trial_features,
                                    fp16=False,
                                    args=args,
                                    concat_trial_features=args.concat_trial_features,
                                    encoder_layers_finetune=args.encoder_layers_finetune)
    encoder.encoder.convs[2].relation_weights = False
    return args, runner, encoder


def load_model_and_data(trained_model_path, device=0):
    (dataset, tasks), encoder, bert_encoder, model, args = load_model(trained_model_path)
    dataset = prepare_dataset(dataset)
    args, runner, encoder = prepare_runner(args, dataset, encoder, bert_encoder, model, device)
    return (dataset, tasks), encoder, bert_encoder, model, args, runner


@torch.no_grad()
def model_inference(runner, mode='valid'):
    encoder, bert_encoder, model = runner.put_model(runner.devices)
    encoder.eval(); model.eval()
    pbar = runner.pbar
    pbar.set_postfix({}, refresh=False)
    model.eval()
    clf_sampler = runner.samplers[mode]
    pbar.reset(total=len(clf_sampler))
    y_pred = []
    y_true = []
    xs = []
    for i, batch in enumerate(clf_sampler):
        x, y = runner._batch_xy(batch)
        embs = runner._encode(x, runner.devices, mode='eval')
        scores = model(embs, runner.devices)
        y_pred.append(scores[0].cpu())
        y_true.append(y[0].cpu())
        xs.append(x.cpu())
        pbar.update()
    y_true = torch.cat(y_true, dim=0)
    y_scores = torch.sigmoid(torch.cat(y_pred, dim=0))
    return y_true, y_scores, torch.cat(xs, dim=0)
