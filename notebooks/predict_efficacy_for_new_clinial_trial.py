import os
import sys
import json
import torch
import pickle
import itertools

import networkx as nx
import numpy as np
import pandas as pd
import pickle as pkl
import matplotlib.pyplot as plt 
from tqdm import tqdm
from collections import Counter 
from copy import deepcopy
from torch.utils.data import TensorDataset

import time
from contextlib import contextmanager
import threading

@contextmanager
def suppress_output_with_progress(description):
    """
    Suppresses stdout/stderr while showing an automatically updating progress timer.
    """
    # Save the original stdout/stderr
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    
    # Open null device
    null_fd = os.open(os.devnull, os.O_WRONLY)
    try:
        # Redirect stdout/stderr to null device
        new_stdout = os.dup(1)
        new_stderr = os.dup(2)
        
        # Create progress bar before redirecting output
        pbar = tqdm(desc=f"{description}... Time elapsed", 
                   bar_format='{desc}: {n:.1f}s',
                   ncols=40)
        
        # Now redirect output
        os.dup2(null_fd, 1)
        os.dup2(null_fd, 2)
        
        # Start time tracking
        start_time = time.time()
        
        # Create flag for thread control
        running = threading.Event()
        running.set()
        
        def update_thread():
            while running.is_set():
                elapsed = time.time() - start_time
                pbar.n = elapsed
                pbar.refresh()
                time.sleep(0.1)  # Update every 0.1 seconds
        
        # Start update thread
        thread = threading.Thread(target=update_thread)
        thread.daemon = True  # Make thread daemon so it exits when main thread exits
        thread.start()
        
        try:
            yield
        finally:
            running.clear()  # Signal thread to stop
            thread.join()    # Wait for thread to finish
            pbar.close()
            
    finally:
        # Restore original stdout/stderr
        os.dup2(new_stdout, 1)
        os.dup2(new_stderr, 2)
        # Close all file descriptors
        os.close(null_fd)
        os.close(new_stdout)
        os.close(new_stderr)
        sys.stdout = old_stdout
        sys.stderr = old_stderr


sys.path.insert(0, "../")

import knowledge_graph
from knowledge_graph.kg import *

def load_kg_vocab():
    kgid2x = {}
    x2kgid = {}
    for line in open('../data/graph/entities.dict'):
        x, kgid = line.split()
        kgid2x[kgid] = int(x)
        x2kgid[int(x)] = kgid
    relname2etype = {}
    etype2relname = {}
    for line in open('../data/graph/relations.dict'):
        etype, name = line.split()
        name = name[name.find('rel-name-'):][len('rel-name-'):]
        relname2etype[name] = int(etype)
        etype2relname[int(etype)] = name 
    return kgid2x, x2kgid, relname2etype, etype2relname

kgid2x, x2kgid, relname2etype, etype2relname = load_kg_vocab()

from utils.demo_utils import load_model, load_model_and_data, model_inference, prepare_runner
from gcn_models.utils import set_seed
from gcn_models.evaluator import Evaluator
set_seed(24)

#Load BERT embedder
from utils.text_bert_features import TextBertFeatures
bert_model = TextBertFeatures(
    bert_model='microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext',
    device='cpu'
)


# Load the new clinical trial data
new_trial_data = pkl.load(open('small_data/trial_data_NCT02370680.pkl', 'rb'))

new_trial_1, new_trial_2 = new_trial_data


def get_trial_feature(new_trial):
    _new_emb = bert_model._embed(new_trial['arm_text'])
    return np.concatenate([_new_emb, new_trial['trial_attribute_feats_vec']])

with suppress_output_with_progress("Encoding data"):
    new_emb_1 = get_trial_feature(new_trial_1)
    new_emb_2 = get_trial_feature(new_trial_2)


def get_new_edges(new_trial, the_x):
    new_edges = []; new_etypes = []; seen = set()
    for edge in new_trial_1['trial_arm_edges']:
        h = the_x
        t = kgid2x[edge['kg_id']]
        r = relname2etype[edge['relation']]
        new_edges.append([h,t])
        new_etypes.append(r)
        new_edges.append([t,h])
        new_etypes.append(r+26)
        seen.add(r)
    for r in [21,22,23,24,25]:
        if r not in seen:
            new_edges.append([the_x,0])
            new_etypes.append(r)
            new_edges.append([0,the_x])
            new_etypes.append(r+26)
    new_edges = torch.tensor(new_edges).t()
    new_etypes = torch.tensor(new_etypes)
    # print(new_edges.shape)
    # print(new_etypes.shape)
    return new_edges, new_etypes



# Predict for the efficacy task
def add_new_trial_to_efficacy_dataset(dataset):
    # Update the df
    _df = dataset.efficacy_df[dataset.efficacy_df['split']=='test'].head(1)
    dataset.efficacy_df = _df
    the_x1, the_x2 = _df.iloc[0]['x1'], _df.iloc[0]['x2']
    the_kgid1, the_kgid2 = _df.iloc[0]['kgid1'], _df.iloc[0]['kgid2']
    # print('the_x1', the_x1, 'the_kgid1', the_kgid1, 'the_x2', the_x2, 'the_kgid2', the_kgid2)
    
    # Update the node features
    _node_feats = deepcopy(dataset.node_feats)
    pos = _node_feats[_node_feats['node_id'] == the_kgid1].index[0]
    _node_feats.at[pos, 'emb'] = new_emb_1
    pos = _node_feats[_node_feats['node_id'] == the_kgid2].index[0]
    _node_feats.at[pos, 'emb'] = new_emb_2
    dataset.node_feats = _node_feats
    
    # Update the graph edges
    _graph = deepcopy(dataset.graph)
    new_edges_1, new_etypes_1 = get_new_edges(new_trial_1, the_x1)
    new_edges_2, new_etypes_2 = get_new_edges(new_trial_2, the_x2)
    
    _edge_index = _graph.data.edge_index.clone()
    columns_with_the_x1 = _edge_index.eq(the_x1).any(dim=0)
    _edge_index = _edge_index[:, ~columns_with_the_x1]
    columns_with_the_x2 = _edge_index.eq(the_x2).any(dim=0)
    _edge_index = _edge_index[:, ~columns_with_the_x2]
    _edge_index = torch.cat([_edge_index, new_edges_1, new_edges_2], dim=1)
    _graph.data.edge_index = _edge_index
    
    _edge_type = _graph.data.edge_type.clone()
    _edge_type = _edge_type[~columns_with_the_x1]
    _edge_type = _edge_type[~columns_with_the_x2]
    _edge_type = torch.cat([_edge_type, new_etypes_1, new_etypes_2], dim=0)
    _graph.data.edge_type = _edge_type
    
    dataset.graph = _graph
    
    # Make the dataset ready for inference
    x = dataset._get_data_x(_df)
    # print('x', x)
    tensors = [x]
    tensors.extend([dataset.task_ys[0][0].repeat(len(x), 1)])
    tensors.extend([dataset.sample_weight_masks[0][0].repeat(len(x), 1)])
    dataset.datasets['test'] = TensorDataset(*tensors)
    
    return dataset

with suppress_output_with_progress("Running efficacy task"):
    (ef_dataset, ef_tasks), ef_encoder, ef_bert_encoder, ef_model, ef_args \
        = load_model('../data/models/efficacy_model_34l5ms9m/ckpt.pt')

    ef_dataset_bk = deepcopy(ef_dataset)
    ef_tasks_bk = deepcopy(ef_tasks)
    ef_args_bk = deepcopy(ef_args)

    ef_dataset = add_new_trial_to_efficacy_dataset(ef_dataset)
    ef_args, ef_runner, ef_encoder \
        = prepare_runner(ef_args, ef_dataset, ef_encoder, ef_bert_encoder, ef_model, device='cpu')

    _, y_test_pred, _ = model_inference(ef_runner, mode='test') 


efficacy_pred = y_test_pred[0].item()


print()
print('*' * 50)
print('Probabilty of trial 1 being more effective than trial 2:', efficacy_pred)
print('*' * 50)