from typing import List

from torch.utils.data import TensorDataset

try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper

import numpy as np
import os
from torch_geometric.data import Data
import torch
import pandas as pd
import pickle as pkl


from datasets import Dataset as hf_Dataset
from transformers import BertForSequenceClassification, AutoTokenizer, BertModel, Trainer, TrainingArguments
from collections import defaultdict


class KGDataset:
    """
    Load a knowledge graph

    The folder with a knowledge graph has five files:
    * entities stores the mapping between entity Id and entity name.
    * relations stores the mapping between relation Id and relation name.
    * train stores the triples in the training set.
    * valid stores the triples in the validation set.
    * test stores the triples in the test set.

    The mapping between entity (relation) Id and entity (relation) name is stored as 'id\tname'.

    The triples are stored as 'head_name\trelation_name\ttail_name'.
    """

    def __init__(self, train_path,
                 valid_path=None, test_path=None, format=(0, 1, 2),
                 delimiter='\t', skip_first_line=False, use_population: bool = False, remove_function: bool = False,
                 remove_outcomes: bool = False, add_extra_dis_gene_edges: bool = False, test_infer_path: str = None):
        self.delimiter = delimiter
        self.use_population = use_population
        self.remove_outcomes = remove_outcomes
        self.remove_function = remove_function
        self.train = self.read_triple(train_path, "train", add_extra_dis_gene_edges, skip_first_line, format)
        if valid_path is not None:
            self.valid = self.read_triple(valid_path, "valid", add_extra_dis_gene_edges, skip_first_line, format)
        else:
            self.valid = None
        if test_path is not None:
            self.test = self.read_triple(test_path, "test", add_extra_dis_gene_edges, skip_first_line, format)
        else:
            self.test = None
        if test_infer_path is not None:
            self.infer_test = self.read_triple(test_infer_path + "test_extra.tsv", "test_infer",
                                               add_extra_dis_gene_edges, skip_first_line, format)
        else:
            self.infer_test = None

    def read_triple(self, path, mode, read_extra: bool = False, skip_first_line=False, format=(0, 1, 2)):
        # mode: train/valid/test
        if path is None:
            return None

        print('Reading {} triples....'.format(mode))
        heads = []
        tails = []
        rels = []
        print(self.remove_outcomes)
        with open(path) as f:
            if skip_first_line:
                _ = f.readline()
            for line in f:
                triple = line.strip().split(self.delimiter)
                h, r, t = triple[format[0]], triple[format[1]], triple[format[2]]
                if not self.use_population and r in ['source-CLINICAL_TRIAL:rel-name-eligibility-exclusion',
                                                     'source-CLINICAL_TRIAL:rel-name-eligibility-inclusion']:
                    continue
                if self.remove_outcomes and r in ['source-CLINICAL_TRIAL:rel-name-primary_outcome']:
                    continue
                if self.remove_function and r in ['source-GO:rel-name-is_a', 'source-GO:rel-name-part_of',
                                                  'source-GAF:rel-name-affects', 'source-GO:rel-name-regulates',
                                                  'source-GO:rel-name-negatively_regulates',
                                                  'source-GO:rel-name-positively_regulates',
                                                  'source-CTD:rel-name-decreases', 'source-CTD:rel-name-increases',
                                                  'source-CTD:rel-name-affects'
                                                  ]:
                    continue
                heads.append(self.entity2id[h])
                rels.append(self.relation2id[r])
                tails.append(self.entity2id[t])
        if read_extra:
            with open(os.path.join(os.path.dirname(path), 'extra-dis-gene-' + os.path.basename(path))) as f:
                if skip_first_line:
                    _ = f.readline()
                for line in f:
                    triple = line.strip().split(self.delimiter)
                    h, r, t = triple[format[0]], triple[format[1]], triple[format[2]]
                    heads.append(self.entity2id[h])
                    rels.append(self.relation2id[r])
                    tails.append(self.entity2id[t])
        heads = np.array(heads, dtype=np.int64)
        tails = np.array(tails, dtype=np.int64)
        rels = np.array(rels, dtype=np.int64)
        print('Finished. Read {} {} triples.'.format(len(heads), mode))

        return heads, rels, tails


def _parse_srd_format(format):
    if format == "hrt":
        return [0, 1, 2]
    if format == "htr":
        return [0, 2, 1]
    if format == "rht":
        return [1, 0, 2]
    if format == "rth":
        return [2, 0, 1]
    if format == "thr":
        return [1, 2, 0]
    if format == "trh":
        return [2, 1, 0]


def read_dictionary(filename, id_lookup=True, name_lookup=False):
    d = {}
    name_d = {}
    for line in open(filename, 'r'):
        line = line.strip().split('\t')
        if name_lookup:
            name_d[line[2]] = line[1]
        if id_lookup:
            d[int(line[0])] = line[1]
        else:
            d[line[1]] = int(line[0])
    if name_lookup:
        return d, name_d
    return d


class KGDatasetUDDRaw(KGDataset):
    """Load a knowledge graph user defined dataset

    The user defined dataset has five files:
    * entities stores the mapping between entity name and entity Id.
    * relations stores the mapping between relation name relation Id.
    * train stores the triples in the training set. In format [src_name, rel_name, dst_name]
    * valid stores the triples in the validation set. In format [src_name, rel_name, dst_name]
    * test stores the triples in the test set. In format [src_name, rel_name, dst_name]

    The mapping between entity (relation) name and entity (relation) Id is stored as 'name\tid'.
    The triples are stored as 'head_nid\trelation_id\ttail_nid'. Users can also use other delimiters
    other than \t.
    """

    def __init__(self, path, name, delimiter, files, format, *, use_population, remove_function, remove_outcomes,
                 add_extra_dis_gene_edges, test_infer_path):
        self.name = name
        self.test_infer_path = test_infer_path
        for f in files:
            assert os.path.exists(os.path.join(path, f)), \
                'File {} does not exist in {}'.format(f, path)

        assert len(format) == 3
        format = _parse_srd_format(format)
        self._load_entity_relation(path, test_infer_path)

        assert len(files) == 1 or len(files) == 3, 'raw_udd_{htr} format requires 1 or 3 input files. ' \
                                                   'When 1 files are provided, they must be train_file. ' \
                                                   'When 3 files are provided, they must be train_file, valid_file and test_file.'

        if delimiter not in ['\t', '|', ',', ';']:
            print('WARNING: delimiter {} is not in \'\\t\', \'|\', \',\', \';\'' \
                  'This is not tested by the developer'.format(delimiter))
        # Only train set is provided
        if len(files) == 1:
            super(KGDatasetUDDRaw, self).__init__(os.path.join(path, files[0]),
                                                  format=format,
                                                  delimiter=delimiter,
                                                  use_population=use_population,
                                                  remove_function=remove_function,
                                                  remove_outcomes=remove_outcomes,
                                                  add_extra_dis_gene_edges=add_extra_dis_gene_edges,
                                                  test_infer_path=test_infer_path)
        # Train, validation and test set are provided
        elif len(files) == 3:
            super(KGDatasetUDDRaw, self).__init__(os.path.join(path, files[0]),
                                                  os.path.join(path, files[1]),
                                                  os.path.join(path, files[2]),
                                                  format=format,
                                                  delimiter=delimiter,
                                                  use_population=use_population,
                                                  remove_function=remove_function,
                                                  remove_outcomes=remove_outcomes,
                                                  add_extra_dis_gene_edges=add_extra_dis_gene_edges,
                                                  test_infer_path=test_infer_path)

    def _load_entity_relation(self, path, test_infer_path=None):
        self.entity2id = read_dictionary(os.path.join(path, "entities.dict"), id_lookup=False)
        self.n_entities = len(self.entity2id)

        self.relation2id = read_dictionary(os.path.join(path, "relations.dict"), id_lookup=False)
        self.n_relations = len(self.relation2id)

        print("Adding infer nodes and edges")
        if test_infer_path is not None:
            self.extra_entity2id = read_dictionary(os.path.join(test_infer_path, "new_entities.dict"), id_lookup=False)
            self.entity2id.update(self.extra_entity2id)
            self.n_entities = len(self.entity2id)

    def read_entity(self, entity_path):
        return self.entity2id, self.n_entities

    def read_relation(self, relation_path):
        return self.relation2id, self.n_relations

    @property
    def emap_fname(self):
        return 'entities.dict'

    @property
    def rmap_fname(self):
        return 'relations.dict'


class BaseDataset:
    def __init__(self, *, data_path: str, dataset: str, data_files: List[str],
                 bidirectional: bool, node_feats_df_path: str, use_population: bool, remove_function: bool,
                 remove_outcomes: bool, dont_include_bert_embs_trial: bool = False,
                 dont_include_trial_node_features: bool = False,
                 add_extra_dis_gene_edges: bool = False,
                 only_trial_embeddings: bool = False,
                 test_infer_path: str = None):
        file_format = 'hrt'
        delimiter = '\t'
        self.data_path = data_path
        self.dataset = KGDatasetUDDRaw(data_path,
                                       dataset,
                                       delimiter,
                                       data_files,
                                       file_format,
                                       use_population=use_population,
                                       remove_function=remove_function,
                                       remove_outcomes=remove_outcomes,
                                       add_extra_dis_gene_edges=add_extra_dis_gene_edges,
                                       test_infer_path=test_infer_path)
        self.bidirectional = bidirectional
        self._load_node_features(node_feats_df_path, dont_include_trial_node_features,
                                 dont_include_bert_embs_trial, only_trial_embeddings)

    def _load_node_features(self, node_feats_df_path, dont_include_trial_node_features,
                            dont_include_bert_embs_trial, only_trial_embeddings):
        dataset = self.dataset
        if node_feats_df_path is not None:
            node_feats_df = pd.read_pickle(node_feats_df_path)
            if dont_include_bert_embs_trial:
                mask = node_feats_df.etype == 'TRIAL_ARM'
                node_feats_df['emb'][mask] = node_feats_df['emb'][mask].map(lambda x: x[768:])
            print(np.stack(node_feats_df[node_feats_df['etype'] == 'TRIAL_ARM']['emb'].values).shape)
            if only_trial_embeddings:
                node_feats_df = node_feats_df[node_feats_df.etype == 'TRIAL_ARM']
            mask = node_feats_df['node_id'].isin(dataset.entity2id.keys())
            print(node_feats_df.etype.value_counts())
            node_feats_df = node_feats_df[mask]
            if dont_include_trial_node_features:
                print(node_feats_df.shape)
                mask = node_feats_df['etype'] != 'TRIAL_ARM'
                node_feats_df = node_feats_df[mask]
                print(node_feats_df.shape)
            node_feats_df['id'] = node_feats_df['node_id'].map(lambda x: dataset.entity2id[x])
            print('\nFinal:')
            print(node_feats_df.etype.value_counts())
            self.node_feats = node_feats_df
        else:
            self.node_feats = None


class Dataset(BaseDataset):
    def __init__(self, *, edge_disjoint: bool, **kwargs):
        super(Dataset, self).__init__(**kwargs)
        self.datasets = {}
        for split in ['train', 'valid', 'test']:
            self.datasets[split] = SplitDataset.get_split_class(split)(self.dataset,
                                                                       bidirectional=self.bidirectional,
                                                                       edge_disjoint=edge_disjoint)


def _get_id(dict, key):
    id = dict.get(key, None)
    if id is None:
        id = len(dict)
        dict[key] = id
    return id


class ClassificationBaseDataset(BaseDataset):
    def __init__(self, trial_id_key: str, trial_feats_df_path: str, split: str, enrollment_filter: int,
                 bert_data: bool, concat_trial_features: bool, no_relations: bool, combine_bert, bert_model_path: str, bert_max_seq_len: int, args,
                 **kwargs):
        if not bert_data:
            super(ClassificationBaseDataset, self).__init__(**kwargs)
        else:
            self.data_path = kwargs['data_path']
        self.trial_id_key = trial_id_key
        self.enrollment_filter = enrollment_filter
        self.split_strategy = split
        self.trial_feats_df_path = trial_feats_df_path
        self.concat_trial_features = concat_trial_features
        self.combine_bert = combine_bert
        self.bert_model_path = bert_model_path
        self.bert_max_seq_len = bert_max_seq_len
        self.no_relations = no_relations
        self.args = args
        assert not concat_trial_features or self.trial_feats_df_path is not None
        # self.df = None

    def _load_graph(self):
        self.graph = InferenceDataset(self.dataset, bidirectional=self.bidirectional, edge_disjoint=False,
                                      no_relations=self.no_relations)

    def _load_trial_features(self):
        if self.trial_feats_df_path is not None:
            trial_feats = pd.read_pickle(self.trial_feats_df_path)

            trial_feats = trial_feats.set_index('nct_id')['embeddings']
            self.df['trial_feats'] = self.df[self.trial_id_key].map(lambda x: trial_feats.loc[x])
            self.trial_features = True
        else:
            self.trial_features = False

    def _get_data_x(self, df):
        if self.args.default_task_name == 'binary_pair_efficacy':
            x = torch.tensor(np.stack([df.x1.values, df.x2.values], axis=-1), dtype=torch.long) #[n_examples, 2]
        else:
            x = torch.tensor(df.x.values, dtype=torch.long) #[n_examples, ]

        if 'trial_feats' in self.df.columns:
            if self.args.default_task_name == 'binary_pair_efficacy':
                trial_feats = torch.tensor(np.stack([np.stack(df.trial_feats1.values), np.stack(df.trial_feats2.values)], axis=-1), dtype=torch.float) #[n_examples, dim, 2]
            else:
                trial_feats = torch.tensor(np.stack(df.trial_feats.values), dtype=torch.float) #[n_examples, dim]

        if self.combine_bert:
            if self.args.default_task_name == 'binary_pair_efficacy':
                input_ids = torch.tensor(np.stack([np.stack(df.input_ids1.values), np.stack(df.input_ids2.values)], axis=-1), dtype=torch.long) #[n_examples, seqlen, 2]
                attention_mask = torch.tensor(np.stack([np.stack(df.attention_mask1.values), np.stack(df.attention_mask2.values)], axis=-1), dtype=torch.long) #[n_examples, seqlen, 2]
                token_type_ids = torch.tensor(np.stack([np.stack(df.token_type_ids1.values), np.stack(df.token_type_ids2.values)], axis=-1), dtype=torch.long) #[n_examples, seqlen, 2]
            else:
                input_ids = torch.tensor(np.stack(df.input_ids.values), dtype=torch.long) #[n_examples, seqlen]
                attention_mask = torch.tensor(np.stack(df.attention_mask.values), dtype=torch.long) #[n_examples, seqlen]
                token_type_ids = torch.tensor(np.stack(df.token_type_ids.values), dtype=torch.long) #[n_examples, seqlen]

        if self.combine_bert: #Added
            return input_ids, attention_mask, token_type_ids, x
        elif self.concat_trial_features:
            return trial_feats, x
        elif self.trial_features:
            return trial_feats
        else:
            return x

    def _make_column_ids(self, data_path): #Important
        node_id_df = pd.read_pickle(os.path.join(data_path, 'unique_arms.pkl')).reset_index()
        kgid_map = {}
        for idx, row in node_id_df.iterrows():
            key = (row['trial'], row['drugs'], row['diseases'])
            assert key not in kgid_map, key
            kgid_map[key] = row['node_id']
        self.df['kgid'] = self.df[[self.trial_id_key, 'drugs', 'diseases']].apply(
            lambda row: kgid_map[(row[self.trial_id_key], row['drugs'], row['diseases'])], axis=1)
        self.df['x'] = self.df['kgid'].map(lambda _id: self.dataset.entity2id[_id])
        print (self.df[['trial', 'kgid']].head(n=5))

        if self.combine_bert:
            #Get text  #Added
            summaries = pd.read_pickle(self.args.bert_text_path).set_index('node_id')
            self.df['arm_text'] = self.df['kgid'].map(lambda _id: summaries.loc[_id]['arm_text'])
            summary = self.df['arm_text']

            #Process text with BERT tokenizer  #Added
            assert self.bert_model_path is not None
            print ('self.bert_model_path (tokenizer)', self.bert_model_path)
            tokenizer = AutoTokenizer.from_pretrained(self.bert_model_path)
            def tokenize(batch):
                return tokenizer(batch['summary'], padding='max_length', truncation=True, max_length=self.bert_max_seq_len)

            hf_dataset = hf_Dataset.from_dict({'summary': summary, 'kgid': self.df['kgid']})
            hf_dataset = hf_dataset.map(tokenize, batched=True, batch_size=1024)
            self.df['input_ids'] = hf_dataset['input_ids'] #nested list of (n_examples, seqlen=512)
            self.df['attention_mask'] = hf_dataset['attention_mask'] #nested list of (n_examples, seqlen=512)
            if 'token_type_ids' in hf_dataset:
                self.df['token_type_ids'] = hf_dataset['token_type_ids'] #nested list of (n_examples, seqlen=512)
            else:
                _n_examples = len(hf_dataset['input_ids'])
                _seqlen     = len(hf_dataset['input_ids'][0])
                self.df['token_type_ids'] = [[0]*_seqlen] * _n_examples

            print ('text data tokenized')
            print (self.df[['trial', 'kgid', 'arm_text', 'input_ids']].head(n=5))


    def _filter(self):
        if self.enrollment_filter > 0:
            print ('enrollment_filter', self.enrollment_filter)
            mask = self.df['enrollment'] > self.enrollment_filter
            print ('len(self.df)', len(self.df))
            self.df = self.df[mask]
            self.df = self.df.reset_index(drop=True)
            print ('len(self.df)', len(self.df))

    def _split(self):
        split_strategy = self.split_strategy
        if split_strategy == 'random':
            n = len(self.df)
            rng = np.random.RandomState(34)
            split = ['train' if i < 0.7 * n else 'val' if 0.7 * n <= i < 0.85 * n else 'test' for i in range(n)]
            split = np.array(split)
            rng.shuffle(split)
            self.df['split'] = split
        elif split_strategy.startswith('temporal-leave-out-one-start-'): #e.g. temporal-leave-out-one-start-2007
            year = int(split_strategy[len('temporal-leave-out-one-start-'):])
            print (f'\nSplit: leaving out {year}')
            assert 2000 <= year < 2020
            self.df['complete_year'] = self.df['primary_end_year'] + (self.df['primary_end_year'] == 0) * \
                                       self.df['end_year']
            TF1 = (self.df.start_year.values >= year) & (self.df.start_year.values < year+1)
            TF2 = (self.df.complete_year.values > 0) & (self.df.complete_year.values < 2020.5)
            train_TF = TF2 & (~TF1)
            val_TF = TF2 & (TF1)
            test_TF = ~(train_TF | val_TF)
            self.df['split'] = [None] * len(self.df)
            self.df['split'][train_TF] = 'train'
            self.df['split'][val_TF] = 'val'
            self.df['split'][test_TF] = 'test'
        elif split_strategy.startswith('temporal-leave-out-since-start-'): #e.g. temporal-leave-out-one-start-2007
            year = int(split_strategy[len('temporal-leave-out-since-start-'):])
            print (f'\nSplit: leaving out {year} and all after')
            assert 2000 <= year < 2020
            self.df['complete_year'] = self.df['primary_end_year'] + (self.df['primary_end_year'] == 0) * \
                                       self.df['end_year']
            TF1 = (self.df.start_year.values >= year)
            TF2 = (self.df.complete_year.values > 0) & (self.df.complete_year.values < 2020.5)
            train_TF = TF2 & (~TF1)
            val_TF = TF2 & (TF1)
            test_TF = ~(train_TF | val_TF)
            self.df['split'] = [None] * len(self.df)
            self.df['split'][train_TF] = 'train'
            self.df['split'][val_TF] = 'val'
            self.df['split'][test_TF] = 'test'
        elif split_strategy.startswith('temporal-leave-out-one-end-'): #e.g. temporal-leave-out-one-end-2007
            year = int(split_strategy[len('temporal-leave-out-one-end-'):])
            print (f'\nSplit: leaving out {year}')
            assert 2000 <= year < 2020
            self.df['complete_year'] = self.df['primary_end_year'] + (self.df['primary_end_year'] == 0) * \
                                       self.df['end_year']
            TF1 = (self.df.complete_year.values >= year) & (self.df.complete_year.values < year+1)
            TF2 = (self.df.complete_year.values > 0) & (self.df.complete_year.values < 2020.5)
            train_TF = TF2 & (~TF1)
            val_TF = TF2 & (TF1)
            test_TF = ~(train_TF | val_TF)
            self.df['split'] = [None] * len(self.df)
            self.df['split'][train_TF] = 'train'
            self.df['split'][val_TF] = 'val'
            self.df['split'][test_TF] = 'test'
        elif split_strategy == 'temporal-start':
            self.df['split'] = self.df.start_year.map(
                lambda x: 'train' if 0 < x <= 2012 else 'val' if 0 < x <= 2014 else 'test')
        elif split_strategy == 'temporal-end':
            self.df['complete_year'] = self.df['primary_end_year'] + (self.df['primary_end_year'] == 0) * \
                                       self.df['end_year']
            # self.df['split'] = self.df.complete_year.map(
            #     lambda x: 'train' if 0 < x <= 2014.5 else 'val' if 0 < x <= 2016.3 else 'test')
            years = self.df.sort_values(by=['complete_year'])['complete_year'].tolist()
            train_end_year = years[int(len(years) * 0.7)]
            val_end_year = years[int(len(years) * 0.85)]
            print ('train_end_year', train_end_year, 'val_end_year', val_end_year, 'latest_year', years[-1])
            self.df['split'] = self.df.complete_year.map(
                  lambda x: 'train' if 0 <= x <= train_end_year else 'val' if 0 <= x <= val_end_year else 'test')
        elif split_strategy == 'temporal-end-test':
            self.df['complete_year'] = self.df['primary_end_year'] + (self.df['primary_end_year'] == 0) * \
                                       self.df['end_year']
            # self.df['split'] = self.df.complete_year.map(
            #     lambda x: 'train' if 0 < x <= 2016.3 else 'test')
            years = self.df.sort_values(by=['complete_year'])['complete_year'].tolist()
            train_end_year = years[int(len(years) * 0.7)]
            val_end_year = years[int(len(years) * 0.85)]
            print ('train_end_year', train_end_year, 'val_end_year', val_end_year, 'latest_year', years[-1])
            self.df['split'] = self.df.complete_year.map(
                  lambda x: 'train' if 0 <= x <= val_end_year else 'test')
        elif split_strategy == 'withdrawn-drug-test':
            withdrawn_drugs = set(pkl.load(open('../data/graph/withdrawn_drugs.pkl', 'rb')))
            TFs = self.df.drugs.map(lambda x: len([_x for _x in x if _x in withdrawn_drugs]) > 0)
            self.df['split'] = TFs.map(lambda x: 'train' if not x else 'test')
        elif split_strategy == 'drug-disease':
            split_fractions = {
                'train': 0.7,
                'val': 0.15,
                'test': 0.15
            }
            disdrugs_grouped = self.df.groupby(['drugs', 'diseases'])
            num_trials = len(self.df)
            rng = np.random.RandomState(34)
            seed = np.arange(disdrugs_grouped.ngroups)
            rng.shuffle(seed)
            disdrugs = list(disdrugs_grouped.groups.keys())
            split_map = {}
            k = 0
            for idx, (split, fraction) in enumerate(split_fractions.items()):
                cnt = int(num_trials * fraction)
                split_cnt = 0
                while (split_cnt < cnt or idx == len(split_fractions) - 1) and k < len(seed):
                    split_map[disdrugs[seed[k]]] = split
                    split_cnt += len(disdrugs_grouped.get_group(disdrugs[seed[k]]))
                    k += 1

            self.df['split'] = self.df[['drugs', 'diseases']].apply(lambda x: split_map[(x['drugs'], x['diseases'])],
                                                                    axis=1)
        elif split_strategy in ['drug-disease-trial', 'drug-disease-trial-test']:
            split_fractions = {
                'train': 0.7,
                'val': 0.15,
                'test': 0.15
            }
            disdrugs_grouped = self.df.groupby(['drugs', 'diseases'])
            num_trials = len(self.df)
            rng = np.random.RandomState(34)
            disdrugs = list(disdrugs_grouped.groups.keys())
            rng.shuffle(disdrugs)
            #
            disdrug2trials = {}
            trial2disdrugs = defaultdict(set)
            for disdrug in disdrugs:
                trials = set(disdrugs_grouped.get_group(disdrug)['trial'])
                disdrug2trials[disdrug] = trials
                for trial in trials:
                    trial2disdrugs[trial].add(disdrug)
            #
            def add_disdrug(split_map, split, split_cnt, disdrug, disdrugs, disdrug2trials, trial2disdrugs):
                try:
                    disdrugs.remove(disdrug)
                except:
                    return
                split_map[disdrug] = split; split_cnt[0] += len(disdrugs_grouped.get_group(disdrug))
                trials_to_include = disdrug2trials[disdrug]
                while len(trials_to_include) > 0:
                    _trial = trials_to_include.pop()
                    for _disdrug in trial2disdrugs[_trial]:
                        add_disdrug(split_map, split, split_cnt, _disdrug, disdrugs, disdrug2trials, trial2disdrugs)
            #
            split_map = {}
            for idx, (split, fraction) in enumerate(split_fractions.items()):
                cnt = int(num_trials * fraction)
                split_cnt = [0]
                while (split_cnt[0] < cnt or idx == len(split_fractions) - 1) and 0 < len(disdrugs):
                    disdrug = disdrugs[0]
                    add_disdrug(split_map, split, split_cnt, disdrug, disdrugs, disdrug2trials, trial2disdrugs)
            #
            self.df['split'] = self.df[['drugs', 'diseases']].apply(lambda x: split_map[(x['drugs'], x['diseases'])], axis=1)
            if split_strategy == 'drug-disease-trial-test':
                self.df['split'] = self.df['split'].map(lambda x: 'train' if x=='val' else x)
        elif split_strategy in ['drug-trial', 'drug-trial-test']:
            split_fractions = {
                'train': 0.7,
                'val': 0.15,
                'test': 0.15
            }
            drugs_grouped = self.df.groupby(['drugs'])
            num_trials = len(self.df)
            rng = np.random.RandomState(34)
            drugs = list(drugs_grouped.groups.keys())
            rng.shuffle(drugs)
            #
            drug2trials = {}
            trial2drugs = defaultdict(set)
            for drug in drugs:
                trials = set(drugs_grouped.get_group(drug)['trial'])
                drug2trials[drug] = trials
                for trial in trials:
                    trial2drugs[trial].add(drug)
            #
            def add_drug(split_map, split, split_cnt, drug, drugs, drug2trials, trial2drugs):
                try:
                    drugs.remove(drug)
                except:
                    return
                split_map[drug] = split; split_cnt[0] += len(drugs_grouped.get_group(drug))
                trials_to_include = drug2trials[drug]
                while len(trials_to_include) > 0:
                    _trial = trials_to_include.pop()
                    for _drug in trial2drugs[_trial]:
                        add_drug(split_map, split, split_cnt, _drug, drugs, drug2trials, trial2drugs)
            #
            split_map = {}
            for idx, (split, fraction) in enumerate(split_fractions.items()):
                cnt = int(num_trials * fraction)
                split_cnt = [0]
                while (split_cnt[0] < cnt or idx == len(split_fractions) - 1) and 0 < len(drugs):
                    drug = drugs[0]
                    add_drug(split_map, split, split_cnt, drug, drugs, drug2trials, trial2drugs)
            #
            self.df['split'] = self.df[['drugs']].apply(lambda x: split_map[(x['drugs'])], axis=1)
            if split_strategy == 'drug-trial-test':
                self.df['split'] = self.df['split'].map(lambda x: 'train' if x=='val' else x)
        elif split_strategy in ['disease-trial', 'disease-trial-test']:
            split_fractions = {
                'train': 0.7,
                'val': 0.15,
                'test': 0.15
            }
            diss_grouped = self.df.groupby(['diseases'])
            num_trials = len(self.df)
            rng = np.random.RandomState(34)
            diss = list(diss_grouped.groups.keys())
            rng.shuffle(diss)
            #
            dis2trials = {}
            trial2diss = defaultdict(set)
            for dis in diss:
                trials = set(diss_grouped.get_group(dis)['trial'])
                dis2trials[dis] = trials
                for trial in trials:
                    trial2diss[trial].add(dis)
            #
            def add_dis(split_map, split, split_cnt, dis, diss, dis2trials, trial2diss):
                try:
                    diss.remove(dis)
                except:
                    return
                split_map[dis] = split; split_cnt[0] += len(diss_grouped.get_group(dis))
                trials_to_include = dis2trials[dis]
                while len(trials_to_include) > 0:
                    _trial = trials_to_include.pop()
                    for _dis in trial2diss[_trial]:
                        add_dis(split_map, split, split_cnt, _dis, dis, dis2trials, trial2diss)
            #
            split_map = {}
            for idx, (split, fraction) in enumerate(split_fractions.items()):
                cnt = int(num_trials * fraction)
                split_cnt = [0]
                while (split_cnt[0] < cnt or idx == len(split_fractions) - 1) and 0 < len(diss):
                    dis = diss[0]
                    add_dis(split_map, split, split_cnt, dis, diss, dis2trials, trial2diss)
            #
            self.df['split'] = self.df[['diseases']].apply(lambda x: split_map[(x['diseases'])], axis=1)
            if split_strategy == 'disease-trial-test':
                self.df['split'] = self.df['split'].map(lambda x: 'train' if x=='val' else x)
        elif split_strategy == 'drug-disease-train-on-valid':
            self.df['split'] = self.df.split.map(lambda x: 'train' if x == 'val' else x)
        elif split_strategy == 'drug':
            drug_split_df = pd.read_pickle('../data/drug_split.pkl').set_index('idx')
            self.df['split'] = self.df.index.map(lambda x: drug_split_df.loc[x]['split_drug'])
        elif split_strategy == 'num-arms':
            counts = self.df.trial.value_counts()
            trial2split = {}
            rng = np.random.RandomState(34)
            for trial in counts.index:
                if counts.loc[trial] <= 1 or rng.rand() < 0.25:
                    trial2split[trial] = 'train'
                elif rng.rand() < 0.5:
                    trial2split[trial] = 'val'
                else:
                    trial2split[trial] = 'test'
            self.df['split'] = self.df.trial.map(lambda x: trial2split[x])
            self.df.split.value_counts()
        elif split_strategy == 'num-arms-exact':
            counts = self.df.trial.value_counts()
            trial2split = {}
            rng = np.random.RandomState(34)
            mul_trials = counts[(counts > 1)].index.tolist()
            mul_trials = rng.permutation(mul_trials)
            for trial in counts[(counts == 1)].index:
                trial2split[trial] = 'train'
            for trial in mul_trials[:len(mul_trials) // 2]:
                trial2split[trial] = 'val'
            for trial in mul_trials[len(mul_trials) // 2:]:
                trial2split[trial] = 'test'
            self.df['split'] = self.df.trial.map(lambda x: trial2split[x])
        else:
            raise RuntimeError("Unexpected split type: ", split_strategy)
        print("Split strategy: ", split_strategy)
        print(self.df.split.value_counts())
        assert not self.df.split.isna().any()

    def _split_for_pair_efficacy(self):
        assert self.split_strategy == 'trial'
        efficacy_df = self.efficacy_df
        split_fractions = {
            'train': 0.7,
            'val': 0.15,
            'test': 0.15
        }
        if True:
            nct_grouped = efficacy_df.groupby(['nct_id'])
            num_examples = len(efficacy_df)
            print ('efficacy split: setting seed=', self.args.random_seed + 33)
            rng = np.random.RandomState(self.args.random_seed + 33)
            seed = np.arange(nct_grouped.ngroups)
            rng.shuffle(seed)
            ncts = list(nct_grouped.groups.keys())
            split_map = {}
            k = 0
            for idx, (split, fraction) in enumerate(split_fractions.items()):
                cnt = int(num_examples * fraction)
                split_cnt = 0
                while (split_cnt < cnt or idx == len(split_fractions) - 1) and k < len(seed):
                    split_map[ncts[seed[k]]] = split
                    split_cnt += len(nct_grouped.get_group(ncts[seed[k]]))
                    k += 1
        # if True:
        #     assert len(efficacy_df) % 2 == 0
        #     efficacy_df_hf1 = efficacy_df.head(len(efficacy_df)//2)
        #     efficacy_df_hf2 = efficacy_df.tail(len(efficacy_df)//2)
        #     assert efficacy_df_hf1.nct_id.tolist() == efficacy_df_hf2.nct_id.tolist()
        #     assert efficacy_df_hf1.pair.tolist() == [(p[1], p[0]) for p in efficacy_df_hf2.pair.tolist()]
        #     nct_grouped = efficacy_df_hf1.groupby(['nct_id'])
        #     num_examples = len(efficacy_df_hf1)
        #     rng = np.random.RandomState(self.args.random_seed + 33)
        #     seed = np.arange(nct_grouped.ngroups)
        #     rng.shuffle(seed)
        #     ncts = list(nct_grouped.groups.keys())
        #     split_map_retry = {}
        #     k = 0
        #     for idx, (split, fraction) in enumerate(split_fractions.items()):
        #         cnt = int(num_examples * fraction)
        #         split_cnt = 0
        #         while (split_cnt < cnt or idx == len(split_fractions) - 1) and k < len(seed):
        #             split_map_retry[ncts[seed[k]]] = split
        #             split_cnt += len(nct_grouped.get_group(ncts[seed[k]]))
        #             k += 1
        # assert split_map == split_map_retry

        if self.args.subsample_train > 0:
            train_ncts = sorted([k for k in split_map if split_map[k] == 'train'])
            rng.shuffle(train_ncts)
            count = 0
            for nct in train_ncts:
                count += len(nct_grouped.get_group(nct))
                if count < self.args.subsample_train:
                    pass
                else:
                    split_map[nct] = 'none'
        efficacy_df['split'] = efficacy_df['nct_id'].map(lambda x: split_map[x])

        train_pairs = efficacy_df[efficacy_df['split']=='train'].pair.tolist()
        train_pairs_set = set(train_pairs)
        for pair in train_pairs:
            assert (pair[1], pair[0]) in train_pairs_set
        print('efficacy split sanity check done.')

        self.efficacy_df = efficacy_df
        print("Split strategy: ", self.split_strategy)
        print(self.efficacy_df.split.value_counts())
        assert not self.efficacy_df.split.isna().any()
        print()


    def _create_datasets(self, task_ys, sample_weight_masks):
        print ('_create_datasets...')
        split_strategy = self.split_strategy
        self.datasets = {}
        # splits = ['train', 'val', 'test']
        # if split == 'drug-disease-train-on-valid':
        #     splits = ['train', 'test']
        # self.datasets = {
        #     split: TensorDataset(self._get_data_x(df),
        #                          torch.tensor(np.stack(df.ae_one_hot.values), dtype=torch.float))
        #     for split, df in
        #     map(lambda x: (x if x != 'val' else 'valid', self.df[self.df.split == x]), splits)}
        # if split == 'drug-disease-train-on-valid':
        #     self.datasets['valid'] = self.datasets['test']

        if self.args.default_task_name == 'binary_pair_efficacy':
            main_df = self.efficacy_df
        else:
            main_df = self.df
        print ('len(main_df)', len(main_df))
        x = self._get_data_x(main_df)

        print ('len(x)', len(x))
        splits = ['train', 'valid', 'test']
        if split_strategy in ['drug-disease-train-on-valid', 'temporal-end-test', 'withdrawn-drug-test', 'drug-disease-trial-test', 'drug-trial-test', 'disease-trial-test', 'trial-test']:
            splits = ['train', 'test']
        for split_ in splits:
            split_name = split_ if split_ != 'valid' else 'val'
            mask = main_df.split == split_name
            # print ('len(mask)', len(mask))
            # print ('x', x)
            # print ('mask', mask)
            tensors = []
            if self.combine_bert: #Added
                tensors.extend((x[0][mask], x[1][mask], x[2][mask], x[3][mask]))
            elif not self.concat_trial_features:
                tensors.append(x[mask])
            else:
                tensors.extend((x[0][mask], x[1][mask]))
            tensors.extend([task_y[mask] for task_y in task_ys])
            tensors.extend([sample_weight_mask[mask] for sample_weight_mask in sample_weight_masks])
            print ('split', split_, len(tensors[0]))
            if split_=='train' and len(tensors[0]) < 512:
                repeat = 512//len(tensors[0]) +1
                tensors = [torch.cat([t.clone() for _ in range(repeat)], dim=0) for t in tensors]
                print ('--> split', split_, len(tensors[0]))
            self.datasets[split_] = TensorDataset(*tensors)
        if split_strategy in ['drug-disease-train-on-valid', 'temporal-end-test', 'withdrawn-drug-test', 'drug-disease-trial-test', 'drug-trial-test', 'disease-trial-test', 'trial-test']:
            self.datasets['valid'] = self.datasets['test']
        print ('_create_datasets done.')

    @staticmethod
    def _balanced_sample_weights(y, example_mask, max_ratio=10):
        y_true = y.cpu().numpy() #[n_examples, num_subtasks]
        n = example_mask.sum(axis=0)
        mask = np.zeros_like(y_true).astype(np.bool)
        mask[y_true == 1] = True
        num_pos = y_true.sum(axis=0).astype(np.int)
        num_negs = n - num_pos
        sampled_num_negs = np.minimum(max_ratio * num_pos, num_negs)
        for i in range(y.shape[1]):
            sampled_negs = np.random.choice(np.arange(n)[(y_true[:, i] == 0) & (example_mask[:, i])],
                                            sampled_num_negs[i])
            # mask[y_true == 0] = np.random.randn(y_true.size(0) - num_pos) > 0.5
            mask[sampled_negs, i] = True
        print(mask.sum(axis=0))
        return torch.tensor(mask, dtype=torch.bool) #[n_examples, num_subtasks]

#not used
class ClassificationDataset(ClassificationBaseDataset):
    def __init__(self, *, clf_df_path: str, is_binary_classification: str,
                 label_column_name: str, binary_threshold: float,
                 single_ae: str = None, bert_data=False, balanced_sample: bool,
                 **kwargs):
        if type(is_binary_classification) == bool and is_binary_classification:
            is_binary_classification = 'total-based'

        if is_binary_classification == 'total-based':
            self.df = pd.read_pickle(os.path.join(clf_df_path, 'binary-data.pkl'))
        else:
            self.df = pd.read_pickle(os.path.join(clf_df_path, 'data.pkl'))

        super(ClassificationDataset, self).__init__(trial_id_key='trial_id', bert_data=bert_data, **kwargs)
        if not bert_data:
            self._load_graph()
            self._load_trial_features()

        nct_info = pd.read_pickle(os.path.join(self.data_path, 'nctinfo.pkl'))
        self.df = self.df.merge(nct_info.rename(columns={'nct_id': 'trial_id'}), on='trial_id')

        self.ae2id = {}

        self.all_columns = ['drugs', 'diseases']

        for column in self.all_columns + [label_column_name, 'trial_id']:
            assert column in self.df.columns, f'{column} not in dataset columns'

        if not bert_data:
            self._make_column_ids(self.data_path)

        if is_binary_classification == 'total-based':
            self._make_binary_data(binary_threshold)
        elif is_binary_classification == 'any-ae-based':
            self._make_binary_any(label_column_name)
        else:
            self._make_ids(label_column_name, single_ae)

        self._filter()
        self._split()
        if not bert_data:
            task_y = torch.tensor(np.stack(self.df.ae_one_hot.values), dtype=torch.float)

            if balanced_sample:
                sample_weight_mask = self._balanced_sample_weights(task_y, np.ones(task_y.size(), dtype=np.bool))
            else:
                sample_weight_mask = torch.ones_like(task_y, dtype=torch.bool)

            self._create_datasets([task_y], [sample_weight_mask])

    def _make_binary_data(self, threshold):
        self.df['is_unsafe'] = self.df['ae_freq_percentage'] > threshold
        n_aes = 1
        print(f"N aes: {n_aes}")
        self.df['ae_one_hot'] = self.df['is_unsafe'].map(lambda x: [1] if x else [0])
        self.num_tasks = n_aes

    def _make_binary_any(self, label_column_name):
        ae_set = set()
        for aes in self.df[label_column_name]:
            ae_set.update(aes)
        for ae in sorted(ae_set):
            _get_id(self.ae2id, ae)

        self.df['ae_ids'] = self.df[label_column_name].map(lambda x: [self.ae2id[i] for i in x])

        n_aes = len(self.ae2id)
        print(f"N aes: {n_aes}")

        def one_hot(ids):
            if len(ids) > 0:
                return [1]
            return [0]

        self.df['ae_one_hot'] = self.df['ae_ids'].map(one_hot)
        self.num_tasks = 1

    def _make_ids(self, label_column_name, single_ae):
        ae_set = set()
        for aes in self.df[label_column_name]:
            ae_set.update(aes)
        for ae in sorted(ae_set):
            _get_id(self.ae2id, ae)

        self.df['ae_ids'] = self.df[label_column_name].map(lambda x: [self.ae2id[i] for i in x])
        n_aes = len(self.ae2id)
        print(f"N aes: {n_aes}")

        def one_hot(ids):
            v = np.zeros(n_aes)
            v[ids] = 1
            if single_ae is not None:
                id_ = self.ae2id[single_ae]
                v = v[id_:id_ + 1]
            return v

        self.df['ae_one_hot'] = self.df['ae_ids'].map(one_hot)
        self.num_tasks = n_aes
        if single_ae is not None:
            self.num_tasks = 1

    @property
    def pos_weights(self):
        y = np.stack(self.df[self.df.split == 'train'].ae_one_hot.values)
        n = len(y)
        pos_count = np.sum(y, axis=0)
        pos_weight = np.sqrt((n - pos_count) / pos_count)
        return pos_weight


class MultiTaskDataset(ClassificationBaseDataset):
    def __init__(self, clf_df_path: str, tasks, bert_data: bool = False, balanced_sample: bool = False, **kwargs):
        super(MultiTaskDataset, self).__init__(trial_id_key='trial', bert_data=bert_data, **kwargs)

        self.df = pd.read_pickle(os.path.join(clf_df_path, "data.pkl")) #Important
        if self.args.default_task_name == 'binary_pair_efficacy':
            self.efficacy_df = pd.read_pickle(os.path.join(clf_df_path, "efficacy_survival_pairs.pickle"))
        if self.args.default_task_name == 'with_or_without_results':
            self.df = pd.read_pickle(os.path.join(clf_df_path, "unique_arms.pkl"))
        self.ae2id, self.aename2kgid = read_dictionary(os.path.join(clf_df_path, 'aes.dict'), id_lookup=False,
                                                       name_lookup=True)

        if not bert_data:
            self._load_graph()
            self._load_trial_features()

        nct_info = pd.read_pickle(os.path.join(self.data_path, 'nctinfo.pkl'))
        self.df = self.df.merge(nct_info.rename(columns={'nct_id': 'trial'}), on='trial') #Important
        if not bert_data:
            self._make_column_ids(self.data_path)

        task_ys = []
        sample_weight_masks = []
        for task in tasks:
            task_y = self._make_task(task['name'], task)
            if type(task_y) == tuple:
                raise NotImplementedError
                task_y, example_mask = task_y
            elif type(task_y) == list:
                task_y, idx2aename = task_y
                task['idx2aename'] = idx2aename
                example_mask = np.ones(task_y.size(), dtype=np.bool)
            else:
                example_mask = np.ones(task_y.size(), dtype=np.bool)
            task_ys.append(task_y)
            task['num_subtasks'] = task_y.shape[1]
            print(f"Task: {task['name']}, Subtasks: {task['num_subtasks']}")
            # print(task_y.shape, (task_y * example_mask).sum(dim=0))
            print('task_y.shape', task_y.shape) #tensor[n_examples, num_subtasks]
            if balanced_sample:
                sample_weight_mask = self._balanced_sample_weights(task_y, example_mask)
            else:
                sample_weight_mask = torch.tensor(example_mask, dtype=torch.bool) #tensor[n_examples, num_subtasks]
            sample_weight_masks.append(sample_weight_mask)
        task_ys = torch.stack(task_ys).transpose(0,1).tolist() #[n_examples, tasks, num_subtasks]
        sample_weight_masks = torch.stack(sample_weight_masks).transpose(0,1).tolist() #[n_examples, tasks, num_subtasks]

        if self.args.default_task_name == 'binary_pair_efficacy':
            self.efficacy_df['task_ys'] = task_ys
            self.efficacy_df['sample_weight_masks'] = sample_weight_masks
            self._split_for_pair_efficacy()
            self.task_ys = list(torch.unbind(torch.tensor(list(self.efficacy_df['task_ys'])).transpose(0,1))) #list of tensor[n_examples, num_subtasks]
            print ('len(self.task_ys[0])', len(self.task_ys[0]))
            self.sample_weight_masks = list(torch.unbind(torch.tensor(list(self.efficacy_df['sample_weight_masks'])).transpose(0,1))) #list of tensor[n_examples, num_subtasks]
        else:
            self.df['task_ys'] = task_ys
            self.df['sample_weight_masks'] = sample_weight_masks
            self._filter()
            self._split()
            self.task_ys = list(torch.unbind(torch.tensor(list(self.df['task_ys'])).transpose(0,1))) #list of tensor[n_examples, num_subtasks]
            print ('len(self.task_ys[0])', len(self.task_ys[0]))
            self.sample_weight_masks = list(torch.unbind(torch.tensor(list(self.df['sample_weight_masks'])).transpose(0,1))) #list of tensor[n_examples, num_subtasks]

        # self._filter()
        # self._split()
        if not bert_data:
            self._create_datasets(self.task_ys, self.sample_weight_masks)

        self.tasks = tasks

    def _make_task(self, task, task_params):
        total_ae_idx = self.ae2id[self.aename2kgid['total']]
        print(f"Total AE idx: {total_ae_idx}")
        if task == 'binary':
            freqs = np.stack(self.df['serious_vec_ae_freq_percentage'].values)[:, total_ae_idx]
            labels = freqs > task_params['threshold']
            print("Labels: ", np.unique(labels, return_counts=True))
            return torch.tensor(labels[:, np.newaxis], dtype=torch.float)
        elif task == 'binary_or':
            ors = np.stack(self.df['serious_vec_odds_ratio'].values)[:, total_ae_idx] #[n_examples, ]
            labels = ors > task_params['threshold'] #[n_examples, ]
            print("Labels: ", np.unique(labels, return_counts=True))
            return torch.tensor(labels[:, np.newaxis], dtype=torch.float)
        elif task == 'with_or_without_results':
            return torch.tensor(self.df.has_results.tolist(), dtype=torch.float).unsqueeze(1) #[n_examples, 1]
        elif task == 'binary_pair_efficacy':
            print ('building efficacy data...')
            _tmp_df = self.df.set_index('kgid')
            _kgids = set(self.df['kgid'].values)
            efficacy_df = self.efficacy_df
            efficacy_df_cp = self.efficacy_df.copy()
            efficacy_df_cp['pair'] = efficacy_df['pair'].map(lambda x: (x[1], x[0]))
            efficacy_df_total = pd.concat([efficacy_df, efficacy_df_cp])
            efficacy_df_total = efficacy_df_total[efficacy_df_total['pair'].map(lambda x: (x[0] in _kgids) and (x[1] in _kgids))].reset_index(drop=True)
            efficacy_df_total['kgid1'] = efficacy_df_total['pair'].map(lambda x: x[0])
            efficacy_df_total['kgid2'] = efficacy_df_total['pair'].map(lambda x: x[1])
            labels = (efficacy_df_total['kgid2'] == efficacy_df_total['positive_example']).map(lambda x: int(x)).values #arm2 is positive => label is 1
            labels = torch.tensor(labels) #[n_examples]
            for pair, pos, label in zip(efficacy_df_total['pair'], efficacy_df_total['positive_example'], labels):
                assert pair[label] == pos

            efficacy_df_total['x1'] = efficacy_df_total['kgid1'].map(lambda kgid: _tmp_df.loc[kgid]['x'])
            efficacy_df_total['x2'] = efficacy_df_total['kgid2'].map(lambda kgid: _tmp_df.loc[kgid]['x'])
            if 'input_ids' in _tmp_df.columns:
                efficacy_df_total['input_ids1'] = efficacy_df_total['kgid1'].map(lambda kgid: _tmp_df.loc[kgid]['input_ids'])
                efficacy_df_total['input_ids2'] = efficacy_df_total['kgid2'].map(lambda kgid: _tmp_df.loc[kgid]['input_ids'])
                efficacy_df_total['attention_mask1'] = efficacy_df_total['kgid1'].map(lambda kgid: _tmp_df.loc[kgid]['attention_mask'])
                efficacy_df_total['attention_mask2'] = efficacy_df_total['kgid2'].map(lambda kgid: _tmp_df.loc[kgid]['attention_mask'])
                efficacy_df_total['token_type_ids1'] = efficacy_df_total['kgid1'].map(lambda kgid: _tmp_df.loc[kgid]['token_type_ids'])
                efficacy_df_total['token_type_ids2'] = efficacy_df_total['kgid2'].map(lambda kgid: _tmp_df.loc[kgid]['token_type_ids'])
            if 'trial_feats' in _tmp_df.columns:
                efficacy_df_total['trial_feats1'] = efficacy_df_total['kgid1'].map(lambda kgid: _tmp_df.loc[kgid]['trial_feats'])
                efficacy_df_total['trial_feats2'] = efficacy_df_total['kgid2'].map(lambda kgid: _tmp_df.loc[kgid]['trial_feats'])
            self.efficacy_df = efficacy_df_total
            return torch.tensor(labels, dtype=torch.float).unsqueeze(1) #[n_examples, 1]

        elif task in ['ae_clf', 'ae_regr', 'ae_clf_freq']:
            serious = np.stack(self.df['serious_vec_ae_freq_percentage'].values)
            other = np.stack(self.df['other_vec_ae_freq_percentage'].values)

            if task_params['merge_strategy'] == 'max':
                freqs = np.maximum(serious, other)
            elif task_params['merge_strategy'] == 'serious':
                freqs = serious
            elif task_params['merge_strategy'] == 'mix':
                freqs = np.maximum(serious * 5, other)
            freqs = np.delete(freqs, [total_ae_idx], axis=1)

            if task == 'ae_clf':
                if task_params['merge_strategy'] == 'mix':
                    labels = np.logical_or(serious > 1, other > 5)
                    labels = np.delete(labels, [total_ae_idx], axis=1)
                else:
                    # labels = freqs/100
                    labels = freqs > task_params['threshold']
                cnt = labels.sum(axis=0)
                mask = np.ones_like(cnt, dtype=np.bool)
                mask[cnt < task_params['example_threshold']] = False
                return torch.tensor(labels[:, mask], dtype=torch.float)
            elif task == 'ae_clf_freq':
                labels = freqs > task_params['threshold']
                cnt = labels.sum(axis=0)
                mask = np.ones_like(cnt, dtype=np.bool)
                mask[cnt < task_params['example_threshold']] = False
                return torch.tensor(freqs[:, mask] / 100, dtype=torch.float)
            else:
                return torch.tensor(freqs, dtype=torch.float)
        elif task in ['ae_clf_or', 'ae_clf_or_l']:
            kgid2name = {v:k for k,v in self.aename2kgid.items()}
            aeidx2name = {v: k for k, v in self.ae2id.items()}
            names = [aeidx2name[i] for i in range(len(aeidx2name))]
            names = np.delete(names, [total_ae_idx], axis=0)
            #
            serious = np.stack(self.df['serious_vec_odds_ratio'].values) #[n_examples, n_AEs]
            other = np.stack(self.df['other_vec_odds_ratio'].values) #[n_examples, n_AEs]
            ors = np.maximum(serious, other) #[n_examples, n_AEs]
            ors = np.delete(ors, [total_ae_idx], axis=1)
            print ('*****[NOTE] Using OR threshold', task_params['threshold'], '*****')
            labels = ors > task_params['threshold'] #[n_examples, n_AEs]
            cnt = labels.sum(axis=0) - (ors == -1).sum(axis=0) #[n_AEs, ]
            mask = np.ones_like(cnt, dtype=np.bool) #[n_AEs, ]
            mask[cnt < task_params['example_threshold']] = False #[n_AEs, ]
            print("Number of dropped examples: ", (ors == -1)[:, mask].sum())
            #
            idx2aename = names[mask] #[final_AEs, ]
            task_y =  torch.tensor(labels[:, mask], dtype=torch.float) #[n_examples, final_AEs]
            return [task_y, idx2aename]
        else:
            raise RuntimeError(f"Unknown task type: {task}")

    @property
    def pos_weights(self):
        task_y = self.task_ys[0]
        y = task_y[self.df.split == 'train'].numpy()
        n = len(y)
        pos_count = np.sum(y, axis=0)
        pos_weight = np.sqrt((n - pos_count) / pos_count)
        return pos_weight


class SplitDataset:
    def __init__(self, dataset: KGDatasetUDDRaw, *, bidirectional: bool, edge_disjoint: bool, no_relations: bool=False):
        self.bidirectional = bidirectional
        self.edge_disjoint = edge_disjoint
        self.pop_relations = ['source-CLINICAL_TRIAL:rel-name-eligibility-exclusion',
                              'source-CLINICAL_TRIAL:rel-name-eligibility-inclusion']
        self.no_relations = no_relations
        # self.pop_rel_ids = [dataset.relation2id[x] for x in self.pop_relations]
        self._init_data(dataset)

    def _init_data(self, dataset):
        raise NotImplementedError

    def _construct_graph(self, edges, dataset):
        src, etype_id, dst = edges

        if self.bidirectional:
            edge_index = torch.tensor([np.concatenate([src, dst]), np.concatenate([dst, src])], dtype=torch.long)
            data = Data(edge_index=edge_index)
            data.edge_type = torch.tensor(np.concatenate([etype_id, etype_id + dataset.n_relations]), dtype=torch.long)
            data.num_relations = dataset.n_relations * 2
            # pop_rel_ids = self.pop_rel_ids + [x + dataset.n_relations for x in self.pop_rel_ids]
            # data.pop_edges_mask = torch.tensor(np.isin(data.edge_type, pop_rel_ids), dtype=torch.bool)
            data.pop_edges_mask = None
        else:
            edge_index = torch.tensor([dst, src], dtype=torch.long)
            data = Data(edge_index=edge_index)
            data.edge_type = torch.tensor(etype_id, dtype=torch.long)
            data.num_relations = dataset.n_relations

            # data.pop_edges_mask = torch.tensor(np.isin(data.edge_type, self.pop_rel_ids), dtype=torch.bool)
            data.pop_edges_mask = None

        if self.no_relations:
            data.edge_index = data.edge_index.T.unique(dim=0).T
            print(data.edge_index.size(), edge_index.size())
            data.edge_type = torch.zeros_like(data.edge_index[0])
            data.num_relations = 1

        data.num_nodes = dataset.n_entities
        data.x = torch.arange(dataset.n_entities, dtype=torch.long)
        self.data = data

    @staticmethod
    def get_split_class(split):
        if split == 'train':
            return TrainDataset
        elif split == 'valid':
            return ValidDataset
        elif split == 'test':
            return TestDataset


class InferenceDataset(SplitDataset):
    def _init_data(self, dataset):
        if dataset.infer_test is None:
            graph_edges = self._merge_edges([dataset.train, dataset.valid, dataset.test])
        else:
            graph_edges = self._merge_edges([dataset.train, dataset.valid, dataset.test, dataset.infer_test])
        self.triples = []
        self._construct_graph(graph_edges, dataset)

    @staticmethod
    def _merge_edges(edges):
        h = np.concatenate([x[0] for x in edges])
        r = np.concatenate([x[1] for x in edges])
        t = np.concatenate([x[2] for x in edges])
        return h, r, t


class TrainDataset(SplitDataset):
    @staticmethod
    def _divide_edges(edges):
        h, r, t = edges
        mask = np.zeros(len(r), dtype=np.bool)
        for i in np.unique(r):
            r_i = np.flatnonzero(r == i)
            mask[np.random.choice(r_i, int(len(r_i) * 0.3))] = True
        print("Number of train triplets:", mask.sum())
        return (h[mask], r[mask], t[mask]), (h[~mask], r[~mask], t[~mask])

    def _init_data(self, dataset):
        if not self.edge_disjoint:
            triples, graph_edges = dataset.train, dataset.train
        else:
            triples, graph_edges = self._divide_edges(dataset.train)
        self.triples = triples
        self._construct_graph(graph_edges, dataset)
        # self.weights = sklearn.utils.class_weight.compute_class_weight('balanced', y=self.data.edge_type)


class ValidDataset(SplitDataset):
    def _init_data(self, dataset):
        triples, graph_edges = dataset.valid, dataset.train
        self.triples = triples
        self._construct_graph(graph_edges, dataset)


class TestDataset(SplitDataset):
    def _init_data(self, dataset):
        triples, graph_edges = dataset.test, dataset.train
        self.triples = triples
        self._construct_graph(graph_edges, dataset)
