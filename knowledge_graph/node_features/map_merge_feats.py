import os

import numpy as np
import pandas as pd
from tqdm import tqdm

from ..kg import EntityType


class FeatureMergerAndMapper:
    def __init__(self, feature_base_path):
        self.feature_base_path = feature_base_path
        self._load_features()

    def _load_features(self):
        basepath = self.feature_base_path
        self.trial_embs = pd.read_pickle(os.path.join(basepath, 'trial-embs.pkl')).set_index('nct_id')[
            'embs']
        self.trial_feats = pd.read_pickle(os.path.join(basepath, 'trial-attribute-feats.pkl')).set_index('nct_id')[
            'attribute_feats']
        self.drug_embs = pd.read_pickle(os.path.join(basepath, 'drug-embs.pkl')).set_index('primary_id')['embs']
        protein_df = pd.read_pickle(os.path.join(basepath, 'protein_features.pkl'))
        self.protein_df = protein_df.drop(columns=['kgid', 'GN', 'accession', 'uniprot_id', 'OS']).set_index('entrez_id')
        umls_embs = pd.read_csv(os.path.join(basepath, 'cui2vec_pretrained.csv'))
        umls_embs.rename(columns={'Unnamed: 0': 'Concept ID'}, inplace=True)
        self.umls_embs = umls_embs.set_index('Concept ID')
        self.dis_embs = pd.read_pickle(os.path.join(basepath, 'disease-embs.pkl')).set_index('mesh_id')['embs']

    def map_and_merge(self, biokg, drug_data):
        node_embs = []

        etype2embs = {
            EntityType.DISEASE: lambda x: self.dis_embs.loc[x],
            EntityType.UMLS: lambda x: self.umls_embs.loc[x].values,
            EntityType.TRIAL_ARM: lambda x: np.concatenate(
                (self.trial_embs.loc[x.split(":")[0]], self.trial_feats.loc[x.split(":")[0]])),
            EntityType.PROTEIN: lambda x: self.protein_df.loc[x].values,
            EntityType.DRUG: lambda x: self.drug_embs.loc[x],
        }

        oid2primary = {}
        for idx, row in tqdm(drug_data.iterrows()):
            for other_id in row['other_id']:
                if other_id.startswith("DB") and not other_id.startswith("DBSALT"):
                    oid2primary[other_id] = row['primary_id']

        for node, data in tqdm(biokg.nodes(data=True)):
            ntypes = data['ntypes']
            req_type = None
            for etype in ntypes:
                if etype not in [EntityType.DISEASE, EntityType.DRUG, EntityType.PROTEIN, EntityType.UMLS,
                                 EntityType.TRIAL_ARM, EntityType.PRIMARY_OUTCOME]:
                    continue
                if req_type is None or req_type == EntityType.UMLS:
                    req_type = etype
            #         assert req_type == etype, (req_type, etype)
            if req_type is None:
                continue
            eids = []
            for e in data['entities'].keys():
                if e.type != req_type:
                    continue
                eids.append(e.id)

            if req_type in [EntityType.TRIAL_ARM, EntityType.PROTEIN]:
                assert len(eids) == 1, (node, data)

            emb_fn = etype2embs[req_type]
            embs = []
            for eid in eids:
                try:
                    embs.append(emb_fn(eid))
                except KeyError as e:
                    if req_type == EntityType.DRUG:
                        assert eid == 'D#######' or oid2primary[eid] in eids, (eid, eids, node)
                    elif req_type not in [EntityType.UMLS, EntityType.PROTEIN]:
                        raise e
            if len(embs) == 0 and (req_type in [EntityType.PROTEIN, EntityType.UMLS] or eids == ['D#######']):
                continue
            assert len(embs) > 0, (node, data)
            embs = np.array(embs)
            emb = np.mean(embs, axis=0)
            node_embs.append({
                'node_id': node,
                'emb': emb,
                'etype': req_type.name
            })

        node_emb_df = pd.DataFrame(node_embs)
        node_emb_df['etype'] = node_emb_df['etype'].astype('category')
        return node_emb_df
