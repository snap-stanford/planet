import os
from collections import defaultdict

from tqdm import tqdm


class UMLSUtils:
    def __init__(self, umls_dir: str):
        self.umls_dir = umls_dir
        self.cuid2parents = {}
        self._load_cuid2concept()
        self._concept2vocab = None

    def _load_cuid2concept(self):
        cuid2concept = {}
        with open(f'{self.umls_dir}/META/MRCONSO.RRF') as f:
            for concept in tqdm(f):
                details = concept.strip().split("|")
                if details[1] == 'ENG' and details[2] == 'P':
                    cuid2concept[details[0]] = details[14]
        self.cuid2concept = cuid2concept

    @property
    def concept2vocab(self):
        if self._concept2vocab is None:
            self._load_concept2vocab()
        return self._concept2vocab

    def _load_concept2vocab(self):
        concept2vocab = defaultdict(lambda: defaultdict(set))
        with open(f'{self.umls_dir}/META/MRCONSO.RRF') as f:
            for line in tqdm(f):
                cols = ['CUI', 'LAT', 'TS', 'LUI', 'STT', 'SUI', 'ISPREF', 'AUI', 'SAUI', 'SCUI', 'SDUI', 'SAB', 'TTY',
                        'CODE', 'STR', 'SRL', 'SUPPRESS', 'CVF']
                vals = line.strip().split("|")[:-1]
                # rint(vals, cols)
                assert len(vals) == len(cols)
                parsed = dict(zip(cols, vals))
                if parsed['SUPPRESS'] in ['', "N"]:
                    concept2vocab[parsed['CUI']][parsed['SAB']].add(parsed['CODE'])
        self._concept2vocab = concept2vocab

    def load_relations(self):
        relations_path = os.path.join(self.umls_dir, 'concept_relations.pkl')
        if True or not os.path.exists(relations_path):
            relations = defaultdict(lambda: defaultdict(set))
            relations_flat = defaultdict(set)
            # CUI1 | AUI1 | STYPE1 | REL | CUI2 | AUI2 | STYPE2 | RELA | RUI | SRUI | SAB | SL | RG | DIR | SUPPRESS |CVF
            # 0    | 1    | 2      | 3   | 4    | 5    | 6      | 7    | 8   | 9    | 10 | 11  | 12 | 13  | 14       | 15
            # CUI2 has relation REL with CUI1
            with open(f'{self.umls_dir}/META/MRREL.RRF') as f:
                for rel in tqdm(f):
                    rel = rel.strip().split("|")
                    relations[rel[3]][rel[4]].add((rel[0], rel[7], rel[10], rel[11], rel[14]))
                    relations_flat[rel[4]].add((rel[0], rel[3], rel[7], rel[10], rel[11], rel[14]))
            self.relations = {k: v for k, v in relations.items()}
            self.relations_flat = {k: v for k, v in relations_flat.items()}
        #     with open(relations_path, 'wb') as f:
        #         pickle.dump(self.relations, f)
        #     with open(os.path.join(self.umls_dir, 'concept_relations_flat.pkl'), 'wb') as f:
        #         pickle.dump(self.relations_flat, f)
        # else:
        #     with open(relations_path, 'rb') as f:
        #         gc.disable()
        #         self.relations_flat = pickle.load(f)
        #         gc.enable()
        #     with open(os.path.join(self.umls_dir, 'concept_relations_flat.pkl'), 'rb') as f:
        #         gc.disable()
        #         self.relations = pickle.load(f)
        #         gc.enable()

    def parents(self, cuid):
        if cuid not in self.cuid2parents:
            self.cuid2parents[cuid] = self._parents(cuid)
        return self.cuid2parents[cuid]

    @staticmethod
    def _get_relation_attribute(rel, attribute):
        # (rel[0], rel[7], rel[10], rel[11], rel[14])
        #  0       1      2         3        4
        # CUI1 | AUI1 | STYPE1 | REL | CUI2 | AUI2 | STYPE2 | RELA | RUI | SRUI | SAB | SL | RG | DIR | SUPPRESS |CVF
        # 0    | 1    | 2      | 3   | 4    | 5    | 6      | 7    | 8   | 9    | 10 | 11  | 12 | 13  | 14       | 15
        if attribute == 'SUPPRESS':
            return rel[4]
        if attribute == 'VOCAB':
            return rel[2]
        if attribute == 'RELA':
            return rel[1]

    def _parents(self, cuid):
        relations = self.relations
        cuid2concept = self.cuid2concept
        parents = set()
        if cuid in relations['CHD']:  # if cuid is child of any concept
            for rel in relations['CHD'][cuid]:
                if self._get_relation_attribute(rel, 'SUPPRESS') in ['N', ''] and rel[0] in cuid2concept:
                    parents.add(rel[0])  # select the ones from some semantic list
            if cuid in parents:
                parents.remove(cuid)
        if len(parents) == 0 and cuid in relations['RN']:
            for rel in relations['RN'][cuid]:
                if self._get_relation_attribute(rel, 'SUPPRESS') in ['N', ''] and rel[0] in cuid2concept:
                    parents.add(rel[0])  # select the ones from some semantic list
            if cuid in parents:
                parents.remove(cuid)
        if len(parents) == 0 and cuid in relations['RQ']:
            rels = []
            for rel in relations['RQ'][cuid]:
                if self._get_relation_attribute(rel, 'RELA') == 'classified_as' \
                        and self._get_relation_attribute(rel, 'VOCAB') == 'MDR':
                    rels.append(rel)
            if len(rels) > 1:
                pass
            elif len(rels) == 1:
                # replace the current term with a representative term
                # print(criterion.term, criterion.concept['name'], cuid2concept[rels[0][0]])
                parents = self.parents(rels[0][0])
            else:
                for rel in relations['RQ'][cuid]:
                    if self._get_relation_attribute(rel, 'RELA') == 'isa' and rel[0] in cuid2concept:
                        parents.add(rel[0])
                if cuid in parents:
                    parents.remove(cuid)
        if len(parents) == 0 and cuid in relations['RO']:
            for rel in relations['RO'][cuid]:
                if self._get_relation_attribute(rel, 'VOCAB') == 'MTH' and rel[0] in cuid2concept:
                    parents.add(rel[0])
            if cuid in parents:
                parents.remove(cuid)
        # if len(parents) == 0:
        #     parents.add(cuid)
        return parents
