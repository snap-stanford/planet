from collections import defaultdict
from enum import Enum
from typing import NamedTuple, Set

import networkx as nx
from tqdm import tqdm


class EntityType(Enum):
    DRUG = 1,
    DISEASE = 2,
    PROTEIN = 3,
    SIDE_EFFECT = 4,
    FUNCTION = 5,
    DRUGONT = 6,
    PRIMARY_OUTCOME = 7,
    TRIAL = 8,
    TRIAL_ARM = 9
    UMLS = 10
    SIDE_EFFECT_CAT = 11

    def __str__(self):
        return self.name


class Source(Enum):
    MESH = 1,
    DRUGBANK = 2,
    GO = 3,
    ENTREZ = 4,
    DIS_ONT = 5,
    CLASSYFIRE = 6,
    MULTISCALE_INTERACTOME = 7,
    GAF = 8,
    CLINICAL_TRIAL = 9,
    CUSTOM_VOCAB = 10,
    UMLS = 11,
    CTD = 12,
    DISGENET = 13,
    MEDRA = 14,
    ADRECS_TARGET = 15

    def __str__(self):
        return self.name


class EntityKey(NamedTuple):
    type: EntityType
    id: str
    source: Source


class Entity(NamedTuple):
    key: EntityKey
    attrs: dict
    name: str

    def __eq__(self, other):
        return self.key == other.key


class Concept(NamedTuple):
    id: str
    name: str
    entities: Set[Entity]


class Relation(NamedTuple):
    name: str
    source: Source


class BioKG:
    def __init__(self):
        self.graph = nx.MultiDiGraph()

    def add_entity(self, entity: Entity, allow_existing: bool = False):
        if entity.key in self.graph and not allow_existing:
            raise RuntimeError(f"Entity {entity.key} already in KG")

        self.graph.add_node(entity.key, label=entity.name, attrs=entity.attrs)

    def add_relation(self, e1: EntityKey, e2: EntityKey, relation: str, source: Source, attrs=None):
        if attrs is None:
            attrs = {}
        if e1 not in self.graph:
            raise RuntimeError(f"Node {e1} not in in graph")
        if e2 not in self.graph:
            raise RuntimeError(f"Node {e2} not in in graph")
        self.graph.add_edge(e1, e2, key=Relation(name=relation, source=source), relation=relation, source=source,
                            attrs=attrs)

    def add_mapping(self, e1: EntityKey, e2: EntityKey, source: Source):
        self.add_relation(e1, e2, 'KG-MERGE-SAME', source)


class UnionFind:
    def __init__(self):
        self.parent = {}

    # A utility function to find the subset of an element i
    def find_parent(self, i):
        if i not in self.parent:
            return i
        return self.find_parent(self.parent[i])

        # A utility function to do union of two subsets

    def union(self, x, y):
        x_set = self.find_parent(x)
        y_set = self.find_parent(y)
        if x_set != y_set:
            self.parent[x_set] = y_set


def to_networkx(biokg):
    self = biokg
    print("v.20")

    concept2merge = defaultdict(list)
    current_concept_id = 0
    cnt = 0

    uf = UnionFind()
    for u, v, data in tqdm(self.graph.edges(data=True)):
        if data['relation'] == 'KG-MERGE-SAME':
            cnt += 1
            uf.union(u, v)
    print("Node pairs to merge: ", cnt)
    g = nx.MultiDiGraph()
    entity2concept = {}
    for node, data in tqdm(self.graph.nodes(data=True)):
        nid = uf.find_parent(node)
        if nid not in entity2concept:
            entity2concept[nid] = f'KG{current_concept_id:08}'
            current_concept_id += 1

        cid = entity2concept[nid]

        e = Entity(key=node, name=data['label'], attrs=data['attrs'])
        if cid not in g:
            g.add_node(cid, entities={e.key: e}, ntypes={e.key.type})
        else:
            g.nodes[cid]['entities'][e.key] = e
            g.nodes[cid]['ntypes'].add(e.key.type)

    for u, v, k, data in tqdm(self.graph.edges(data=True, keys=True)):
        if data['relation'] == 'KG-MERGE-SAME':
            continue
        cidu = entity2concept[uf.find_parent(u)]
        cidv = entity2concept[uf.find_parent(v)]
        if not g.has_edge(cidu, cidv, k):
            g.add_edge(cidu, cidv, key=k, relation=data['relation'], source=data['source'], attrs=[data['attrs']],
                       extra_attrs={})
        elif data['attrs']:
            g.edges[cidu, cidv, k]['attrs'].append(data['attrs'])

    return g, concept2merge, entity2concept
