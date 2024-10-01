import csv
import os
from collections import defaultdict, Counter

import networkx as nx
import obonet
from goatools import obo_parser
from tqdm import tqdm

from .kg import BioKG, Entity, EntityKey, EntityType, Source, Relation

DATA_DIR = "data"
class KnowledgeGraphBuilder:
    def __init__(self, mesh_dis_data, drug_data, ext_basepath, cuid2term, umls_utils,
                 umls_graph_clip_threshold=10, 
                 build_ae=False):
        self.biokg = None
        self.mesh_dis_data = mesh_dis_data
        self.drug_data = drug_data
        self.ext_basepath = ext_basepath
        self._load_external_networks(ext_basepath)
        self.build_ae = build_ae
        self.umls_utils = umls_utils
        self.cuid2term = cuid2term
        self.umls_graph_clip_threshold = umls_graph_clip_threshold

    def _load_external_networks(self, ext_basepath):
        def read_tsv(filename):
            data = []
            with open(filename) as f:
                reader = csv.DictReader(f, delimiter='\t')
                for row in reader:
                    data.append(row)
            return data

        # external datasets
        # chemical ontology for drug-drug
        classyfire_path = f'{ext_basepath}/classyfire'

        chemical_ontology = os.path.join(classyfire_path, 'ChemOnt_2_1.obo')
        self.chemical_ont_graph = obonet.read_obo(chemical_ontology)

        # function-function
        go_obo_url = 'http://purl.obolibrary.org/obo/go-basic.obo'
        go_base_path = f'{ext_basepath}/go'

        self.go_dag = obo_parser.GODag(os.path.join(go_base_path, 'go-basic.obo'), ['relationship', 'xref'])

        # protein-function
        self.protein_function = read_tsv(os.path.join(ext_basepath, 'protein-function.tsv'))

        # drug-phenotype
        self.drug_phenotype = read_tsv(os.path.join(ext_basepath, 'drug-phenotype.tsv'))

        # drug-protein
        self.drug_protein = read_tsv(os.path.join(ext_basepath, 'drug-protein.tsv'))

        # protein-protein
        self.protein_protein = read_tsv(os.path.join(ext_basepath, 'multiscale-interactome', 'protein_to_protein.tsv'))

        # disease-protein/gene - disgenet
        self.disease_protein = read_tsv(os.path.join(ext_basepath, 'disgenet', 'curated_gene_disease_associations.tsv'))

        self.disease_mappings = read_tsv(os.path.join(ext_basepath, 'disgenet', 'disease_mappings.tsv'))

    def build_external_networks(self):
        self.biokg = BioKG()
        self._disease_disease()
        self._drug_drug()
        self._protein_protein()
        self._function_function()
        self._protein_function()
        self._drug_protein()
        self._drug_phenotype()
        self._disease_gene()

        # trials Stuff
        self._umls()
        if self.build_ae:
            self._ae_ae()
            self._ae_protein()

        self._primary_outcomes(f'{DATA_DIR}/outcome_data/clusters-outcome-measures.txt')

    def build_all(self, trial_df, filter_fn):
        self.build_external_networks()
        self._add_trial_graph(trial_df, filter_fn=filter_fn)
        self._remove_incomplete_arms()
        self._remove_incomplete_arms()

    def _disease_disease(self):
        biokg = self.biokg
        mesh_graph = nx.DiGraph()
        mesh_dis_key = lambda _id: EntityKey(type=EntityType.DISEASE, id=_id, source=Source.MESH)
        for key, value in tqdm(self.mesh_dis_data.items()):
            disease = Entity(key=mesh_dis_key(key), name=value.name, attrs={})
            biokg.add_entity(disease)
            mesh_graph.add_node(mesh_dis_key(key), name=value.name)
        biokg.add_entity(Entity(key=mesh_dis_key('HEALTHY'), name='HEALTHY', attrs={}))
        mesh_graph.add_node(mesh_dis_key('HEALTHY'), name='HEALTHY')
        for key, value in self.mesh_dis_data.items():
            e1 = mesh_dis_key(key)
            for parent in value.parents:
                e2 = mesh_dis_key(parent.ID)
                biokg.add_relation(e1, e2, relation='has_parent', source=Source.MESH)
                mesh_graph.add_edge(e1, e2, relation='has_parent')
        self.mesh_dis_key = mesh_dis_key
        self.mesh_graph = mesh_graph

    def _drug_drug(self):
        ontname2ontid = {}
        biokg = self.biokg
        drug_ont_key = lambda _id: EntityKey(type=EntityType.DRUGONT, id=_id, source=Source.CLASSYFIRE)
        for node, data in self.chemical_ont_graph.nodes(data=True):
            ontname2ontid[data['name']] = node
            biokg.add_entity(Entity(key=drug_ont_key(node), name=data['name'], attrs=data))

        for node in self.chemical_ont_graph.nodes():
            u = drug_ont_key(node)
            for parent in self.chemical_ont_graph[node]:
                v = drug_ont_key(parent)
                biokg.add_relation(u, v, relation='is_a', source=Source.CLASSYFIRE)

        drug_key = lambda _id: EntityKey(type=EntityType.DRUG, id=_id, source=Source.DRUGBANK)
        for idx, row in tqdm(self.drug_data.iterrows()):
            drug_id = row['primary_id']
            name = row['name']
            biokg.add_entity(Entity(key=drug_key(drug_id), name=name, attrs={}))
            for other_id in row['other_id']:
                if other_id.startswith("DB") and not other_id.startswith("DBSALT"):
                    biokg.add_entity(Entity(key=drug_key(other_id), name=name, attrs={}))
                    biokg.add_mapping(drug_key(drug_id), drug_key(other_id), Source.DRUGBANK)

            classification = row['classification']
            if not isinstance(classification, dict):
                continue
            for parent in classification.get('alternative-parent', []) + [classification['direct-parent']]:
                biokg.add_relation(drug_key(drug_id), drug_ont_key(ontname2ontid[parent]), relation='has_parent',
                                   source=Source.DRUGBANK)

        biokg.add_entity(Entity(drug_key('D#######'), name='placebo', attrs={}))
        self.drug_key = drug_key

    def _protein_protein(self):
        protein_key = lambda _id: EntityKey(type=EntityType.PROTEIN, id=_id, source=Source.ENTREZ)
        biokg = self.biokg
        for row in tqdm(self.protein_protein):
            n1 = protein_key(row['node_1'])
            n2 = protein_key(row['node_2'])
            biokg.add_entity(Entity(key=n1, name=row['node_1_name'], attrs={}), allow_existing=True)
            biokg.add_entity(Entity(key=n2, name=row['node_2_name'], attrs={}), allow_existing=True)
            biokg.add_relation(n1, n2, relation='affects', source=Source.MULTISCALE_INTERACTOME)
        self.protein_key = protein_key

    def _function_function(self):
        function_key = lambda _id: EntityKey(type=EntityType.FUNCTION, id=_id, source=Source.GO)
        biokg = self.biokg
        for goterm in tqdm(self.go_dag.values()):
            id = goterm.id
            namespace = goterm.namespace
            name = goterm.name

            # Avoid obsolete terms, reconsider if some drug/disease function edge does nt have corresponding go term
            if goterm.is_obsolete:
                continue
            if namespace != 'biological_process':
                continue

            n1 = function_key(id)
            biokg.add_entity(Entity(key=n1, name=name, attrs={}), allow_existing=True)

            def add_edge(relation, term):
                if term.namespace != 'biological_process':
                    return
                n2 = function_key(term.id)
                biokg.add_entity(Entity(key=n2, name=term.name, attrs={}), allow_existing=True)
                biokg.add_relation(n1, n2, relation=relation, source=Source.GO)

            for relation in goterm.relationship:
                for term in goterm.relationship[relation]:
                    add_edge(relation, term)
            for parent in goterm._parents:
                add_edge('is_a', self.go_dag[parent])

            for alt_id in goterm.alt_ids:
                biokg.add_entity(Entity(key=function_key(alt_id), name=name, attrs={}), allow_existing=True)
                biokg.add_mapping(n1, function_key(alt_id), source=Source.GO)
        self.function_key = function_key

    def _protein_function(self):
        biokg = self.biokg
        for row in tqdm(self.protein_function):
            if row['function_namespace'] != 'BP':
                continue
            protein = self.protein_key(row['protein_entrez_id'])
            if protein not in biokg.graph:
                # print(protein)
                continue
            biokg.add_relation(protein, self.function_key(row['function_id']),
                               relation='affects', source=Source.GAF, attrs={'evidence_code': row['evidence_code']})

    def _drug_phenotype(self):
        cnt = 0
        biokg = self.biokg
        for row in tqdm(self.drug_phenotype):
            drugk = self.drug_key(row['drug_id'])
            fkey = self.function_key(row['phenotype_id'])
            if fkey not in biokg.graph:
                cnt += 1
                continue
            biokg.add_relation(drugk, fkey, relation=row['relation'], source=Source.CTD)
        print(cnt)

    def _drug_protein(self):
        cnt = 0
        biokg = self.biokg
        for row in tqdm(self.drug_protein):
            drugk = self.drug_key(row['drug_id'])
            proteink = self.protein_key(row['protein_entrez_id'])
            if proteink not in biokg.graph:
                cnt += 1
                continue
            actions = row['action'].split(",")
            if row['action'] == '':
                actions = ['unspecified']
            for action in actions:
                if action not in ['inhibitor', 'substrate', 'antagonist', 'agonist', 'inducer']:
                    action = 'other'
                biokg.add_relation(drugk, proteink, relation=action, source=Source.DRUGBANK)
        print(cnt)

    def _disease_gene(self):
        biokg = self.biokg
        allowed_cuis = defaultdict(set)
        for row in tqdm(self.disease_mappings):
            cuid = row['diseaseId']
            if row['vocabulary'] == 'MSH' and self.mesh_dis_key(row['code']) in biokg.graph:
                allowed_cuis[cuid].add(('MSH', row['code']))

        cnt = 0
        dis = set()
        cnt2 = 0
        for row in tqdm(self.disease_protein):
            proteink = self.protein_key(row['geneId'].strip())
            disease_id = row['diseaseId']
            if disease_id not in allowed_cuis:
                cnt += 1
                dis.add(disease_id)
                continue
            if proteink not in biokg.graph:
                cnt2 += 1
                continue
            codes = allowed_cuis[disease_id]
            for code in codes:
                if code[0] == 'MSH':
                    biokg.add_relation(self.mesh_dis_key(code[1]), proteink, relation='dis-gene',
                                       source=Source.DISGENET)
        print(cnt, len(dis), cnt2)

    def _umls(self):
        cuid2term = self.cuid2term
        biokg = self.biokg
        umls_concepts = set()
        for cuid, parents in cuid2term[self.umls_graph_clip_threshold].items():
            umls_concepts.update(parents)

        self.umls_key = lambda _id: EntityKey(type=EntityType.UMLS, id=_id, source=Source.UMLS)
        for cuid in umls_concepts:
            biokg.add_entity(
                Entity(key=self.umls_key(cuid), name=self.umls_utils.cuid2concept.get(cuid, cuid), attrs={}),
                allow_existing=True)

        for cuid in umls_concepts:
            for vocab, vocab_key in [('MSH', self.mesh_dis_key), ('DRUGBANK', self.drug_key),
                                     ('GO', self.function_key)]:
                for code in self.umls_utils.concept2vocab[cuid][vocab]:
                    vk = vocab_key(code)
                    if not biokg.graph.has_node(vk):
                        if vocab == 'MSH':
                            # get the relations from chemical ontology here
                            # print(vk, cuid2concept[cuid])
                            pass
                        continue
                    biokg.add_mapping(self.umls_key(cuid), vk, source='UMLS')

    def _ae_ae(self):
        raise NotImplementedError

    def _ae_protein(self):
        raise NotImplementedError

    def _primary_outcomes(self, cid2cluster_path):
        cid2cluster = {}
        with open(cid2cluster_path, 'r') as f:
            for idx, line in enumerate(f.readlines()):
                cluster, count = line.strip().split("\t")
                cid2cluster[idx] = cluster
        self.pom_key = lambda _id: EntityKey(type=EntityType.PRIMARY_OUTCOME, id=_id, source=Source.CUSTOM_VOCAB)
        for idx, cluster_content in cid2cluster.items():
            self.biokg.add_entity(Entity(key=self.pom_key(idx), name=cluster_content, attrs={}))

    def _mesh_children(self):
        mesh_shortest_paths = nx.shortest_path(self.mesh_graph)
        mesh_id2children = {}
        for source in mesh_shortest_paths:
            children = set([target.id for target in mesh_shortest_paths[source]])
            children.remove(source.id)
            mesh_id2children[source.id] = children
        self.mesh_id2children = mesh_id2children

    def _add_trial_graph(self, trial_df, filter_fn):
        self._mesh_children()
        self.ae_cat_key = lambda _idx: EntityKey(type=EntityType.SIDE_EFFECT_CAT, id=_idx, source=Source.CLINICAL_TRIAL)
        cnt = 0
        weird = 0
        for idx, row in tqdm(trial_df.iterrows(), total=len(trial_df)):
            if not filter_fn(row):
                continue

            builder = TrialGraphBuilder(self, row)
            builder.build(use_population=True)
            if builder.weird:
                weird += 1
            if builder.no_disease:
                cnt += 1

        print(cnt, weird)

    def _remove_incomplete_arms(self):
        cnt = [0, 0, 0, 0]
        biokg = self.biokg
        for node, data in tqdm(list(biokg.graph.nodes(data=True))):
            if node.type == EntityType.TRIAL_ARM:
                cnt[3] += 1
                attrs = data['attrs']
                if attrs['is_incomplete'] or attrs['non_drug']:
                    biokg.graph.remove_node(node)
                    cnt[0] += 1
                    continue
                has_ae = False
                for nbr in biokg.graph[node]:
                    if nbr.type == EntityType.SIDE_EFFECT:
                        has_ae = True
                        break
                has_drugs = False
                for nbr in biokg.graph[node]:
                    if nbr.type == EntityType.DRUG:
                        has_drugs = True
                        break
                if not has_drugs:
                    # add placebo drug if an arm has no drug associated
                    biokg.add_relation(node, self.drug_key('D#######'), relation='arm_tests_drug',
                                       source=Source.CLINICAL_TRIAL)
                    cnt[2] += 1
                if not has_ae:
                    cnt[1] += 1
                    # biokg.graph.remove_node(node)
        print(cnt)

    def print_graph_stats(self, biograph, ent2concept):
        placebo_arms = set()
        drugs2arms = defaultdict(set)
        for node, data in tqdm(biograph.nodes(data=True)):
            ntypes = data['ntypes']
            if ntypes != {EntityType.TRIAL_ARM}:
                continue
            assert len(data['entities']) == 1

            drugs = set()
            for nbr in biograph[node]:
                if biograph.has_edge(node, nbr, key=Relation(name='arm_tests_drug', source=Source.CLINICAL_TRIAL)):
                    drugs.add(nbr)
            drugs2arms[tuple(sorted(drugs))].add(node)
            if len(drugs) == 1 and list(drugs)[0] == ent2concept[self.drug_key('D#######')]:
                placebo_arms.add(node)

        type_count = Counter()
        for node, data in tqdm(biograph.nodes(data=True)):
            if node in placebo_arms:
                type_count.update(['placebo_arm'])
            else:
                type_count.update(data['ntypes'])

        rel_count = Counter()
        for u, v, key in tqdm(biograph.edges(keys=True)):
            if u not in placebo_arms:
                rel_count[key] += 1

        for ntype, cnt in type_count.items():
            print(ntype, "\t", cnt)
        print("-" * 80)
        for rel, cnt in rel_count.items():
            print(rel.name, "\t", rel.source, "\t", cnt)


class TrialGraphBuilder:
    def __init__(self, graph_builder: KnowledgeGraphBuilder, trial):
        self.trial = trial
        self.graph_builder = graph_builder
        self.build_ae = graph_builder.build_ae
        self.mesh_id2children = graph_builder.mesh_id2children
        self.no_disease = False

    def build(self, use_population):

        self._determine_results()
        self._filter_mesh_ids()

        if not self.filtered_mesh_ids:
            self.no_disease = True
            return

        self._build_arms()
        self._disease()
        self._primary_outcome()
        self._interventions()
        if use_population:
            self._population()
        if not self.has_results:
            return
        self._results()

    def _determine_results(self):
        has_results = True
        row = self.trial
        if row['has_results'] == False:
            has_results = False
        self.reported_events = None
        if has_results:
            clinical_results = row['clinical_results']
            if 'reported_events' not in clinical_results:
                has_results = False  # no seious adverse events, just continue
            else:
                self.reported_events = clinical_results['reported_events']
        self.has_results = has_results

    def _filter_mesh_ids(self):
        row = self.trial
        mesh_ids = row['mesh_ids']
        if not isinstance(mesh_ids, set):
            mesh_ids = set()
        # We need to connect to only the child meshIDs and hence remove any meshId for which there is a childId
        filtered_ids = set()
        for mesh_id in mesh_ids:
            children = self.mesh_id2children[mesh_id]
            if not mesh_ids.intersection(children):
                filtered_ids.add(mesh_id)
        self.weird = mesh_ids and not filtered_ids
        self.filtered_mesh_ids = filtered_ids

    def _build_arms(self):
        row = self.trial
        nct_id = row['nct_id']
        # Add nodes for arms
        arms = row['arm_group']
        if not isinstance(arms, list):
            arms = [{'arm_group_label': 'default', 'arm_group_type': ''}]
        self.arm_labels = dict()

        self.arm_key = lambda _idx: EntityKey(type=EntityType.TRIAL_ARM, id=f'{nct_id}:{_idx}',
                                              source=Source.CLINICAL_TRIAL)
        for idx, arm in enumerate(arms):
            label = arm['arm_group_label'].lower()
            self.arm_labels[label] = idx
            arm['is_incomplete'] = False
            arm['non_drug'] = False
            self.graph_builder.biokg.add_entity(Entity(key=self.arm_key(idx), name=label, attrs=arm),
                                                allow_existing=True)

        if self.has_results:
            self._build_result_arms()

    def _build_result_arms(self):
        biokg = self.graph_builder.biokg
        group_list = self.reported_events['group_list']
        self.gid2group = {}
        for group in group_list['group']:
            id = group['@group_id']
            title = group.get('title', '')
            self.gid2group[id] = group
            if 'arm_id' in group:
                group['arm_key_id'] = group['arm_id']
            elif 'drug_details' in group:
                drug_ids = group['drug_ids']
                all_drugs_matched = group['all_drugs_matched']
                if drug_ids and all_drugs_matched and not group['has_extra_drugs']:
                    group['arm_key_id'] = f"group-{id}"
                    self.arm_labels[f'group-{title}-{id}'] = group['arm_key_id']
                    group['is_incomplete'] = False
                    group['non_drug'] = False
                    biokg.add_entity(Entity(key=self.arm_key(group['arm_key_id']), name=title, attrs=group))
                    for drug_id in group['drug_ids']:
                        biokg.add_relation(self.arm_key(group['arm_key_id']), self.graph_builder.drug_key(drug_id),
                                           relation='arm_tests_drug', source=Source.CLINICAL_TRIAL)
            else:
                continue

    def _disease(self):
        for mesh_id in self.filtered_mesh_ids:
            for arm_id in self.arm_labels.values():
                self.graph_builder.biokg.add_relation(self.arm_key(arm_id), self.graph_builder.mesh_dis_key(mesh_id),
                                                      relation='study-disease',
                                                      source=Source.CLINICAL_TRIAL)

    def _interventions(self):
        interventions = self.trial['intervention']
        if not isinstance(interventions, list):
            interventions = []
        for intervention in interventions:
            arm_group_label = intervention.get('arm_group_label', ['default'])
            if not isinstance(arm_group_label, list):
                arm_group_label = ['default']
            drug_ids = intervention.get('drug_ids', set())
            is_incomplete = len(drug_ids) == 0 or intervention['is_incomplete']
            non_drug = intervention['intervention_type'] not in ['Drug', 'Placebo']
            for arm_label in arm_group_label:
                arm_label = arm_label.lower()
                arm_id = self.arm_labels[arm_label]
                self.graph_builder.biokg.graph.nodes()[self.arm_key(arm_id)]['attrs']['is_incomplete'] |= is_incomplete
                self.graph_builder.biokg.graph.nodes()[self.arm_key(arm_id)]['attrs']['non_drug'] |= non_drug

            for drug_id in drug_ids:
                for arm_label in arm_group_label:
                    arm_label = arm_label.lower()
                    arm_id = self.arm_labels[arm_label]
                    self.graph_builder.biokg.add_relation(self.arm_key(arm_id), self.graph_builder.drug_key(drug_id),
                                                          relation='arm_tests_drug',
                                                          source=Source.CLINICAL_TRIAL)

    def _primary_outcome(self):
        row = self.trial
        primary_outcomes = row['primary_outcome']
        if not isinstance(primary_outcomes, list):
            primary_outcomes = []
        for primary_outcome in primary_outcomes:
            cluster_ids = primary_outcome['cids']
            for cluster_id in cluster_ids:
                for arm_id in self.arm_labels.values():
                    self.graph_builder.biokg.add_relation(self.arm_key(arm_id), self.graph_builder.pom_key(cluster_id),
                                                          relation='primary_outcome',
                                                          source=Source.CLINICAL_TRIAL)

    def _population(self):
        row = self.trial
        crit = row['ec_umls']
        for cat in crit:
            for inclusion in crit[cat]:
                for term_crit in crit[cat][inclusion]:
                    if term_crit.concept:
                        cuid = term_crit.concept['ui']
                        for p_cuid in self.graph_builder.cuid2term[self.graph_builder.umls_graph_clip_threshold].get(
                                cuid, []):
                            for arm_id in self.arm_labels.values():
                                self.graph_builder.biokg.add_relation(self.arm_key(arm_id),
                                                                      self.graph_builder.umls_key(p_cuid),
                                                                      relation=f'eligibility-{inclusion}',
                                                                      source=Source.CLINICAL_TRIAL)

    def _results(self):
        for event_type in ['serious_events', 'other_events']:
            event_dict = self.reported_events.get(event_type, None)
            if not event_dict:
                continue

            gid2subjects = {}

            for category in event_dict['category_list'].get('category', []):
                for event in category['event_list'].get('event', []):
                    for count in event['counts']:
                        gid = count['@group_id']
                        if '@subjects_at_risk' in count:
                            at_risk = int(count['@subjects_at_risk'])
                            gid2subjects[gid] = at_risk

            self._ae_category(event_dict, gid2subjects, event_type)

            if self.build_ae:
                self._aes(event_dict, gid2subjects, event_type)

    def _ae_category(self, event_dict, gid2subjects, event_type):
        biokg = self.graph_builder.biokg
        # Extract info for top level headers also separately as we do not need to match to AE id for this
        for category in event_dict['category_list'].get('category', []):
            title = category.get('title', '').lower()
            arm2info = defaultdict(list)
            for event in category['event_list'].get('event', []):
                subtitle = event.get('sub_title', 'NONE')
                if not isinstance(subtitle, str):
                    subtitle = subtitle['$']
                for count in event['counts']:
                    gid = count['@group_id']
                    group = self.gid2group[gid]
                    if 'arm_key_id' not in group:  # add the arm for drug details also
                        continue
                    affected = int(count.get('@subjects_affected', 0))
                    # take the total at risk from the total field
                    at_risk = int(count.get('@subjects_at_risk', gid2subjects.get(gid, 0)))
                    num_events = int(count.get('@events', 0))
                    arm2info[group['arm_key_id']].append([subtitle, affected, at_risk, num_events])

            biokg.add_entity(Entity(key=self.graph_builder.ae_cat_key(title), name=title, attrs={}),
                             allow_existing=True)
            for arm_id, info in arm2info.items():
                biokg.add_relation(self.arm_key(arm_id), self.graph_builder.ae_cat_key(title),
                                   relation=f'has_side_effect_category_{event_type}',
                                   source=Source.CLINICAL_TRIAL,
                                   attrs={'event_type': event_type, 'ae_category': title, 'events': info})

    def _aes(self, event_dict, gid2subjects, event_type):
        biokg = self.graph_builder.biokg
        for category in event_dict['category_list'].get('category', []):
            title = category.get('title', '').lower()
            for event in category['event_list'].get('event', []):
                subtitle = event.get('sub_title', 'NONE')
                vocab = None
                if not isinstance(subtitle, str):
                    ae = subtitle['$']
                    vocab = subtitle['@vocab']
                else:
                    ae = subtitle
                ae = ae.lower()  # convert to UMLS/MedRA id
                cuids = self.ae2cuids[ae]
                if not cuids and vocab:
                    cuids = self.ae2cuids[vocab]

                for cuid in cuids:
                    for count in event['counts']:
                        gid = count['@group_id']
                        group = self.gid2group[gid]
                        if 'arm_key_id' not in group:  # add the arm for drug details also
                            continue
                        affected = int(count.get('@subjects_affected', 0))
                        # take the total at risk from the total field
                        at_risk = int(count.get('@subjects_at_risk', gid2subjects.get(gid, 0)))
                        num_events = int(count.get('@events', 0))
                        biokg.add_relation(self.arm_key(group['arm_key_id']), self.graph_builder.ae_key(cuid),
                                           relation=f'has_side_effect_{event_type}',
                                           source=Source.CLINICAL_TRIAL,
                                           attrs={'affected': affected, 'at_risk': at_risk, 'event_type': event_type,
                                                  'ae': ae})
