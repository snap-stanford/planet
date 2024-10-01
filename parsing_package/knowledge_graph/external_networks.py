import csv
import gzip
import os

from goatools import obo_parser
from goatools.anno.gaf_reader import GafReader
from tqdm import tqdm


class ExternalNetworkGenerator:
    def __init__(self, external_data_root):
        self.external_data_root = external_data_root
        self.uniprot2entrez = self._load_uniprot_to_entrez()

    def _load_uniprot_to_entrez(self):
        uniprot2entrez = {}
        with gzip.open(os.path.join(f'{self.external_data_root}/uniprot',
                                    'HUMAN_9606_idmapping_selected.tab.gz'),
                       mode='rt') as f:
            for line in tqdm(f):
                uniprot, _, entrez = line.split("\t")[:3]
                if uniprot in uniprot2entrez:
                    print(uniprot, entrez, uniprot2entrez[uniprot])
                uniprot2entrez[uniprot] = entrez
        return uniprot2entrez

    def _save_edges(self, edges, keys, name):
        basepath = self.external_data_root
        with open(f'{basepath}/{name}.tsv', 'w', newline='') as f:
            writer = csv.DictWriter(f, keys, delimiter='\t')
            writer.writeheader()
            for row in tqdm(edges):
                writer.writerow(row)

    def drug_protein(self, drug_data):
        uniprot2entrez = self.uniprot2entrez
        drug_data = drug_data.fillna(value={'targets': 0, 'carriers': 0, 'transporters': 0, 'enzymes': 0})

        drug_protein = []
        unfound_proteins = set()
        # https://www.uniprot.org/uploadlists/
        mapping_unfound = '''Q95IE3	3123
        Q9GIY3	3123
        Q30167	3123
        P01912	3123
        Q30134	3123
        P04229	3123
        Q29974	3123
        P18465	3106
        P13760	3123
        Q5Y7A7	3123
        P13761	3123
        Q13748	113457
        Q13748	7278
        Q9TQE0	3123
        P20039	3123
        P03989	3106'''
        for mapping in mapping_unfound.split("\n"):
            pro, entrez = mapping.split("\t")
            uniprot2entrez[pro] = entrez

        for idx, row in tqdm(drug_data.iterrows()):
            primary_id = row['primary_id']

            for key in ['targets', 'enzymes']:
                targets = row[key] or {key[:-1]: []}
                for target in targets[key[:-1]]:
                    actions = (target['actions'] or {'action': []})['action']
                    for protein in target.get('polypeptide', []):
                        uniprot_id = protein['@id']
                        source = protein['@source']
                        organism = protein['organism']
                        if organism['@ncbi-taxonomy-id'] != '9606':
                            continue
                        if uniprot_id not in uniprot2entrez:
                            unfound_proteins.add(uniprot_id)
                            continue
                        drug_protein.append({
                            "drug_id": primary_id,
                            "protein_uniprot_id": uniprot_id,
                            "protein_entrez_id": uniprot2entrez[uniprot_id],
                            "source": source,
                            "organism_id": organism['@ncbi-taxonomy-id'],
                            "organism_name": organism['$'],
                            "type": key[:-1],
                            "action": ",".join(actions)
                        })
        print(unfound_proteins)
        self._save_edges(drug_protein, name='drug-protein',
                         keys=['drug_id', 'protein_uniprot_id', 'protein_entrez_id', 'source', 'organism_id',
                               'organism_name', 'type', 'action'])

    def protein_function(self):
        go_base_path = f'{self.external_data_root}/go'

        uniprot2entrez = self.uniprot2entrez
        go_dag = obo_parser.GODag(go_base_path + '/go-basic.obo', ['relationship'])

        goa_human_path = os.path.join(go_base_path, 'goa_human.gaf')

        ogaf = GafReader(goa_human_path, godag=go_dag)

        protein_function = []
        evidence_subset = {'EXP', 'IDA', 'IMP', 'IGI', 'HTP', 'HDA', 'HMP', 'HGI'}
        for association in tqdm(ogaf.get_associations()):
            if association.Evidence_Code in evidence_subset:
                if association.DB_ID not in uniprot2entrez:
                    print(association.DB_ID)
                    continue
                protein_function.append({
                    "protein_id": association.DB_ID,
                    "protein_entrez_id": uniprot2entrez[association.DB_ID],
                    "protein_symbol": association.DB_Symbol,
                    "function_id": association.GO_ID,
                    "evidence_code": association.Evidence_Code,
                    "protein name": association.DB_Name,
                    "function_namespace": association.NS
                })
        self._save_edges(protein_function, name='protein-function',
                         keys=["protein_id", 'protein_entrez_id', "protein_symbol", "function_id", "evidence_code",
                               "protein name", "function_namespace"])

    def drug_phenotype(self):
        # TODO: need to updated the ctd but they dont have drugbank mapping now
        ctd_base_path = f'{self.external_data_root}/ctd'
        chemicals2drugbankid = {}
        # TODO: to update need to figure out how to map the mesh ids to drugbank id
        with gzip.open(os.path.join(ctd_base_path, 'CTD_chemicals.tsv.gz'), mode='rt') as f:
            for line in f:
                if line.strip() == '# Fields:':
                    break
            labels = next(f)[1:].strip().split("\t")
            next(f)
            reader = csv.DictReader(f, labels, delimiter='\t')
            for row in tqdm(reader):
                mesh_id = row['ChemicalID'][5:]
                drugbank_ids = row['DrugBankIDs'].split("|")
                if row['DrugBankIDs'] and len(drugbank_ids) > 0:
                    chemicals2drugbankid[mesh_id] = drugbank_ids
        print(len(chemicals2drugbankid))

        drug_phenotype = []
        ctd_base_path = '/afs/cs.stanford.edu/u/prabhat8/dfs/trials-data/external_data/ctd'
        with gzip.open(os.path.join(ctd_base_path, 'CTD_pheno_term_ixns.tsv.gz'), mode='rt') as f:
            for line in f:
                if line.strip() == '# Fields:':
                    break
            labels = next(f)[1:].strip().split("\t")
            next(f)
            reader = csv.DictReader(f, labels, delimiter='\t')
            for row in tqdm(reader):
                mesh_id = row['chemicalid']
                if mesh_id in chemicals2drugbankid and row['organismid'] == '9606':
                    for drugid in chemicals2drugbankid[mesh_id]:
                        for interaction in {x.split("^")[0] for x in row['interactionactions'].split("|")}:
                            drug_phenotype.append({
                                'drug_id': drugid,
                                'phenotype_id': row['phenotypeid'],
                                'phenotype_name': row['phenotypename'],
                                'relation': interaction
                            })
        self._save_edges(drug_phenotype, name='drug-phenotype',
                         keys=['drug_id', 'phenotype_id', 'phenotype_name', 'relation'])
