import os
import re

from preprocessing.disease_data import MeshData


class DiseaseExtract:
    def __init__(self, data_dir: str, data_year: int):
        self.special_maps = {
            'HIV-1 Infection': 'HIV Infections',
            'HIV': 'HIV Infections',
            'ADHD': 'Attention Deficit Disorder with Hyperactivity',
            'HIV-1': 'HIV Infections',
            'Diabetes': 'Diabetes Mellitus',
            'HIV/AIDS': 'HIV Infections',
            'HIV I Infection': 'HIV Infections',
            'HIV-2 Infection': 'HIV Infections',
            'HIV-1 Infections': 'HIV Infections',
            'HIV-1-infection': 'HIV Infections',
            'Type Two Diabetes Mellitus': 'Diabetes Mellitus, Type 2',
            'Learning Disorders': 'Learning Disabilities',
            'Unspecified Adult Solid Tumor, Protocol Specific': 'Neoplasms',
            'Solid Neoplasm': 'Neoplasms',
            'Solid Neoplasms': 'Neoplasms',
            'Various Neoplasm': 'Neoplasms',
        }
        self.mesh_dis_data = self._get_disease_data(f'{data_dir}/disease_data/{data_year}',
                                                    data_year)
        self.cnt = 0

    @staticmethod
    def _get_disease_data(root_dir, data_year):
        mesh_file = os.path.join(root_dir, f'd{data_year}.bin')
        disgenet_file = os.path.join(root_dir, 'disease_mappings.tsv')
        submesh_file = os.path.join(root_dir, f"c{data_year}.bin")
        return MeshData(mesh_file, submesh_file, disgenet_file, filter_categ=["C", "F01", "F02", "F03"])

    def _map_conditions(self, condition_entries, mesh_terms=(), ignore=False):
        """"
        condition_entries: original condition names and mesh terms
        mesh_data: data from MeSH hierarchy. Contains disease IDs, entry terms and hieararchy structure
        include_healthy: include healthy as condition
        filter_disgenet: use only MeSH term covered in DisGeNET
        return MeshIDs of conditions
        """
        if mesh_terms is None:
            mesh_terms = set()
        mesh_ids = set()
        healthy_pattern = re.compile(r"healthy\s*(human)?\s*(volunteer(s)?|subject(s)?)?|health|human volunteer")
        for condition in condition_entries:
            mesh_id = self.mesh_dis_data.get_meshID(condition)
            if mesh_id is None and 'Cancer' in condition:
                condition = condition.replace('Cancer', 'Neoplasm')
                mesh_id = self.mesh_dis_data.get_meshID(condition)
            if mesh_id is None and 'Neoplasm' in condition and 'Advanced' in condition:
                condition = condition.replace('Advanced', '').strip()
                condition = re.sub(r'\s+', ' ', condition)
                if condition[-1] in [',', '.']:
                    condition = condition[:-1]
                mesh_id = self.mesh_dis_data.get_meshID(condition)
            if mesh_id is None:
                if condition in self.special_maps:
                    mesh_id = self.mesh_dis_data.get_meshID(self.special_maps[condition])
                    assert mesh_id is not None
                elif healthy_pattern.search(condition.lower()):
                    mesh_id = 'HEALTHY'
                else:
                    if ignore:
                        continue
                    else:
                        return False
            mesh_ids.add(mesh_id)

        for condition in mesh_terms:
            mesh_id = self.mesh_dis_data.get_meshID(condition)
            mesh_ids.add(mesh_id)

        return mesh_ids

    def get_disease_ids(self, trial):
        conditions = trial['condition']

        mesh_terms = trial.get('intervention_mesh_terms', [])
        # We are actually using condition mesh terms, they are switched in parsing
        if not isinstance(mesh_terms, list):
            mesh_terms = []

        mesh_ids = self._map_conditions(conditions)
        if not mesh_ids:
            self.cnt += 1
            mesh_ids = self._map_conditions(conditions, mesh_terms=mesh_terms, ignore=True)

        return mesh_ids