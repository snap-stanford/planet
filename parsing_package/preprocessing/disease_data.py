'''
Created on Oct 30, 2018

@author: maria
'''

from collections import defaultdict


class MeshReader:

    def __init__(self, mesh_file="d2019.bin", filter_categ=("C", "F01", "F02", "F03")):
        self.mesh_file = mesh_file
        self.filter_categ = filter_categ

    def __iter__(self):
        """Returns one mesh term record which belongs to specified categories at a time.
        Default categories are C (diseases) and F (Psychiatry and Psychology).
        """

        with open(self.mesh_file, encoding='utf-8') as fstream:
            add_entry = False
            name, ID, entries, treeIDs, desc = self._reset_attributes()

            for line in fstream:

                if line[0:7] == "*NEWREC":
                    if add_entry:
                        mesh_term = MeshTerm(name, ID, entries, treeIDs, desc)
                        yield mesh_term

                    add_entry = False
                    name, ID, entries, treeIDs, desc = self._reset_attributes()

                elif line[0:4] == "MH =":
                    name = line[4:].strip()

                elif line[0:5] == "ENTRY" or line[0:5] == "PRINT":
                    entries.add(line.split("=")[1].strip().split("|")[0])

                elif line[0:4] == "MN =":
                    tree_num = line[4:].strip()
                    for cat in self.filter_categ:
                        if tree_num.startswith(cat):
                            treeIDs.add(tree_num)
                            add_entry = True

                elif line[0:4] == "UI =":
                    ID = line[4:].strip()

                elif line[0:4] == "MS =":
                    desc = line[4:].strip()

    def _reset_attributes(self):
        name = ""
        ID = ""
        entries = set()
        treeIDs = set()
        desc = ""
        return name, ID, entries, treeIDs, desc


class SubMeshReader():

    def __init__(self, submesh_file="c2019.bin"):
        self.submesh_file = submesh_file

    def __iter__(self):
        """Iterates through mesh subheadings."""
        with open(self.submesh_file, encoding='utf-8') as fstream:
            head_mappings = []
            for line in fstream:
                if line[0:4] == ("HM ="):
                    mapping_name = line[4:].strip().replace('*', '').lower()
                    head_mappings.append(mapping_name)
                elif line[0:4] == ("UI ="):
                    submesh_id = line[4:].strip()
                    yield submesh_id, head_mappings
                    head_mappings = []


class MeshTerm():
    """Stores information about each MeSH term."""

    def __init__(self, name, ID, entry_terms, treeIDs, description):
        """
        name: name of MeSH term
        ID: unique ID in MeSH database
        entry_terms: other possible entries/names that can be used to refer to this term.
        tree_IDs: IDs of term in tree structure. Each term can have more tree_IDs.
        """
        self.name = name
        self.ID = ID
        self.entry_terms = entry_terms
        self.treeIDs = treeIDs
        self.description = description

        self.level = self._set_level()
        self.parents = set()
        self.children = set()

    def add_parent(self, parent):
        self.parents.add(parent)

    def add_child(self, child):
        self.children.add(child)

    def _set_level(self):
        """Sets level of the MeSH term. Level is the smallest number of nodes
        required to reach the root."""
        level = 10
        for entry in self.treeIDs:
            curr_level = len(entry.split("."))
            level = min(level, curr_level)
        return level

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        mesh_str = 'ID: {}, name: {}, desc: {} '.format(self.ID, self.name, self.description[:10])
        tree_nums = ','.join(self.treeIDs)
        mesh_str += 'tree numbers: {}, '.format(tree_nums)
        mesh_str += 'level: {}, '.format(self.level)
        parents = ','.join([parent.ID for parent in self.parents])
        mesh_str += 'parents: {}'.format(parents)

        return mesh_str


class MeshData(dict):
    """Holds the MeSH terms as dict."""

    def __init__(self, mesh_file="d2019.bin", submesh_file="", disgenet_file="",
                 filter_categ=("C", "F01", "F02", "F03")):
        super().__init__()
        self.filter_categ = filter_categ
        self._load_mesh_file(mesh_file)
        self.submesh_file = submesh_file
        self.disgenet_file = disgenet_file
        self.mesh2disgenet = {}
        self.disgenet_propagate = False
        self._set_names2id()

    def _load_mesh_file(self, mesh_file):
        """Read mesh file and save results."""
        reader = MeshReader(mesh_file, filter_categ=self.filter_categ)

        # Save alt_ids and their corresponding main GO ID. Add to GODag after populating GO Terms
        treeid2id = {}
        for term in reader:
            self[term.ID] = term
            for treeID in term.treeIDs:
                treeid2id[treeID] = term.ID

        self._set_parents(treeid2id)

    def _set_parents(self, treeid2id):
        """Add parent/child relationships."""
        for childID in self.keys():
            tree_nums = self[childID].treeIDs
            for child_treeID in tree_nums:
                parent_treeID = '.'.join(child_treeID.split(".")[:-1]).strip()
                if parent_treeID:
                    parentID = treeid2id[parent_treeID]
                    self[childID].add_parent(self[parentID])
                    self[parentID].add_child(self[childID])

    def _set_names2id(self):
        """Set mapping of all possible term names and entries to IDs."""
        self.names2id = {}
        for uid in self.keys():
            curr_mesh_term = self[uid]
            name = curr_mesh_term.name
            entry_terms = curr_mesh_term.entry_terms
            self.names2id[name.lower()] = uid
            for term in entry_terms:
                self.names2id[term.lower()] = uid

    def get_meshID(self, disease_name):
        """For given disease name, return mesh ID."""
        disease_name = disease_name.lower().strip()
        return self.names2id.get(disease_name)

    def get_level_parents(self, ID, parent_level):
        """Return all IDs of all parents at the given level."""
        all_parents = set()
        for parent in self[ID].parents:
            if parent.level == parent_level:
                all_parents.add(parent.ID)
            all_parents |= self.get_level_parents(parent.ID, parent_level)
        return all_parents

    def get_all_parents(self, ID):
        """Return all IDs of all parents at the given level."""
        all_parents = set()
        for parent in self[ID].parents:
            all_parents.add(parent.ID)
            all_parents |= self.get_all_parents(parent.ID)
        return all_parents

    def cut_mesh_tree(self, IDs, min_level):
        """All disease IDs with lower level are mapped to parent ID."""
        newIDs = set()
        for disID in IDs:
            if disID.startswith("HE"):  #indicates healthy subject, not from MeSH database
                newIDs.add(disID)
                continue
            mesh_term = self[disID]
            if mesh_term.level <= min_level:
                newIDs.add(disID)
            else:
                newIDs.update(self.get_level_parents(disID, min_level))
        return newIDs

    def get_disgenetID(self, meshID, propagate):
        """Get disgenet ID from mesh ID.
        meshID:
        propagate: propagate associations to all mesh parent terms
        """
        if not self.mesh2disgenet or self.disgenet_propagate != propagate:
            self._set_mesh2disgenet(propagate)
            self.disgenet_propagate = propagate

        return self.mesh2disgenet[meshID]

    def _set_mesh2disgenet(self, propagate):
        """
        Returns mapping of MeSH IDs to DisGeNET mapping.
        propagate: propagate associations to all mesh parent terms
        """

        self.mesh2disgenet, submesh2disgenet = self._read_disgenet_file()

        reader = SubMeshReader(self.submesh_file)
        for submesh_id, head_mappings in reader:
            disIDs = submesh2disgenet[submesh_id]
            for hm in head_mappings:
                if hm in self.names2id:
                    meshID = self.names2id[hm]
                    for disID in disIDs:
                        self.mesh2disgenet[meshID].add(disID)

        if propagate:  # propagate mapped meshID to all mesh parents
            mapped_mesh = list(self.mesh2disgenet.keys())
            for child in mapped_mesh:
                parents = self.get_all_parents(child)
                for p in parents:
                    self.mesh2disgenet[p].update(self.mesh2disgenet[child])

        #print(len(mesh2disgenet))
        #s = set()
        #for disIDs in mesh2disgenet.values():
        #    s.update([d for d in disIDs])
        #print(len(s))

        #new = set()
        #for mesh_term in mesh2disgenet.keys():
        #    if self[mesh_term].level==4:
        #        new.add(mesh_term)
        #    parents = self.get_level_parents(mesh_term, 4)
        #    new.update(parents)
        #print(len(new))

    def _read_disgenet_file(self):
        """Reads DisGeNET mapping. If possible maps directly MeSH ID to DisGeNET ID,
        otherwise maps subheading MeSH ID to DisGeNET ID. Returns both mappings."""
        #TODO check 1:1 mapping
        mesh2disgenet = defaultdict(set)
        submesh2disgenet = defaultdict(set)

        with open(self.disgenet_file, encoding='ISO-8859-1') as f:
            lines = f.readlines()

        disIDs = set()
        for line in lines[1:]:
            data = line.split("\t")
            disID = data[0]
            disIDs.add(disID)
            dis_name1 = data[1].lower()
            dis_name2 = data[4].lower()

            code = data[2]
            if code == "MSH":
                submesh_id = data[3]
                submesh2disgenet[submesh_id].add(disID)

            # try mapping names directly
            if dis_name1 in self.names2id:
                meshID = self.names2id[dis_name1]
                mesh2disgenet[meshID].add(disID)

            if dis_name2 in self.names2id:
                meshID = self.names2id[dis_name2]
                mesh2disgenet[meshID].add(disID)
        print(len(disIDs))
        return mesh2disgenet, submesh2disgenet


def __main__():
    ct_dir = "/Users/maria/Documents/Stanford/Clinical trials/external data/disease data/"
    mesh_file = ct_dir + "mesh_terms/d2019.bin"
    disgenet_file = ct_dir + "/disease-disease network/disgenet/disease_mappings.tsv"
    submesh_file = ct_dir + "/mesh_terms/c2019.bin"
    md = MeshData(mesh_file, submesh_file, disgenet_file)


if __name__ == "__main__":
    __main__()
#md.get_mesh2disgenet()
#print(md["D012559"])
#names2id = md.get_names2id()
#print(md.get_level_parents("D000544", 3))