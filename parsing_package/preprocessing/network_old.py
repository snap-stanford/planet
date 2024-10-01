'''
Created on Oct 21, 2018

@author: maria
'''
import os
import pandas as pd
from collections import Counter
from builtins import float
import networkx as nx
from preprocessing import drug_data, disease_data
import re

def create_drug_disease_network(data, index, drugbank_data, mesh_data, min_appearance=1000, 
                       outdir="", include_healthy=False, min_mesh_level=4, 
                       filter_disgenet=True):
    """
    Generates drug-disease network for given index in clinical trial dataset.
    data: data frame of clinical trials data
    index: which index (column) should be used for generating connections. One network
           is created for each possible value of the given index
    drugbank_data: data from DrugBank containing mappings of drugs to IDs
    mesh_data: data from MeSH hierarchy. Contains disease IDs, entry terms and hieararchy structure
    min_apperance: minimum number of appearances of a value needed to construct
                   a network
    outdir: directory where resulting networks will be saved
    include_healthy: include healthy as condition 
    min_mesh_level: map all mesh terms with lower level to min_level parent
    filter_disgenet: use only MeSH term covered in DisGeNET
    """
    # TODO see how to use numeric indexes (enrollment, duration), dates (start, completion) and years
    # (min_age and max_age)
    data = data.loc[:,['nct_id','condition','mesh_terms','drug_name', 'drug_other_name',index]]
    data.dropna(subset=[index])
    
    freq_values = Counter()
    for _,rows in data.iterrows():
        value = rows[index]
        if isinstance(value, float):
            continue
        values = str_to_list(value)
        if len(values)>10:
            continue
        for v in values:
            freq_values.update([v])
   
    print(freq_values)
    graphs = {}
    
    num_mapped = 0
    
    for _,rows in data.iterrows():
        mapped = False
        nctID = rows['nct_id']
        conditions = rows['condition']
        mesh_terms = rows['mesh_terms']
        values = rows[index]
        drugs = str_to_list(rows['drug_name'])
        other_names = dict(zip(drugs,str_to_list(rows['drug_other_name'])))
        
        conditions = str_to_list(conditions)
        if not conditions:
            continue
        mesh_terms = str_to_list(mesh_terms)
        values = str_to_list(values)
        
        for value in values:
            if freq_values[value]<min_appearance:
                continue
            graphs.setdefault(value, nx.DiGraph())
            
            disIDs = map_conditions(conditions+mesh_terms, mesh_data, include_healthy, filter_disgenet)
            #newIDs = []
            #for disID in disIDs:
            #    newIDs+=mesh2disgenet[disID]
            #disIDs = newIDs
            
            if min_mesh_level!=None:
                disIDs = cut_mesh_tree(disIDs, mesh_data, min_mesh_level)
        
            #if not disIDs:
            #   print(nctID+"\t"+str(conditions)+"\t"+str(mesh_terms))
            for disID in disIDs:
                for drug in drugs:
                    drugID = map_drug(drug, other_names, drugbank_data)
                    #print(len(drugbank_mapping))
                    if drugID!=None:
                        graphs[value].add_edges_from([(disID, drugID)], nct_id=nctID)
                        mapped= True
        if mapped:
            num_mapped+=1               
    print("Mapped trials: {}/{}".format(num_mapped, data.shape[0]))
                    
    for k,G in graphs.items():
        nx.write_edgelist(G, os.path.join(outdir, k.replace("/","_")+".edgelist"), delimiter=",")
        
def create_location_network(data, drug, index, drugbank_data, 
                            mesh_data, min_appearance=1000, outdir="", include_healthy=False, 
                            min_mesh_level=4, filter_disgenet=False):
    """
    Generates drug-location or disease-location network for given index in clinical trial dataset.
    data: data frame of clinical trials data
    index: which index (column) should be used for generating connections. One network
           is created for each possible value of the given index
    drugbank_mapping: set of drug names from drugbank (synonyms and international
                    brands) and associated drugbank id
    mesh_data: data from MeSH hierarchy. Contains disease IDs, entry terms and hieararchy structure
    min_apperance: minimum number of appearances of a value needed to construct
                   a network
    outdir: directory where resulting networks will be saved
    include_healthy: include healthy as condition 
    min_mesh_level: map all mesh terms with lower level to min_level parent
    filter_disgenet: use only MeSH term covered in DisGeNET
    """
    # TODO see how to use numeric indexes (enrollment, duration), dates (start, completion) and years
    # (min_age and max_age)
    data = data.loc[:,['nct_id','location_countries','condition','mesh_terms','drug_name', 
                       'drug_other_name',index]]
    data.dropna(subset=[index])
    
    freq_values = Counter()
    for _,rows in data.iterrows():
        value = rows[index]
        if isinstance(value, float):
            continue
        values = str_to_list(value)
        if len(values)>10:
            continue
        for v in values:
            freq_values.update([v])
   
    print(freq_values)
    graphs = {}
    
    num_mapped = 0
    
    for _,rows in data.iterrows():
        mapped = False
        nctID = rows['nct_id']
        conditions = rows['condition']
        mesh_terms = rows['mesh_terms']
        location = rows['location_countries']
        values = rows[index]
        drugs = str_to_list(rows['drug_name'])
        other_names = dict(zip(drugs,str_to_list(rows['drug_other_name'])))
        
        conditions = str_to_list(conditions)
        if not conditions:
            continue
        mesh_terms = str_to_list(mesh_terms)
        values = str_to_list(values)
        location = str_to_list(location)
        
        for value in values:
            if freq_values[value]<min_appearance:
                continue
            graphs.setdefault(value, nx.DiGraph())
            
            IDs = set()
            if not drug:
                IDs = map_conditions(conditions+mesh_terms, mesh_data, include_healthy, filter_disgenet)
            
                if min_mesh_level!=None:
                    IDs = cut_mesh_tree(IDs, mesh_data, min_mesh_level)
            else:
                for drug in drugs:
                    drugID = map_drug(drug, other_names[drug], drugbank_data)
                    if drugID!=None:
                        IDs.update([drugID])
                        
            for ID in IDs:
                for country in location:
                    graphs[value].add_edges_from([(ID, country)], nct_id=nctID)
                mapped= True
        if mapped:
            num_mapped+=1               
    print("Mapped trials: {}/{}".format(num_mapped, data.shape[0]))
                    
    for k,G in graphs.items():
        nx.write_edgelist(G, os.path.join(outdir, k.replace("/","_")+".edgelist"), delimiter=",")
        
def map_drug(drug_name, other_names, drugbank_data):
    """"
    For given drug name finds drugbank id. If ID is not found, returns none.
    drug_name: original drug name in clinical trials
    other_names: other names from clinical_trials separated by |
    drugbank_data: data from DrugBank containing mappings of drugs to IDs
    """
    ID = drugbank_data.get_ID(drug_name)
    if ID!=None:
        return ID
    
    #otherwise try to map other names
    if not drug_name in other_names:
        return None
    
    other_names = other_names[drug_name]
    other_names = [o for o in other_names.split("|")]
    for synonym in other_names:
        ID = drugbank_data.get_ID(synonym)
        if ID!=None:
            return ID
    return None
        
def cut_mesh_tree(disIDs, mesh_data, min_level):
    """All disease IDs with lower level are mapped to parent ID."""
    newIDs = set()
    for disID in disIDs:
        if disID.startswith("HE"): #indicates healthy subject, not from MeSH database
            newIDs.add(disID)
            continue
        mesh_term = mesh_data[disID]
        if mesh_term.level<=min_level:
            newIDs.add(disID)
        else:
            newIDs.update(mesh_data.get_level_parents(disID, min_level))
    return newIDs
                    
    
def map_conditions(condition_entries, mesh_data, include_healthy, filter_disgenet):
    """"
    condition_entries: original condition names and mesh terms
    mesh_data: data from MeSH hierarchy. Contains disease IDs, entry terms and hieararchy structure
    include_healthy: include healthy as condition 
    filter_disgenet: use only MeSH term covered in DisGeNET
    return MeshIDs of conditions
    """
    meshIDs = set()
    for condition in condition_entries:
        meshID = mesh_data.get_meshID(condition)
        if meshID==None:
            continue
        if filter_disgenet:
            disgenetID = mesh_data.get_disgenetID(meshID, propagate=False)
            if disgenetID!=None:
                meshIDs.add(meshID)
        else:
            meshIDs.add(meshID)
    if meshIDs:
        return meshIDs 
            
    if include_healthy: # try to map healthy
        for condition in condition_entries:
            condition = condition.lower()
            healty_pattern = re.compile("healthy\s*(human)?\s*(volunteer(s)?|subject(s)?)?")
            res = healty_pattern.search(condition)
            if res!=None:
                meshIDs.add("HE00001")
                return meshIDs
    return meshIDs

def str_to_list(text):
    """
    String in form "('a','b')" converts to list.
    """
    if isinstance(text, float): # value is nan
        return []
    if text[0]=='(':
        splitted = text[1:-1].split('\',')
        if not splitted[-1].strip(): # remove last if empty
            splitted = splitted[:-1]
        return [s.strip().replace('\'','') for s in splitted]
    else:
        return [text.strip()]
    
ct_dir = "/Users/maria/Documents/Stanford/Clinical trials/"
drugbank_data = drug_data.DrugBank(os.path.join(ct_dir, "external data/drugbank data/synonyms"))

data = pd.read_csv(os.path.join(ct_dir, "drug-disease network", "structured_drug_data.csv"))
mesh_file = ct_dir+"external data/disease data/mesh_terms/d2018.bin"
disgenet_file = ct_dir+"external data/disease data/disease-disease network/disgenet/disease_mappings.tsv"
submesh_file = ct_dir+"external data/disease data/mesh_terms/c2018.bin"
meshData = disease_data.MeshData(mesh_file, submesh_file, disgenet_file)

#print(disease_mapping['alzheimer\'s disease'])
index = 'facility_state'
create_drug_disease_network(data, index, drugbank_data, meshData, min_appearance=1000,
                   outdir=ct_dir+"/drug-disease network/"+index+"/")
#create_location_network(data, 'True',index,drug_mapping, meshData, min_appearance=1000,
#                   outdir=ct_dir+"/location network/"+index)
