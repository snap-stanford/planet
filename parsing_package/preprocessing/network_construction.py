'''
Created on Oct 21, 2018

@author: maria
'''
import os
import networkx as nx
from preprocessing import drug_data, disease_data, ctrials_data
from preprocessing.ctrials_data import TrialField

def create_drug_disease_network(ctrials, index, min_appearance=500, 
                       outdir=""):
    """
    Generates drug-disease network for given index in clinical trial dataset.
    ctrial_data: dictionary of clinical trial enties
    index: which index (column) should be used for generating connections. One network
           is created for each possible value of the given index
    min_apperance: minimum number of appearances of a value needed to construct
                   a network
    outdir: directory where resulting networks will be saved
    """
    # TODO see how to use numeric indexes (enrollment, duration), dates (start, completion) and years
    # (min_age and max_age)
    freq_values = ctrials.get_frequency(index)
    print(freq_values)
    graphs = {}
    
    num_mapped = 0
    
    for _, trial in ctrials.items():
        mapped = False
        
        values = trial.get_field_values(index)
        if not values:
            continue
        
        for value in values:
            if freq_values[value]<min_appearance:
                continue
            graphs.setdefault(value, nx.DiGraph())
            
            if not trial.all_drugs_mapped:
                #print("not "+trial.nctID)
                continue
            
            if len(trial.drugIDs)>1:
                continue
            #else:
                #print("yes "+trial.nctID)
                
            for disID in trial.disIDs:
                for drugID in trial.drugIDs:
                    graphs[value].add_edges_from([(disID, drugID)], nct_id=trial.nctID)
                    mapped= True
        if mapped:
            num_mapped+=1               
    print("Mapped trials: {}/124k".format(num_mapped))
                    
    for k,G in graphs.items():
        nx.write_edgelist(G, os.path.join(outdir, k.replace("/","_")+".edgelist"), delimiter=",")
        
ct_dir = "/Users/maria/Documents/Stanford/Clinical trials/"
drugbank_data = drug_data.DrugBank(os.path.join(ct_dir, "external data/drugbank data/synonyms"))

ct_file= os.path.join(ct_dir, "drug-disease network", "structured_drug_data.csv")
mesh_file = ct_dir+"external data/disease data/mesh_terms/d2018.bin"
disgenet_file = ct_dir+"external data/disease data/disease-disease network/disgenet/disease_mappings.tsv"
submesh_file = ct_dir+"external data/disease data/mesh_terms/c2018.bin"
mesh_data = disease_data.MeshData(mesh_file, submesh_file, disgenet_file)

index = 'status'
ctrials = ctrials_data.ClinicalTrialsData(ct_file, [TrialField.status], mesh_data, drugbank_data, 
                                          min_mesh_level=None)

create_drug_disease_network(ctrials, TrialField.status, min_appearance=200,
                   outdir=ct_dir+"/drug-disease network/"+index+"/")
