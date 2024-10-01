# utf-8
import pandas as pd
from pprint import pprint
import re
import time
import os
from disease_data import MeshData

import multiprocessing
import traceback
import glob


def get_drug_trials(data):
    interventional_trials = data[data.study_type == 'Interventional']
    is_any_drug = interventional_trials['intervention'].apply(lambda inter: type(inter) == list and any(x['intervention_type'] == 'Drug' for x in inter))
    drug_trials = interventional_trials[is_any_drug]
    return drug_trials#[drug_trials['intervention'].apply(len) < 4]

def get_drug_data(path):
    drug_id_map = pd.read_pickle(path)[['name', 'id', 'primary_name', 'name-category']]
    k = len(drug_id_map)
    drug_id_map.loc[k+1] = ['placebo', 'D#######', 'Placebo', 'primary']
    drug_id_map = drug_id_map.drop_duplicates(subset=['name', 'id'])
    print("Drug Id Map size", len(drug_id_map))
    drug_id_map = drug_id_map.sort_values(['name', 'name-category'])
    drug_id_map.set_index(['name', 'name-category'])
    return drug_id_map

def get_disease_data(root_dir):
    mesh_file = os.path.join(root_dir, 'd2019.bin')
    disgenet_file = os.path.join(root_dir, 'disease_mappings.tsv')
    submesh_file = os.path.join(root_dir, "c2019.bin")
    return  MeshData(mesh_file, submesh_file, disgenet_file)

def clean_drug_name(drug):
    """"
    If drug name is not mapped try to edit name by removing
    number, mg & parts in brackets.
    """
    if('placebo' in drug):
        return 'placebo'
    drug = re.sub('\"', " ", drug)
    drug = re.sub(r'[>≥]?\s*\d+\.?\d*\s*(mg|mcg||%|micrograms)', ' ', drug)
    drug = re.sub(r'\s+[btq]\.?i\.?d\.?', ' ', drug) # bid, tid, qid
    drug = re.sub(r'[^\x00-\x7F]',' ', drug) # remove non ascii characters
    drug = drug.split('(')[0].split('[')[0]
    drug = drug.split('/')[0]
    drug = drug.split(';')[0].split(',')[0]
    drug = drug.split("+")[0] # for now take first, should include both
    drug = drug.split(" or ")[0] # for now take first, should include both
    drug = drug.split(" and ")[0] # for now take first, should include both
    drug = re.sub('capsule(s)?',' ',drug)
    drug = re.sub('pill(s)?',' ',drug)
    drug = re.sub('tablet[s]?',' ',drug)
    drug = re.sub('administration of ', ' ', drug)
    drug = re.sub('cream|antibiotic|gel|spray|drink|ampoule|syrup',' ',drug)
    drug = re.sub('injection(s)?',' ',drug)
    drug = re.sub(' [a-z]*[ai]te',' ',drug) # ionic compounds e.g citrate
    drug = re.sub(' [a-z]*ide',' ',drug) 
    drug = re.sub('calcium|hcl',' ',drug)
    drug = re.sub('administration of', ' ', drug)
    drug = re.sub('implant|extended release|concentrate',' ',drug)
    drug = re.sub('(in)?(\s+|^)(low(er)?|high(er)?|mid)(\s+|$)(dose(s)?)?',' ',drug)
    drug = re.sub('intravenous|oral|intranasal|intravitreal',' ',drug)
    drug = re.sub('experimental:?|given|inhibitor|double',' ',drug)
    drug = re.sub('(\s+|^)i{1,3}v?(\s+|$)',' ',drug)
    drug = " ".join(drug.split())
    return [drug.strip()]

def normalize_drug_name(drug):
    if('placebo' in drug or 'mimic' in drug):
        return ['placebo']
    # Stopwords by tem freqeuncy aanlysis
    extra_stopwords = ['bulk', 'dose', 'formulation', 'water', 'saline', 'weekly', \
                       'daily', 'monthly', 'medicated', 'non-medicated', 'weight', 'a', 'an', 'the', 'of', \
                      'normal', 'sugar', 'traditional', 'all', 'subject', 'subjects', 'treatment', 'arm', 'up' \
                       , 'to', 'group$', 'initial', 'for', 'qd', 'combination', 'fixed', 'once']
    stopwords = extra_stopwords
    drug = " " + drug + " "
    drug = re.sub('\"', " ", drug)
    drug = re.sub(':', " ", drug)
    drug = re.sub(';.*', ' ', drug)
    drug = re.sub('\((.*)\)', "( \g<1> )", drug)
    drug = re.sub('®|™', ' ', drug)
    drug = re.sub(r'([0-9]+[ ]?x[ ]?)?[0-9\.]+[ ]?(mcg|l|mg|milligram|µg|ug|ml|microgram(s)?|%|g|kg|tab.?|t)(/(mcg|mg|ml|l|µg|day|week|month|year|hour|min|sec|micrograms|milliliter|%|g|kg|tab.?|t))?(?P<end>\s+|$|-|,|\)|\()', ' \g<end> ', drug)
    drug = re.sub(r'[0-9]+[ ]?(day|week|month|year|hour|min|sec)s?', ' ', drug)
    
    drug = re.sub(r'\s+[btq]\.?i\.?d\.?', ' ', drug) # bid, tid, qid
    drug = re.sub('(capsule|pill|tablet|injection)s?',' ',drug)
    # drug methods
    drug = re.sub('intravenous|subcutaneous|oral|inhaled|intramuscular|intranasal|ophthalmic|liposomal|suspension|intravitreal|drug|dosage|administration|solution|ointment|nasal|injectable|inhalation',' ',drug)
    drug = re.sub('[ ](cream|antibiotic|gel|spray|product|troche|oros|drink|dentifrice|lotion|ampoule|tab\.?|syrup|(immediate|slow|controlled|delayed|intermediate|sustained|extended)[ -]release|(dr|cr|sr|xr|er)(\s+tab\.?)?|block?)[ |$},]',' ',drug) # delayed/controoled release dr/cr
    drug = re.sub(' [a-z]*[ai]te ',' ',drug) # ionic compounds e.g citrate
    #drug = re.sub(' [a-z]*ide',' ',drug) 
    drug = re.sub('calcium|hcl',' ',drug)
    drug = re.sub('implant|extended release|concentrate',' ',drug)
    drug = re.sub('(in)?(\s+|^)(low(er)?|high(er)?|mid|standard)(\s+|$|-)(dose(s)?)?',' ',drug)
    
    drug = re.sub('experimental( arm)?|given|double',' ',drug)
    
    drug = re.sub('[ ](following|prior) (.*)', ' ', drug)
    drug = re.sub('(\s+|^|,)i{1,3}\.?v\.??(\s+|,|$)',' ',drug)
    
    
    last = None
    pattern = '(\s+|^|,|\()(' + "|".join(stopwords)+ ')(\s+|$|\)|,)'
    #print(pattern)
    while(last != drug):
        last = drug
        drug = re.sub(pattern, ' ', drug)
    drug=drug.strip()
    
    if(len(drug) == 0):
        return 'placebo'
    
    # Split by components
    drug = re.sub('\s+', ' ', drug)
    input2 =drug
    p = re.compile('[ ](?:and|\+|plus|with|or)|&|,|/[^0-9]')
    p2 = re.compile('(.*)[\(\[](.*)[\]\)]')
    drugs = p.split(drug)
    n_drugs = []
    for x in drugs:
        x = x.strip()
        if(not x):
            continue
        for y in p2.split(x):
            if(not y):
                continue
            y = y.strip(' ()-')
            if(not y):
                continue
            n_drugs.append(y)
            
    
    return n_drugs
def populateDrugId(trial, drug_id_map):
    def getMatchedId(drug_name, clean=True):
        drug_ids = drug_id_map[drug_id_map['name'] == drug_name]
        if(len(drug_ids) == 1):
            return drug_ids['id'].values[0], False
        elif(len(drug_ids) > 0):
            priority_order = ['primary', 'synonyms', 'international_brand_names', \
                'external-identifiers', 'product_names', 'mixture_names']
            for category in priority_order:
                drug_ids_cat = drug_ids[drug_ids['name-category'] == category]
                if(len(drug_ids_cat) == 1):
                    return drug_ids_cat['id'].values[0], False
                elif(len(drug_ids_cat) > 1):
                    return drug_ids_cat['id'].values[0], True
        elif(clean):
            for clean_name in normalize_drug_name(drug_name):
                matched_id, dup_match = getMatchedId(clean_name, False)
                if(matched_id):
                    return matched_id, dup_match
        return None, False

    interventions = trial['intervention']
    total = 0
    matched = 0
    dup_matched = 0
    total = 0

    for intervention in interventions:
        if(intervention['intervention_type'] != 'Drug'):
            continue
        total += 1

        names = [intervention['intervention_name'].lower()] + intervention.get('other_name', [])
        
        for name in names:
            drug_id, dup_match = getMatchedId(name)
            if(drug_id is not None):
                matched += 1
                dup_matched += 1 if(dup_match) else 0
                intervention['drug_id'] = drug_id
                break
                
    trial['matched'] = matched
    trial['dup_matched'] = dup_matched
    trial['drugs_total'] = total
    trial['matched_percentage'] = float(matched * 100)/total
    return trial

def populateDiseaseId(trial, mesh_data):
    def map_conditions(condition_entries):
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
            meshIDs.add(meshID)
        
        is_healthy = False
        # try to map healthy
        healty_pattern = re.compile("healthy\s*(human)?\s*(volunteer(s)?|subject(s)?)?")
        for condition in condition_entries:
            condition = condition.lower()
            res = healty_pattern.search(condition)
            if res != None:
                is_healthy = True
                break
        return meshIDs, is_healthy, set(filter(lambda id: len(mesh_data.get_disgenetID(id, False)) > 0, meshIDs))

    trial['mesh_ids'] = []
    trial['is_healthy'] = []
    trial['mesh_ids_disgenet_filtered'] = []

    conditions = trial['condition']
    if(type(conditions) != list or not conditions):
        return trial
    i_mesh_terms = trial.get('intervention_mesh_terms', [])
    c_mesh_terms = trial.get('condition_mesh_terms', [])
    if(type(i_mesh_terms) != list):
        i_mesh_terms = []
    if(type(c_mesh_terms) != list):
        c_mesh_terms = []

    mesh_ids, is_healthy, filtered_mesh_ids = map_conditions(conditions+i_mesh_terms+c_mesh_terms)
    disgenet_ids = [x for x in map(lambda x: mesh_data.get_disgenetID(x, False), filtered_mesh_ids)]
    dis_ids = mesh_data.cut_mesh_tree(mesh_ids, 4)
    filtered_dis_ids = mesh_data.cut_mesh_tree(filtered_mesh_ids, 4)
    trial['mesh_ids'] = mesh_ids
    trial['is_healthy'] = is_healthy
    trial['mesh_ids_disgenet_filtered'] = filtered_mesh_ids
    trial['disgenet_ids'] = disgenet_ids
    trial['dis_ids'] = dis_ids
    trial['filtered_dis_ids'] = filtered_dis_ids
    return trial

def populate_drug_and_disease(x, mesh_data, drug_data):
    return populateDiseaseId(populateDrugId(x, drug_data), mesh_data)

def matchAll(dest_root, prefix, start, count):
    start_time = time.process_time()
    data = pd.read_pickle('/dfs/scratch2/prabhat8/trials-data/trial_data/data_all.pkl')
    drug_path = '/dfs/scratch2/prabhat8/trials-data/drug_data/parsed_data/drug_id_map_all.pkl'
    drug_trials = get_drug_trials(data)
    #print("Drug trials: ", len(drug_trials))
    drug_data = get_drug_data(drug_path)
    mesh_data = get_disease_data('/dfs/scratch2/prabhat8/trials-data/disease_data')
    #print("Mesh data loaded")
    #print("Data load time: ", time.process_time() -start_time)
    print('Started: ', start)

    start_time = time.process_time()
    matched_drug_trials = drug_trials[start:start+count].apply(lambda x: populate_drug_and_disease(x, mesh_data, drug_data), axis=1)
    print("Work ime taken: ", start, time.process_time() -start_time)
    # print(matched_drug_trials['matched_percentage'].value_counts())
    # all_drugs_matched = matched_drug_trials[matched_drug_trials['matched_percentage'] >= 99]
    # print("All drugs matched: ", len(all_drugs_matched))
    # disease_matched = all_drugs_matched[all_drugs_matched['dis_ids'].apply(len) > 0]
    # print("All drugs and diseases matched: ", len(disease_matched))
    matched_drug_trials.to_pickle(dest_root + "/" + prefix + "_" + str(start) +'.pkl')


def concat(dest_root, prefix):
    dfs_names = glob.glob(dest_root + '/' + prefix + '*.pkl')
    dfs = list(map(pd.read_pickle, dfs_names))
    data = pd.concat(dfs, axis=0, sort=True)
    data = data.reset_index()
    data.to_pickle(os.path.join(dest_root, prefix + '_all.pkl'))
    
def matchAllTrials(dest_root):
    batch_size = 1000
    
    prefix = 'drug_dis_matched_trials'
    num_trials = 129315
    # parseDrugs(schema_path, data_path, dest_root, 0, 10)
    with multiprocessing.Pool(processes=60) as pool:
        res = []
        for start in range(0, num_trials, batch_size):
            res.append(pool.apply_async(matchAll, (dest_root, prefix, start, batch_size)))
        for i, x in enumerate(res):
            x.get()
            print("{%d} of {%d} completed"%(i+1, len(res)))
    concat(dest_root, prefix)

if __name__=='__main__':
    dest_root = '/dfs/scratch2/prabhat8/trials-data/trial_data/matched_data_3/'
    matchAllTrials(dest_root)
