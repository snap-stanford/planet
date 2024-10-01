'''
Created on Oct 23, 2018

@author: maria

'''

import xml.etree.cElementTree as ET
import re
from matplotlib._cm_listed import name

class DrugBank():
    """Holds two dictionaries: drug2id mapping all possible drug names to drugbank ID and
    id2drug mapping ID to official drug name."""
    
    def __init__(self, drugbank_synonyms_file):
        super(DrugBank, self).__init__()
        self._read_synonyms_file(drugbank_synonyms_file)

    def _read_synonyms_file(self, drugbank_synonyms_file):
        """
        Returns dictionary with all drug names from Drug bank with
        asssociated drug_id.
        """
        self.drug2id = {}
        self.id2drug = {}
        with open(drugbank_synonyms_file, encoding='utf-8') as f:
            for line in f.readlines():
                names = line.split("\t")
                drugID = names[0]
                names = names[1].split("|")
                for name in names:
                    self.drug2id[name.lower().strip()] = drugID
                self.id2drug[drugID] = names[0].lower().strip()
                
    def get_ID(self, drug_name):
        """"
        drug_name: name of the drug
        Return: drugbank ID, None if drug_name could not be matched to any ID
        """
        drug_name = drug_name.lower().strip()
        if drug_name in self.drug2id:
            return self.drug2id[drug_name]
        
        all_names = []
        all_names.append(drug_name.lower())
        
        cleaned = self.clean_drug_name(drug_name)
        
        return self.drug2id.get(cleaned)
        
    def clean_drug_name(self, drug):
        """"
        If drug name is not mapped try to edit name by removing
        number, mg & parts in brackets.
        """
        drug = re.sub(r'[>≥]?\s*\d+\.?\d*\s*(mg|%|micrograms)', ' ', drug)
        drug = re.sub(r'\s+[btq]\.?i\.?d\.?', ' ', drug) # bid, tid, qid
        drug = re.sub(r'[^\x00-\x7F]',' ', drug) # remove non ascii characters
        drug = drug.split('(')[0].split('[')[0]
        drug = drug.split('/')[0]
        drug = drug.split(';')[0].split(',')[0]
        drug = re.sub('capsule(s)?',' ',drug)
        drug = re.sub('pill(s)?',' ',drug)
        drug = re.sub('tablet[s]?',' ',drug)
        drug = re.sub('cream|antibiotic|gel|spray|drink|ampoule|syrup',' ',drug)
        drug = re.sub('injection(s)?',' ',drug)
        drug = re.sub(' [a-z]*[ai]te',' ',drug) # ionic compounds e.g citrate
        drug = re.sub(' [a-z]*ide',' ',drug) 
        drug = re.sub('calcium|hcl',' ',drug)
        drug = re.sub('implant|extended release|concentrate',' ',drug)
        drug = re.sub('(in)?(\s+|^)(low(er)?|high(er)?|mid)(\s+|$)(dose(s)?)?',' ',drug)
        drug = re.sub('intravenous|oral|intranasal|intravitreal',' ',drug)
        drug = re.sub('experimental:?|given|inhibitor|double',' ',drug)
        drug = re.sub('(\s+|^)i{1,3}v?(\s+|$)',' ',drug)
        
        return drug.strip()
                    
def parse_gene_bank_synonyms(inpath, outpath):
    """
    Parse drug bank xml all data to look for synonyms and international brand names.
    Save data in csv with format drug_id, name, synonyms, international brands.
    """
    xml_tree = ET.parse(inpath)
    
    root = xml_tree.getroot()
    
    all_names = {}
    for drug_root in root:
        dbid = ""
        for drug in drug_root:
            if drug.tag=='{http://www.drugbank.ca}drugbank-id':
                if drug.attrib:
                    dbid = drug.text
                    all_names[dbid] = []
            if drug.tag=='{http://www.drugbank.ca}name':
                all_names[dbid].append(drug.text) 
            if drug.tag=='{http://www.drugbank.ca}synonyms':
                for synonym in drug:
                    synonym_name = synonym.text.strip()
                    if synonym_name:
                        all_names[dbid].append(synonym_name)
            if drug.tag=='{http://www.drugbank.ca}international-brands':
                for synonym in drug:
                    if synonym.tag=='{http://www.drugbank.ca}international-brand':
                        other_name = synonym[0].text.strip()
                        if other_name:
                            all_names[dbid].append(other_name)
    
    f = open(outpath, "w", encoding='utf-8')                    
    for k,v in all_names.items():
        vals = '|'.join(v)
        f.write('{}\t{}\n'.format(k, vals))
              
#parse_gene_bank_synonyms(/drug bank data/full database.xml", 
#                         "/data/drugbank_synonyms")    
    
