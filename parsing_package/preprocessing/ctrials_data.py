'''
Created on Nov 17, 2018

@author: maria
'''

from enum import Enum
import pandas as pd
import re
from collections import Counter

class ClinicalTrial():
    
    def __init__(self, nctID, drugIDs, disIDs, status, other_info, all_drugs_mapped):
        self.nctID = nctID
        self.drugIDs = drugIDs
        self.disIDs = disIDs
        self.status = TrialStatus.get_status(status)
        self.other_info = other_info
        self.all_drugs_mapped = all_drugs_mapped
        self.annotation = None # positive or negative class
        
    def assign_annotation(self, annotation):
        self.annotation = annotation
        
    def get_field_values(self, field_info):
        if field_info==TrialField.status:
            return [self.status.value]
        else:
            values = self.other_info.get(field_info.value)
            if isinstance(values, list):
                return values
            else:
                return [values]
            
    def get_phase(self, bound_lower=True):
        """Returns integer value of phase in a trial.
        bound_lower: if True takes lower value for trials which are between two phases, 
        otherwise higher
        """
        from builtins import int
        phase = self.other_info.get(TrialField.phase.value)
        if phase:
            phase = phase[0]
            if phase[:5]=='Early':
                return int(phase[-1])
            if bound_lower:
                return int(phase[6])
            else:
                return int(phase[-1])
        return None
    
    def is_predecessor(self, other_trial, check_phase=False, check_drug=False,
                     check_disease=True, check_location=True, check_year=True):
        """Checks if current trial is_predecessor of other trial.
        return: True if trial is predecessor; False otherwise
        """
        
        if check_phase:   
            if self.get_phase()+1!=other_trial.get_phase():
                return False
        
        if check_drug:
            trial_drugs = ''.join(sorted(self.drugIDs))
            other_drugs = ''.join(sorted(other_trial.drugIDs))
            if not trial_drugs==other_drugs:
                return False
        
        if check_location:
            trial_loc = set(self.get_field_values(TrialField.location_countries))
            other_loc = set(other_trial.get_field_values(TrialField.location_countries))
            
            if not trial_loc.intersection(other_loc):
                return False
          
        if check_year:
            trial_end_year = self.get_completion_year()
            other_start_year = other_trial.get_start_year()
            if not trial_end_year or not other_start_year:
                return False
            if trial_end_year>other_start_year:
                return False
            
        if check_disease:
            trial_dis = set(self.disIDs)
            other_dis = set(other_trial.disIDs)
            if not trial_dis.intersection(other_dis):
                return False
            #TODO intersection with parent node?
        return True
    
    def get_start_year(self):
        """Returns start year of a trial. None if start year is not specified."""
        month_year = self.other_info.get(TrialField.start_date.value)
        if month_year:
            month_year = month_year[0]
            return int(month_year[-4:])
        return month_year
        
    def get_completion_year(self):
        """Returns completion year of a trial. None if completion year is not specified."""
        month_year = self.other_info.get(TrialField.completion_date.value)
        if month_year:
            month_year = month_year[0]
            return int(month_year[-4:])
        return month_year
            
    def __str__(self):
        trial2string = 'NCT ID: {}\n'.format(self.nctID)
        trial2string += 'Conditions: {}\n'.format(','.join(self.disIDs))
        trial2string += 'Drugs: {}\n'.format(','.join(self.drugIDs))
        trial2string += 'Status: {}\n\n'.format(self.status.value)
        return trial2string
        
class TrialStatus(Enum):
    
    completed = "completed"
    terminated = "terminated"
    withdrawn = "withdrawn"
    suspended = "suspended"
    active = "active, not recruiting"
    available = "available"
    recruiting = "recruiting"
    unknown = "unknown status"
    not_available = "no longer available"
    enrolling_invitation = "enrolling by invitation"
    not_recruiting = "not yet recruiting"
    approved = "approved for marketing"
    tan = "temporarily not available"
    
    @classmethod
    def get_status(cls, status_string):
        """Returns TrialStatus based on string value."""
        return {member.value:member for _, member in cls.__members__.items()
                }.get(status_string.lower().strip())
    
    @classmethod        
    def get_negative_status(cls):
        return [TrialStatus.terminated, TrialStatus.withdrawn, TrialStatus.suspended]
    
class TrialField(Enum):
    
    #TODO add numerical fields but they can't be used for counting frequency
    #TODO edit year
    
    status = 'status'
    phase='phase'
    lead_sponsor_name='lead_sponsor_name'
    lead_sponsor_class='lead_sponsor_class'
    start_date='start_date'
    completion_date='completion_date'
    duration='duration'
    enrollment='enrollment'
    facility_name='facility_name'
    facility_city='facility_city'
    facility_state='facility_state'
    facility_country='facility_country'
    max_age='maximum_age'
    min_age='minimum_age'
    gender='gender'
    location_countries='location_countries'
    
    @classmethod        
    def get_numerical_fields(cls):
        return [TrialField.enrollment, TrialField.duration, TrialField.min_age, TrialField.max_age]
                
                
class ClinicalTrialsData(dict):
    """Stores all clinical trials with basic information as dictionary."""
    
    def __init__(self, ct_file, other_fields, mesh_data, drugbank_data, 
                 include_healthy=False, min_mesh_level=4, filter_disgenet=True):
        """
        ct_file: csv file containing clinical trials data
        other_fields: list of other TrialField defining data that should be included in describing
                    trial properties
        drugbank_data: data from DrugBank containing mappings of drugs to IDs
        mesh_data: data from MeSH hierarchy. Contains disease IDs, entry terms and hieararchy structure
        include_healthy: include healthy as condition 
        min_mesh_level: map all mesh terms with lower level to min_level parent
        filter_disgenet: use only MeSH term covered in DisGeNET
        """
        super(ClinicalTrialsData, self).__init__()
        data = pd.read_csv(ct_file)
       
        other_fields = [field.value for field in other_fields
                         if field!=TrialField.status] # status is included by default
        data = data.loc[:,['nct_id','condition','mesh_terms','drug_name','drug_other_name','status']
                        +other_fields]
        data.dropna(how='all', subset=other_fields)
        self.mesh_data = mesh_data
        self.drugbank_data = drugbank_data
        self.other_fields = other_fields
        self.build_trials_data(data, include_healthy, min_mesh_level, filter_disgenet)
        
    def build_trials_data(self, data, include_healthy, min_mesh_level, filter_disgenet):
        """Builds dict of clinical trials.
        data: data frame containing clinical trials as rows
        include_healthy: include healthy as condition 
        min_mesh_level: map all mesh terms with lower level to min_level parent
        filter_disgenet: use only MeSH term covered in DisGeNET
        """
        num_mapped = 0
        
        for _,rows in data.iterrows():
            mapped = False
            nctID = rows['nct_id']
            
            drugs = self._str_to_list(rows['drug_name'])
            other_names = dict(zip(drugs, self._str_to_list(rows['drug_other_name'])))
            drugIDs = []
            all_drugs_mapped = True
            for drug in drugs:
                if "placebo" in drug.lower(): 
                    continue # not counting this in mapped
                drugID = self._map_drug(drug, other_names)
                if drugID:
                    drugIDs.append(drugID)
                else:
                    all_drugs_mapped = False
            
            conditions = rows['condition']
            mesh_terms = rows['mesh_terms']
            conditions = self._str_to_list(conditions)
            if not conditions:
                continue
            mesh_terms = self._str_to_list(mesh_terms)
            disIDs = self._map_conditions(conditions+mesh_terms, include_healthy, filter_disgenet)
               
            if min_mesh_level!=None:
                disIDs = self.mesh_data.cut_mesh_tree(disIDs, min_mesh_level)
            
            other_info = {}
            for other_field in self.other_fields:
                values = rows[other_field]
                if isinstance(values, float):
                    continue
                values = self._str_to_list(values)
                other_info[other_field] = values
            
            if disIDs and drugIDs:
                #if rows['status'].lower()=='withdrawn' or rows['status'].lower()=='terminated' or rows['status'].lower()=='suspended' or rows['status'].lower()=='completed':
                self[nctID] = ClinicalTrial(nctID, drugIDs, disIDs, rows['status'], 
                                            other_info, all_drugs_mapped)
                
            if mapped:
                num_mapped+=1               
        #print("Mapped trials: {}/{}".format(num_mapped, data.shape[0]))
                        
            
    def _map_drug(self, drug_name, other_names):
        """"
        For given drug name finds drugbank id. If ID is not found, returns none.
        drug_name: original drug name in clinical trials
        other_names: other names from clinical_trials separated by |
        drugbank_data: data from DrugBank containing mappings of drugs to IDs
        """
        ID = self.drugbank_data.get_ID(drug_name)
        if ID!=None:
            return ID
        
        #otherwise try to map other names
        if not drug_name in other_names:
            return None
        
        other_names = other_names[drug_name]
        other_names = [o for o in other_names.split("|")]
        for synonym in other_names:
            ID = self.drugbank_data.get_ID(synonym)
            if ID!=None:
                return ID
        return None
    
    def get_frequency(self, field_value):
        """Returns frequency of each possible value in a given field. For example, if field_value
        is location_countries, returns number of times is location is appearing in the data."""
        freq_values = Counter()
        
        for nct_id in self.keys():
            trial = self[nct_id]
            if field_value==TrialField.status:
                values = trial.status.value
            else:
                values = trial.other_info.get(field_value)
            if not values:
                continue
            if isinstance(values, list):
                if len(values)>10:
                    continue
                for v in values:
                    freq_values.update([v])
            else:
                freq_values.update([values])
        return freq_values
    
    def get_annotated_trials(self):
        """Returns subset of trials with annotation."""
        annotated = {}
        for nctid, trial in self.items():
            if trial.annotation!=None:
                annotated[nctid] = trial
        return annotated
   
    def _map_conditions(self, condition_entries, include_healthy, filter_disgenet):
        """"
        condition_entries: original condition names and mesh terms
        mesh_data: data from MeSH hierarchy. Contains disease IDs, entry terms and hieararchy structure
        include_healthy: include healthy as condition 
        filter_disgenet: use only MeSH term covered in DisGeNET
        return MeshIDs of conditions
        """
        meshIDs = set()
        for condition in condition_entries:
            meshID = self.mesh_data.get_meshID(condition)
            if meshID==None:
                continue
            if filter_disgenet:
                disgenetID = self.mesh_data.get_disgenetID(meshID, propagate=False)
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
    
    def _str_to_list(self, text):
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
        
