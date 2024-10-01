import json
import os
import re

from tqdm import tqdm


class MedexInputGenerator:
    def __init__(self, trial_df, out_basepath, batch_size):
        self.trial_df = trial_df
        os.makedirs(out_basepath)
        self.out_basepath = out_basepath
        self.batch_size = batch_size
        self.cnt = 0

    def _save(self, results):
        with open(os.path.join(self.out_basepath, f'medex_input_{self.cnt}.json'), 'w') as f:
            json.dump(results, f)
        self.cnt += 1

    def generate(self):
        results = {}
        for idx, trial in tqdm(self.trial_df.iterrows(), total=len(self.trial_df)):
            _generate_medex_inputs(trial, results)

            if len(results) > self.batch_size:
                self._save(results)
                results = {}
        if len(results) > 0:
            self._save(results)


def _generate_medex_inputs(trial, inputs):
    nct = trial['nct_id']
    arm_groups = trial['arm_group']
    if type(arm_groups) == list:
        for idx, arm_group in enumerate(arm_groups):
            label = arm_group['arm_group_label'].lower()
            description = arm_group.get('description', '').lower()
            if label not in description:
                description = label + " " + description
            key = '{}_arm_{}.txt'.format(nct, idx)
            text = "medication: " + label + "\n" + "medication: " + description
            inputs[key] = re.sub(r'[^\x00-\x7F]', ' ', text)

    for idx, intervention in enumerate(trial['intervention']):
        if intervention['intervention_type'] not in ['Drug', 'Other']:
            continue
        drug_name = intervention['intervention_name']
        drug_description = intervention.get('description', '')

        # Drug names
        key = '{}_drug_{}.txt'.format(nct, idx)
        description = drug_description.lower()
        if drug_name.lower() not in description:
            description = drug_name.lower() + " " + description
        text = "medication: " + drug_name.lower() + "\n" + "medication: " + description
        inputs[key] = re.sub(r'[^\x00-\x7F]', ' ', text)

        # Othernames
        key = '{}_drug_{}_othernames.txt'.format(nct, idx)
        val = ""
        for other_name in intervention.get('other_name', []):
            name = other_name.lower()
            text = "medication: " + name.lower() + "\n"
            val += re.sub(r'[^\x00-\x7F]', ' ', text)
        inputs[key] = val

    clinical_results = trial['clinical_results']
    if type(clinical_results) == float or 'reported_events' not in clinical_results:
        return
    reported_events = clinical_results['reported_events']
    group_list = reported_events['group_list']
    for idx, group in enumerate(group_list['group']):
        title = group.get('title', '').lower()
        description = group.get('description', '').lower()

        inputs[f'{nct}_rarm_{idx}_title.txt'] = re.sub(r'[^\x00-\x7F]', ' ', title)
        inputs[f'{nct}_rarm_{idx}_title_desc.txt'] = re.sub(r'[^\x00-\x7F]', ' ', title + " " + description)
