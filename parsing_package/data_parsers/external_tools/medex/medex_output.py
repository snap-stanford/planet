import glob
import json
import os
from collections import defaultdict


class MedexOutputParser:
    def __init__(self, base_paths):
        self._load_output(base_paths=base_paths)
        self.base_paths = base_paths

    def _load_output(self, base_paths):
        outputs = {}
        for base_path in base_paths:
            for out_file in glob.glob(os.path.join(base_path, "*.json")):
                with open(out_file, 'r') as f:
                    out_chunk = json.load(f)
                outputs.update(out_chunk)

        outputs_grouped = defaultdict(dict)
        for k, v in outputs.items():
            nct_id = k.split("_")[0]
            subname = "_".join(k.split("_")[1:])[:-4]
            outputs_grouped[nct_id][subname] = v.split("\n")
        self.medex_outputs = {k: v for k, v in outputs_grouped.items()}

    def fill_medex_info(self, trial):
        nctid = trial['nct_id']
        if nctid not in self.medex_outputs:
            print(nctid)
            return trial
        medex_data_all = []
        for _, v in self.medex_outputs[nctid].items():
            medex_data_all.extend(v)
        trial['medex_raw'] = self.medex_outputs[nctid]
        trial['medex_processed'] = parseMedexOutput(medex_data_all)
        return self._extract_names(trial)

    @staticmethod
    def _extract_names(trial):
        interventions = trial['intervention']

        if type(interventions) != list:
            return trial
        medex_raw = trial['medex_raw']
        for idx, intervention in enumerate(interventions):
            if intervention['intervention_type'] != 'Drug':
                continue
            intervention['medex_out'] = parseMedexOutput(medex_raw[f'drug_{idx}'])
            if intervention.get('other_name', []):
                intervention['medex_othernames'] = parseMedexOutput(medex_raw[f'drug_{idx}_othernames'])

        arm_groups = trial['arm_group']
        if type(arm_groups) != list:
            arm_groups = []
        for idx, arm_group in enumerate(arm_groups):
            arm_group['medex_out'] = parseMedexOutput(medex_raw[f'arm_{idx}'])

        clinical_results = trial['clinical_results']
        if type(clinical_results) == float or 'reported_events' not in clinical_results:
            return trial
        reported_events = clinical_results['reported_events']
        group_list = reported_events['group_list']

        for idx, group in enumerate(group_list['group']):
            if f'rarm_{idx}_title' not in medex_raw:
                print(trial['nct_id'], group.get('ttile', ''), idx)
                group['medex_out_title'] = {}
            else:
                group['medex_out_title'] = parseMedexOutput(medex_raw[f'rarm_{idx}_title'])
            group['medex_out_title_desc'] = parseMedexOutput(medex_raw[f'rarm_{idx}_title_desc'])
        return trial


def mergeDrugInfos(infos):
    result = {}
    for info in infos:
        drug_name = info['drug_name']
        if drug_name not in result:
            result[drug_name] = [info]
        else:
            old_infos = result[drug_name]
            # Try to merge with an existing info if no contracdicting info found
            merged = False
            for old_info in old_infos:
                can_merge = True
                for key in old_info:
                    if (key not in ['drug_form', 'strength', 'dose_amount', 'route', 'frequency', 'duration',
                                    'necessity']):
                        continue
                    if len(old_info[key]) > 0 and len(info[key]) > 0:
                        v1 = old_info[key]
                        v2 = info[key]
                        is_equal = v1 == v2 or v1 in v2 or v2 in v1
                        if not is_equal:
                            can_merge = False
                            break
                if can_merge:
                    merged = True
                    for key in old_info:
                        v1 = old_info[key]
                        v2 = info[key]
                        if len(v2) > len(v1):
                            old_info[key] = info[key]
                    break
            if not merged:
                result[drug_name].append(info)
                # print(result[drug_name])
    return result


# steeming for stemming
from stemming.porter2 import stem


def parseMedexOutput(output):
    result = {}
    infos = []
    for line in output:
        line = line.strip()
        if line == "":
            continue
        tags = line.split('|')
        sent_id, sent_text = tags[0].split("\t")
        drug_name = tags[1].split("[")[0].lower()
        drug_form = " ".join(stem(word) for word in tags[3].split("[")[0].lower().split())

        strength = tags[4].split("[")[0].lower()
        dose_amount = tags[5].split("[")[0].lower()
        route = tags[6].split("[")[0].lower()
        frequency = tags[7].split("[")[0].lower()
        duration = tags[8].split("[")[0].lower()
        all_info = {
            "drug_name": drug_name,
            "brand_name": tags[2].split("[")[0].lower(),
            "drug_form": drug_form,
            "strength": strength,
            'dose_amount': dose_amount,
            "route": route,
            "frequency": frequency,
            'duration': duration,
            "necessity": tags[9].split("[")[0].lower(),
            "umls_cui": tags[10].split("[")[0].lower(),
            "rxnorm_cui": tags[11].split("[")[0].lower(),
            "rxnorm_cui_generic_name": tags[12].split("[")[0].lower(),
            "generic_name": tags[13].split("[")[0].lower(),
        }
        infos.append(all_info)

    return mergeDrugInfos(infos)
