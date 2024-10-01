import random
import re

from nltk.tokenize.treebank import TreebankWordTokenizer
import random
import re

from nltk.tokenize.treebank import TreebankWordTokenizer

basedir = '/afs/cs.stanford.edu/u/prabhat8/dfs/trials-data/rarm_medex_output'
count = 0

from .medex_drug_extract import DrugMatcher, get_name_medex_key_map, is_placebo

tokenizer = TreebankWordTokenizer()


def normalize(s):
    s = re.sub(r'[^\x00-\x7F]', '', s)
    tokens = tokenizer.tokenize(s)
    s = ""
    for token in tokens:
        if token == "plus" or token == 'and' or token == '&':
            token = "+"
        s += token + " "
    return s.strip()


def findstem(arr):
    # Determine size of the array
    n = len(arr)

    # Take first word from array
    # as reference
    s = arr[0]
    l = len(s)

    res = ""

    for i in range(l):
        for j in range(i + 1, l + 1):

            # generating all possible substrings
            # of our reference string arr[0] i.e s
            stem = s[i:j]
            matched = True
            for k in range(1, n):

                # Check if the generated stem is
                # common to all words
                if stem not in arr[k]:
                    matched = False
                    break

            # If current substring is present in
            # all strings and its length is greater
            # than current result
            if matched and len(res) < len(stem):
                res = stem

    return res


def match_by_substring(title, arm_labels):
    for arm_label in arm_labels:
        words_title = set(title.split(" "))
        words_arm = set(arm_label.split(" "))

        common = words_title.intersection(words_arm)
        union = words_title.union(words_arm)

        diff = union - common

        print(title, "|", arm_label, diff, common, union)

        if len(diff) <= 1:
            print(diff)


def set_compare(title_set, arm_set):
    allowed_words = {'(', ".", ")", ":", "+", "/", "cohort", 'arm', 'group', 'phase', 'currently', 'subjects',
                     'prescriptions', 'study', 'base', 'treatment', '', 'mg', ',', '-', 'then', 'dose', 'control',
                     'period', 'g', 'part', 'double-blind', 'alone', 'total', 'to', 'active', 'first', 'oral', 'only',
                     'participants', 'follow-up', 'open-label', 'gel', 'patients', 'experiments', 'kg', 'day', '%',
                     'randomized', 'intervention', 'pill', 'followed', 'by', 'mcg', 'drug', 'of', 'er'}

    if "".join(sorted(title_set)) == "".join(sorted(arm_set)):
        return True
    common = title_set.intersection(arm_set)
    union = title_set.union(arm_set)

    diff = union - common
    # print(diff)
    for d in diff:
        if d in allowed_words:
            continue
        return False
    return True


def compare_custom(title, arm_labels, arm_drugs):
    # Without order
    def extract_bracket(x):
        m = re.finditer(r"\(([^)]*)\)", x)
        return {x.group(1) for x in m}

    title_set = set(re.split(r"[ /+]", title))
    if ":" in title:
        title = title.split(":")[1].strip()
    title = re.sub(r"subjects receiving", "", title).strip()

    for arm_label, arm_id in arm_labels.items():
        if arm_label == title:
            return arm_id
        arm_set = set(re.split(r"[ /+]", arm_label))

        if title_set == arm_set:
            return arm_id
        if set_compare(title_set, arm_set):
            return arm_id
        if ":" in arm_label:
            x = arm_label.split(":")[1].strip()
            if x == title:
                return arm_id

        if not arm_drugs[arm_id]['incomplete'] and not arm_drugs[arm_id]['non_drug']:
            tdrugs = {x.strip() for x in title.split("+")}
            drugs = set([x.lower() for x in arm_drugs[arm_id]['drugs']])
            if drugs == tdrugs:
                return arm_id
            inames = set([x.lower() for x in arm_drugs[arm_id]['iname']])
            if inames == tdrugs:
                return arm_id
            inames_sh = set()
            for x in arm_drugs[arm_id]['iname']:
                inames_sh.update(extract_bracket(x.lower()))
            if inames_sh == tdrugs:
                return arm_id

    return -1


match_medex_cnt = 0
match_medex_cnt_d = 0


class ResultArmMatcher:
    def __init__(self, matcher: DrugMatcher, intervention_col_name):
        self.matcher = matcher
        self.intervention_col = intervention_col_name
        pass

    def match(self, row):
        matcher = self.matcher
        global match_medex_cnt, match_medex_cnt_d
        nct_id = row['nct_id']

        # Add nodes for arms
        arms = row['arm_group']
        if type(arms) != list:
            arms = [{'arm_group_label': 'default', 'arm_group_type': ''}]

        arm_labels = dict()
        arm_labels_non_normalized = dict()
        arm_description = dict()
        arm_drugs = dict()
        for idx, arm in enumerate(arms):
            label = arm['arm_group_label'].lower()
            arm_labels_non_normalized[label] = idx
            label = normalize(label)
            arm_labels[label] = idx
            arm_drugs[idx] = {'drugs': set(), 'drug_ids': set(), 'iname': set(), 'incomplete': False, 'non_drug': False}
            arm_type = arm['arm_group_type']
            description = arm.get('description', "")
            if description:
                arm_description[description] = idx
        arm_labels_nospace = {re.sub(r"[ \-]", "", arm_label): v for arm_label, v in arm_labels.items()}
        arm_description_nospace = {re.sub(r"[ \-]", "", arm_desc): v for arm_desc, v in arm_description.items()}

        interventions = row[self.intervention_col]
        if type(interventions) != list:
            interventions = []
        all_drugs = set()
        for intervention in interventions:
            arm_group_label = intervention.get('arm_group_label', ['default'])
            if type(arm_group_label) != list:
                arm_group_label = ['default']
            drug_ids = intervention.get('drug_ids', set())
            is_incomplete = len(drug_ids) == 0 or intervention['is_incomplete']
            non_drug = intervention['intervention_type'] not in ['Drug', 'Placebo']

            for arm_label in arm_group_label:
                arm_id = arm_labels_non_normalized[arm_label.lower()]
                arm_drugs[arm_id]['incomplete'] |= is_incomplete
                arm_drugs[arm_id]['non_drug'] |= non_drug
                arm_drugs[arm_id]['iname'].add(intervention['intervention_name'])
            all_drugs.update(drug_ids)
            for drug_id in drug_ids:
                if drug_id == 'D#######':
                    label = 'placebo'
                else:
                    try:
                        label = matcher.drug_data[matcher.drug_data.primary_id == drug_id]['name'].tolist()[0]
                    except Exception as e:
                        print(drug_id)
                        raise e
                for arm_label in arm_group_label:
                    arm_id = arm_labels_non_normalized[arm_label.lower()]
                    arm_drugs[arm_id]['drugs'].add(label)
                    arm_drugs[arm_id]['drug_ids'].add(drug_id)

        # Adverse Events
        clinical_results = row['clinical_results']
        if type(clinical_results) != dict or 'reported_events' not in clinical_results:
            return True

        reported_events = clinical_results['reported_events']

        group_list = reported_events['group_list']
        group_id2label = {}
        titles = []

        if len(group_list['group']) == len(arm_labels) and len(arm_labels) == 1:
            group_list['group'][0]['arm_id'] = 0
            return True

        all_non_drug = True
        for arm, val in arm_drugs.items():
            if not val['non_drug']:
                all_non_drug = False

        row['all_arm_non_drug'] = all_non_drug

        # find common substring in all titles
        for idx, group in enumerate(group_list['group']):
            title = group.get('title', '').lower()
            title = normalize(title)
            titles.append(title)

        title_rest = []
        stem = findstem(titles)
        titles_stemmed = [re.sub(re.escape(stem), "", x) for x in titles]
        titles_stemmed_ns = [re.sub(r"[ \-]", "", x) for x in titles_stemmed]

        medex_outputs = {}
        for idx, group in enumerate(group_list['group']):
            id = group['@group_id']
            title = group.get('title', '').lower()
            title = normalize(title)
            title_nospace = re.sub(r"[ \-]", "", title)

            if len(title) == 0:
                continue
            # print(title, arm_labels)
            if title in arm_labels:
                group['arm_id'] = arm_labels[title]
                continue
            if title_nospace in arm_labels_nospace:
                group['arm_id'] = arm_labels_nospace[title_nospace]
                continue

            if title in arm_description:
                group['arm_id'] = arm_description[title]
                continue
            if title_nospace in arm_description_nospace:
                group['arm_id'] = arm_description_nospace[title_nospace]
                continue

            if titles_stemmed[idx] in arm_labels:
                group['arm_id'] = arm_labels[titles_stemmed[idx]]
                continue
            if titles_stemmed_ns[idx] in arm_labels_nospace:
                group['arm_id'] = arm_labels_nospace[titles_stemmed_ns[idx]]
                continue

            if titles_stemmed[idx] in arm_description:
                group['arm_id'] = arm_description[titles_stemmed[idx]]
                continue
            if titles_stemmed_ns[idx] in arm_description_nospace:
                group['arm_id'] = arm_description_nospace[titles_stemmed_ns[idx]]
                continue

            tit2 = title.replace('cystic fibrosis', 'cf')
            if tit2 in arm_labels:
                group['arm_id'] = arm_labels[tit2]
                continue
            if tit2 in arm_description:
                group['arm_id'] = arm_description[tit2]
                continue
            arm_id = compare_custom(title, arm_labels, arm_drugs)
            if arm_id != -1:
                # print("comapre cusotm ", title, arm_labels)
                group['arm_id'] = arm_id
                continue

            found = False
            for arm_label, arm_id in arm_labels.items():
                if re.match(r"(arm|regimen|phase|group|cohort|treatment|panel) ([i]+|[\da-z])( [\w])?$", title):
                    if title + " " in arm_label:
                        found = True
                        group['arm_id'] = arm_id
                        break
                if (re.match(r"(arm|regimen|phase|group|cohort|treatment|panel) ([i]+|[\da-z])( [\w])?$",
                             arm_label) and arm_label + " " in title):
                    found = True
                    group['arm_id'] = arm_id
                    break
            if found:
                continue

            # print("Substring: ", match_by_substring(title, arm_labels))
            if 'medex_out_title' not in group:
                print(f'{nct_id}_rarm_{idx}_title.txt')
                medex_outputs[title] = {}
                continue

            medex_out = group['medex_out_title']
            drug_details = []
            # pprint(medex_out)
            drug_ids = set()
            all_drugs_ids = True
            medex_key_map = get_name_medex_key_map(medex_out, title)
            for drug_name, key in medex_key_map:
                if is_placebo(drug_name):
                    dbid = {'D#######'}
                else:
                    _, dbid = matcher.get_rxnorm(medex_out[key], drug_name)
                if not dbid:
                    all_drugs_ids = False
                    if random.random() < 0.01:
                        print(drug_name, key, title)
                    continue
                drug_ids.update(dbid)
            for drug_name, drug_infos in medex_out.items():
                for drug_info in drug_infos:
                    for field in ['drug_name', 'strength', 'frequency']:
                        if drug_info[field]:
                            drug_details.append(drug_info[field].split("(")[0])

            extracted_details = "".join(sorted(" ".join(drug_details).split(" ")))
            title_sorted = "".join(sorted(title.split(" ")))
            group['drug_details'] = medex_out
            group['drug_ids'] = drug_ids
            group['all_drugs_matched'] = all_drugs_ids
            group['has_extra_drugs'] = not drug_ids.issubset(all_drugs)
            # print(extracted_details, title_sorted, medex_out)
            if (extracted_details == title_sorted or set_compare(set(title.split(" ")),
                                                                 set(" ".join(drug_details).split(" ")))):
                # print("yeah matched", drug_details, title, arm_labels)
                group['drug_detail_matches'] = True
                continue
            elif len(drug_ids) > 0 and all_drugs_ids:
                match_medex_cnt += 1
                for arm_id, arm_info in arm_drugs.items():
                    if arm_info['drug_ids'] == drug_ids:
                        match_medex_cnt_d += 1

            medex_outputs[title] = medex_out
        #     if(len(medex_outputs) <= 1 and len(group_list['group']) == len(arm_labels)):
        #         return True
        #     # if arms labels are 1, 2, or a, b then match using drug names
        #     is_single_letter = True
        #     for arm_label in arm_labels:
        #         if(len(arm_label) > 2):
        #             is_single_letter = False

        #     if(is_single_letter):
        #         return True

        if len(medex_outputs) == 0:
            return True
        return False

        # print("Unmatched: ", list(medex_outputs.keys()))
        # pprint(arm_labels)
        # print(nct_id)
        #
        # print(titles, arm_labels)
        # pprint(arm_drugs)
        # tp = []
        # for intervention in interventions:
        #     tpi = {}
        #     for key, value in intervention.items():
        #         if key not in ['dose_details', 'medex_othernames', 'medex_out', 'description']:
        #             tpi[key] = value
        #     tp.append(tpi)
        # # pprint(tp)
        # print(nct_id)
        # return False

#
# for idx, row in tqdm(df_results.iterrows()):
#     match_by_number(row)
#
# print(match_medex_cnt, match_medex_cnt_d)


# df_results['all_result_arm_matched'] = df_results.apply(match_by_number, axis=1)
