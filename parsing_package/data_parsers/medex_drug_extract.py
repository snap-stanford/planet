import json
import pickle
import re
from collections import defaultdict

import pandas as pd
import pubchempy as pcp
from nltk.tokenize.treebank import TreebankWordTokenizer, TreebankWordDetokenizer
from tqdm import tqdm

tokenizer = TreebankWordTokenizer()
detokenizer = TreebankWordDetokenizer()


def get_cleaned_names(name, key, debug=False):
    # remove non-ascii characters
    name = re.sub(r'[^\x00-\x7F]', '', name)
    name = detokenizer.detokenize(name.split(" "))
    cleaned_names = set()
    if len(name) == 0:
        return set()
    name = name.replace(" - ", "-")
    name = name.replace(" , ", ",")
    name = name.replace(" ' ", "'")
    name = re.sub(r" \)[ ]?", ")", name).strip()
    name = re.sub(r"( |^)\([ ]?", "(", name).strip()
    if name[-1] == ".":
        name = name[:-1]
    match = re.match(r"\(([^()]{2,})\)$", name)
    if match:
        name = match.group(1)
    if debug:
        print(name)
    parts = re.match(r"([^()]{2,})[ ]?\(([^()]{2,})\)", name)
    if parts:
        cleaned_names.add((parts.group(1), key))
        cleaned_names.add((parts.group(2), key))
    else:
        for name_part in re.split(r" \+( |$)| /( |$)", name):
            # print(name_part)
            if name_part and len(name_part.strip()) > 1:
                name_part = name_part.strip()
                cleaned_names.add((name_part, key))
    return cleaned_names


def clean_preifx_suffix(key):
    for suffix in ['-', '-single', '-multiple', '-severe', '-moderate', '-xr', '-sr', ".", ";", ",", " er"]:
        if key.endswith(suffix):
            key = key[:-len(suffix)].strip()
    for prefix in [":"]:
        if key.endswith(prefix):
            key = key[len(prefix):].strip()
    return key


def complete_code_incompletes(key, names):
    r = fr'( |\(|^){re.escape(key)}( |\)|$|;|,)'
    regex = fr'[^\s]*{re.escape(key)}-[^\s;]*'
    for name in names:
        if re.search(r, name.lower()):
            return clean_preifx_suffix(key)
    for name in names:
        match = re.search(regex, name.lower())
        if match:
            return clean_preifx_suffix(match.group(0))
    return key


def is_placebo(name):
    return ('placebo' in name or 'saline' in name or 'vehicle' in name) \
           and 'followed by' not in name and 'then' not in name


def handle_interferon(name, full_name):
    if ('interferon' in name or 'ifn' in name) and 'gamma' in name:
        name = 'interferon gamma-1b'

    if 'interferon' in name:
        match = re.search(r'interferon([- ])?(γ|gamma)', full_name)
        if match:
            name = 'interferon gamma-1b'

        match = re.search(r'peginterferon([ \-])(alpha|alfa|α|a)([ \-])?(2a|2b|2-a|2-b)', full_name)
        if match:
            long_name = match.group(0)
            if '2a' in long_name or '2-a' in long_name:
                name = 'peginterferon alfa-2a'
            else:
                name = 'interferon alfa-2b'

    # Some custom chekcs for frequenct
    if 'interferon' in name:
        match = re.search(
            r'(pegylated|peg)([ \-])?interferon([ \-]) ?(\(pegifn\)|\(peg-ifn\)|'
            r'\(peg ifn\))? ?(alpha|alfa|α|a)([ \-])?(2a|2-a)',
            full_name)
        if match:
            name = 'peginterferon alfa-2a'
        else:
            match = re.search(r'interferon([ \-])(alpha|alfa|α|a)([ \-])?(2b|2-b)', full_name)
            if match:
                name = 'interferon alfa-2b'
    return name


def filter_complete_substrings(medex_names):
    names = set()
    for name, key in medex_names:
        add = True
        for name2, _ in medex_names:
            if name == name2:
                continue
            if name in name2:
                add = False
        if add:
            names.add((name, key))
    return names


def get_name_medex_key_map(medex_out, full_name, skip_small_names=True):
    medex_names = set()

    for key in medex_out.keys():
        key_ = complete_code_incompletes(key, [full_name])
        medex_names.update(get_cleaned_names(key_, key))

    names = set()
    for name, key in medex_names:
        # Only consider drug names in either intervention name or other names
        if name not in full_name:
            continue
            
        name = handle_interferon(name, full_name)
        # Do not match names less than len 3 bcoz they are shorthands which are misleading in general
        if skip_small_names and len(name) < 3:
            continue
        names.add((name, key))

    # Filter out complete substrings from medex_names
    # print(medex_names)
    names = filter_complete_substrings(names)
    return names


class DrugMatcher:
    def __init__(self, data_paths):
        """
        :param data_paths: Dictionary with keys
            drug_data, drugname2dbid, rxnorm2drugbank-umls, RXNCONSO
        """
        self.data_paths = data_paths
        self.pubchem_cache = {}
        self._load_data()

    def _load_data(self):
        data_paths = self.data_paths
        self.drug_data = pd.read_pickle(data_paths['drug_data'])
        self._load_pubchem_synonyms()

        self._load_drugbank_data()

        self._load_rxnorm_data()

    def _load_pubchem_synonyms(self):
        with open(self.data_paths['pubchem_synonyms']) as f:
            dbid2synonyms = json.load(f)
        pchem_synonym2id = defaultdict(set)
        for drug_id, synonyms in dbid2synonyms.items():
            for s in synonyms:
                for syn in s['Synonym']:
                    syn = syn.strip()
                    if len(syn) <= 3:
                        continue
                    pchem_synonym2id[syn.lower()].add(drug_id)
        self.pchem_synonym2id = pchem_synonym2id

    def _get_DBId_from_pubchem(self, name):
        if name in self.pubchem_cache:
            return self.pubchem_cache[name]
        syn = pcp.get_synonyms(name, namespace='name')
        drugbank_ids = set()
        pubchem_ids = set()
        for synonym in syn:
            pubchem_ids.add(synonym['CID'])
            for syni in synonym['Synonym']:
                if syni in self.drugid2name:
                    drugbank_ids.add(syni)
        self.pubchem_cache[name] = pubchem_ids, drugbank_ids
        return pubchem_ids, drugbank_ids

    def _load_drugbank_data(self):
        drugid2name = {}
        unii2drugid = {}
        rxcui2drugid = {}
        atc2drugid = defaultdict(set)
        # make priority maps
        name2drug_id = defaultdict(lambda: defaultdict(set))
        for idx, row in tqdm(self.drug_data.iterrows()):
            drugid2name[row['primary_id']] = row['name']

            name2drug_id[row['name'].lower()]['primary'].add(row['primary_id'])

            synonyms = row['synonyms']
            if type(synonyms) != list:
                synonyms = []
            for synonym in synonyms:
                name2drug_id[synonym.lower()]['synonym'].add(row['primary_id'])
            for col in ['international_brands', 'products', 'mixtures']:
                name_list = row[col]
                # print(name_dict)
                if type(name_list) != list:
                    continue
                for name in name_list:
                    # print(name['name'], col)
                    name2drug_id[name['name'].lower().strip()][col].add(row['primary_id'])

            atc_codes = row['atc-codes']
            if type(atc_codes) != dict:
                atc_codes = {'atc-code': []}

            for code in atc_codes['atc-code']:
                atc2drugid[code['@code']].add(row['primary_id'])

            unii2drugid[row['unii']] = row['primary_id']
            salts = row['salts']
            if type(salts) != dict:
                salts = {'salt': []}
            for salt in salts['salt']:
                salt_id = ''
                for _id in salt['drugbank-id']:
                    if _id['@primary']:
                        salt_id = _id['$']
                        break
                drugid2name[salt_id] = salt['name']
                name2drug_id[salt['name'].lower()]['primary'].add(row['primary_id'])
                unii2drugid[salt['unii']] = row['primary_id']

            external_ids = row['external-identifiers-all']
            if type(external_ids) != dict:
                external_ids = {'external-identifier': []}
            for external_id in external_ids['external-identifier']:
                if external_id['resource'] == 'RxCUI':
                    rxcui2drugid[external_id['identifier']] = row['primary_id']
                name2drug_id[external_id['identifier']]['external-ids'].add(row['primary_id'])
        self.drugid2name = drugid2name
        self.unii2drugid = unii2drugid
        self.rxcui2drugid = rxcui2drugid
        self.atc2drugid = atc2drugid
        self.name2drug_id = name2drug_id

    def _dbIdsForName(self, name, use_pubchem_map=True):
        for col_pri in ['primary', 'synonyms', 'international_brands', 'products', 'mixtures', 'external-ids']:
            if self.name2drug_id[name][col_pri]:
                return self.name2drug_id[name][col_pri]
        if use_pubchem_map:
            return self.pchem_synonym2id[name]
        else:
            return set()

    def _load_rxnorm_data(self):
        rxnorm2drugbank = defaultdict(set)
        rxcui2names = defaultdict(set)
        rxcui2pref = defaultdict()
        with open(self.data_paths['RXNCONSO']) as f:
            for line in tqdm(f):
                cols = ['CUI', 'LAT', 'TS', 'LUI', 'STT', 'SUI', 'ISPREF', 'AUI', 'SAUI', 'SCUI', 'SDUI', 'SAB', 'TTY',
                        'CODE', 'STR', 'SRL', 'SUPPRESS', 'CVF']
                vals = line.strip().split("|")[:-1]
                # rint(vals, cols)
                assert len(vals) == len(cols)
                parsed = dict(zip(cols, vals))
                if parsed['ISPREF'] or parsed["SAB"] == 'RXNORM':
                    rxcui2pref[parsed['CUI']] = parsed['STR']
                rxcui2names[parsed['CUI']].add(parsed['STR'])
                if parsed['SAB'] == 'DRUNK':
                    rxnorm2drugbank[parsed['CUI']].add(parsed['CODE'])
                if parsed['SAB'] == 'ATC':
                    atc = parsed['CODE']
                    if atc in self.atc2drugid:
                        rxnorm2drugbank[parsed['CUI']].update(self.atc2drugid[code])

                if parsed['SAB'] == 'MTHSPL' and parsed['TTY'] == 'SU' and parsed['CODE'] != 'NOCODE':
                    code = parsed['CODE']
                    if code in self.unii2drugid:
                        rxnorm2drugbank[parsed['CUI']].add(self.unii2drugid[parsed['CODE']])
                    rxnorm_name = parsed['STR'].lower()
                    dids = self._dbIdsForName(rxnorm_name, use_pubchem_map=False)
                    if dids:
                        rxnorm2drugbank[parsed['CUI']].update(dids)
        self.rxnorm2drugbank = rxnorm2drugbank
        self.rxcui2pref = rxcui2pref

        with open(self.data_paths['rxnorm2drugbank-umls'], "rb") as f:
            self.rxnorm2drugbank_umls = pickle.load(f)

    def get_rxnorm(self, drug_infos, name):
        name = clean_preifx_suffix(name)
        if name == 'liposomal doxorubicin':
            name = 'doxorubicin'
        if name == 'biphasic insulin aspart':
            name = 'insulin aspart'
        if name == 'as2-1':
            name = 'astugenal'  # strangely this is not in drugbank
        if name == 'estrogen':
            name = 'estradiol'  # drugbank direct search gives this, but not found otheriwse
        if name == 'botulinum toxin':
            name = 'botulinum toxin type a'
        if name == 'bupivicaine':
            name = 'bupivacaine'
        if name == 'su011248':
            name = 'sunitinib'
        if name == 'insulin nph':
            name = 'nph insulin'
        if name == 'regular insulin':
            name = 'insulin regular'
        if name == 'recombinant interferon alfa-2b':
            name = 'interferon alfa-2b, recombinant'
        rxnorm = ''
        dbid = ''
        for info in drug_infos:
            if len(info['rxnorm_cui_generic_name']) > 0:
                if rxnorm and rxnorm != info['rxnorm_cui_generic_name']:
                    rxnorm = ''
                    break
                rxnorm = info['rxnorm_cui_generic_name']
        # replace this with a smaller map
        dbids = self._dbIdsForName(name)

        if len(dbids) == 1:
            dbid = dbids

        if rxnorm and not dbid and self.rxcui2drugid.get(rxnorm, ''):
            dbid = {self.rxcui2drugid[rxnorm]}
        if rxnorm and not dbid and self.rxnorm2drugbank[rxnorm]:
            dbid = self.rxnorm2drugbank[rxnorm]
        if rxnorm and not dbid and self.rxcui2pref.get(rxnorm, ""):
            rxnorm_name = self.rxcui2pref[rxnorm].lower()
            rdids = set()
            for name_, _ in get_cleaned_names(rxnorm_name, key=None):
                r, d = self.get_rxnorm([], name_)
                rdids.update(d)
            dbid = rdids
        if rxnorm and not dbid and rxnorm in self.rxnorm2drugbank_umls:
            dbid = self.rxnorm2drugbank_umls[rxnorm]
        if not dbid:
            dbid = dbids
        if not dbid and re.match(r'^[a-z]{2,3}-[0-9]+$', name):
            # print(name)
            pbchem_ids, dbid = self._get_DBId_from_pubchem(name)
            rxnorm = (rxnorm, pbchem_ids)
        if not dbid and "-" in name:
            rxnorm, dbid = self.get_rxnorm(drug_infos, name.replace("-", " "))
            if not dbid and len(name.split("-")) == 2:
                a, b = name.split("-")
                if len(a) > 3 and len(b) > 3:
                    ra, dbida = self.get_rxnorm([], a)
                    rb, dbidb = self.get_rxnorm([], b)
                    if dbida and dbidb:
                        dbid = set()
                        dbid.update(dbida)
                        dbid.update(dbidb)
        return rxnorm, dbid


def get_intervention_drug_ids(matcher: DrugMatcher, intervention, trial):
    if intervention['intervention_type'] != 'Drug':
        # Some trials have placebo arms labelled as Other
        name = intervention['intervention_name']
        if (intervention['intervention_type'] == 'Other'
                and is_placebo(intervention['intervention_name'].lower())
                and "+" not in name and "&" not in name and " and " not in name and ' plus ' not in name):
            # if("+" in intervention['intervention_name']):
            # print("Other: ", intervention['intervention_name'])
            intervention['intervention_type'] = 'Placebo'
            intervention['is_incomplete'] = False
            intervention['drug_ids'] = {'D#######'}
        return intervention

    intervention['is_incomplete'] = False
    other_names = intervention.get('other_name', [])
    name_other_name = intervention['intervention_name'].lower() + " " + " ".join(other_names).lower()

    intername2id = {}

    intervention_name = intervention['intervention_name'].lower()

    # Placebo
    # TODO: check if + is not there in this because it can be drug + some placebo
    has_placebo = False
    if is_placebo(name_other_name):
        if ("+" in name_other_name or '&' in name_other_name
                or ' plus ' in name_other_name or ' and ' in name_other_name):
            parts = re.split(r"&|\+| plus | and ", name_other_name)
            non_placebo_parts = []
            for part in parts:
                if not is_placebo(part):
                    non_placebo_parts.append(part)
            if len(non_placebo_parts) == 0:
                intervention['drug_ids'] = {'D#######'}
                return intervention
            else:
                # print(name_other_name)
                name_other_name = "+".join(non_placebo_parts).strip()
                # print(name_other_name, parts)
                has_placebo = True
            # print(name_other_name)
        else:
            intervention['drug_ids'] = {'D#######'}
            return intervention

    # Drug name is directly matched

    _, drugbank_id = matcher.get_rxnorm([], intervention_name)

    # Create a map of other_names
    for other_name in other_names:
        intername2id[other_name.lower()] = matcher.get_rxnorm([], other_name.lower())[1]

    # pprint(intername2id)

    # If all other names are mapped then we are done (should be right?)
    if intername2id and all(map(lambda x: len(x) > 0, intername2id.values())):
        drug_ids = set()
        for dids in intername2id.values():
            drug_ids.update(dids)
        if drugbank_id and len(drugbank_id) < len(drug_ids):
            intervention['drug_ids'] = drugbank_id
        else:
            intervention['drug_ids'] = drug_ids
        # print(intervention_name, intervention['drug_ids'])
        return intervention
    if drugbank_id:
        intervention['drug_ids'] = drugbank_id
        # print(intervention_name, drugbank_id)
        return intervention

    drug_ids = set()
    medex_names = set()

    for key in intervention['medex_out'].keys():
        clean_key = complete_code_incompletes(key, [intervention_name])
        medex_names.update(get_cleaned_names(clean_key, key))

    for key in list(intervention.get('medex_othernames', {}).keys()):
        if not key:
            continue
        clean_key = complete_code_incompletes(key, other_names)
        add = True

        # Do not add key if a larger name is already matched
        for oname in intername2id:
            if key in oname and intername2id[oname]:
                drug_ids.update(intername2id[oname])
                add = False
        if add:
            medex_names.update(get_cleaned_names(clean_key, key))

    if not medex_names:
        for other_name in other_names:
            medex_names.update(get_cleaned_names(other_name, key=None))

    names = set()
    for name, key in medex_names:
        # Only consider drug names in either intervention name or other names
        if name not in name_other_name:
            continue

        name = handle_interferon(name, name_other_name)

        # Do not match names less than len 3 bcoz they are shorthands which are misleading in general
        if (len(name) <= 3 or name in ['co', 'at 10', 'at 20', 'cr', 'placebo', 'hypertonic saline', 'saline',
                                       'normal saline', 'eye drops', 'eye drop', 'nasal spray', 'corticosteroid',
                                       'corticosteroids', 'adjuvant', 'sodium', 'steroid', 'steroids', 'agonist',
                                       'vaccine',
                                       'pharmaceuticals', 'local anesthetic', 'local anesthesia', 'mesylate',
                                       'long acting', 'prophylactic']):
            continue
        names.add((name, key))

    # Filter out complete substrings from medex_names
    # print(medex_names)
    names = filter_complete_substrings(names)

    # If no medex names found, add to the list of no names
    # We can now use drugs from the arms to fill this set
    # Check if the logic is correct for when is_placebo = True
    # if (not names):
    #     trials.add(trial['nct_id'])

    for drug_name, key in names:
        drug_infos = trial['medex_processed'].get(key, [])
        rxnorm_cui, drugbank_id = matcher.get_rxnorm(drug_infos, drug_name)
        if not drugbank_id:
            # trials.add(trial['nct_id'])
            intervention['is_incomplete'] = True
            # unmatched_drug_names.add((drug_name, rxnorm_cui, trial['nct_id']))
        else:
            drug_ids.update(drugbank_id)

        # print(drug_name, drugbank_id)

    # If we have matched the intervention name now in full, just return that as the drug for the trial
    for drug_id in drug_ids:
        if drug_id in matcher.drugid2name \
                and matcher.drugid2name[drug_id].lower() == intervention['intervention_name'].lower():
            drug_ids = {drug_id}
            intervention['is_incomplete'] = False
            break
    if has_placebo:
        drug_ids.add('D#######')

    intervention['drug_ids'] = drug_ids
    return intervention


if __name__ == '__main__':
    DATA_DIR = 'data'
    drug_matcher = DrugMatcher(data_paths={
        'drug_data': f'{DATA_DIR}/drug_data/drugs_all_03_04_21.pkl',
        'pubchem_synonyms': f'{DATA_DIR}/drug_data/pubchem-drugbankid-synonyms.json',
        'rxnorm2drugbank-umls': f'{DATA_DIR}/drug_data/rxnorm2drugbank-umls.pkl',
        'RXNCONSO': f'{DATA_DIR}/drug_data/RXNCONSO.RRF'
    })
    print(drug_matcher.get_rxnorm([], 'aspirin'))
