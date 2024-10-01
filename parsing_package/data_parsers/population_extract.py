import json
import os
import pickle
import re
from collections import defaultdict
from pprint import pprint

import nltk
import numpy as np
import scipy
from nltk.tokenize.treebank import TreebankWordTokenizer, TreebankWordDetokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm

from .external_tools import umls_search

nltk.download('averaged_perceptron_tagger_eng')
nltk.download('averaged_perceptron_tagger')

VOCABS = ['NCI', 'MSH', 'SNOMEDCT_US', 'DRUGBANK', 'MEDCIN', 'ATC', 'MDR', 'ICD10', 'RXNORM',
          'ICD10PCS', 'ICD10CM', 'NCI_NCI-GLOSS', 'NCI_UCUM', 'MMSL', 'MEDLINEPLUS', 'MTH']

tokenizer = TreebankWordTokenizer()
detokenizer = TreebankWordDetokenizer()
stemmer = nltk.PorterStemmer()


class UMLSConceptSearcher:
    def __init__(self, api_key: str, version: str, cache_dir: str, cache_size: int = 10000):
        self.version = version
        self.cache_dir = os.path.join(cache_dir, version)
        self.cache_ckpt_size = cache_size
        self.umls_client = umls_search.UMLSClient(apikey=api_key, version=self.version)
        self._load_from_cache()
        self.search_umls = True

    def set_umls_search(self, search_umls):
        self.search_umls = search_umls

    def search_term(self, search_term):
        if search_term in self.cache:
            return self.cache[search_term][0]
        if not self.search_umls:
            return None
        terms = self.umls_client.search(search_term, repeat_count=0, searchType='words',
                                        includeSuppressible=False)

        # pprint(terms)
        # concept = traverseRestResource(term['uri'])
        # concept['defaultPreferredAtom'] = traverseRestResource(concept['defaultPreferredAtom'])
        # term['concept'] = concept
        return self._update_cache(search_term, terms)

    def _update_cache(self, search_term, terms):
        if len(terms) == 0:
            term = None
        else:
            term = terms[0]
            # pprint(terms)
            for p_term in terms[:10]:
                if p_term['rootSource'] in VOCABS:
                    term = p_term
                    break
        assert search_term not in self.cache
        self.cache[search_term] = term, terms
        self.new_cache[search_term] = term, terms
        self._save_cache()
        return term

    def _save_cache(self, force=False):
        if len(self.new_cache) < self.cache_ckpt_size and not force:
            return
        last_ckpt = self.cache_state["last_ckpt"]
        ckpt_file_path = os.path.join(self.cache_dir, f'ckpt_{last_ckpt}_{last_ckpt + self.cache_ckpt_size}.pkl')
        with open(ckpt_file_path, 'wb') as f:
            pickle.dump(self.new_cache, f)
        self.cache_state['num_files'] = self.cache_state['num_files'] + 1
        self.cache_state['last_ckpt'] = last_ckpt + self.cache_ckpt_size
        state_file_path = os.path.join(self.cache_dir, 'state.json')
        with open(state_file_path, "w") as f:
            json.dump(self.cache_state, f)
        self.new_cache = {}

    def _load_cache_state(self):
        state_file_path = os.path.join(self.cache_dir, 'state.json')
        if not os.path.exists(state_file_path):
            state = {
                'num_files': 0,
                'last_ckpt': 0
            }
            with open(state_file_path, "w") as f:
                json.dump(state, f)
        with open(state_file_path) as f:
            self.cache_state = json.load(f)

    def _load_from_cache(self):
        self._load_cache_state()
        cache = {}
        for cfile in os.listdir(self.cache_dir):
            if cfile == 'state.json':
                continue
            with open(os.path.join(self.cache_dir, cfile), 'rb') as f:
                cache_block = pickle.load(f)
            cache.update(cache_block)
        self.cache = cache
        print("UMLS Concept Cache loaded, found {} entries".format(len(self.cache)))
        self.new_cache = {}


class Criteria(object):
    def __init__(self, category, term_text):
        self.category = category
        self.orig_term = term_text
        self.term = re.sub(r'[^\x00-\x7F]', ' ', term_text).strip()
        self._normalize_term()
        self.temporal = set()
        self.value = set()
        self.negated = False
        self.matched_term = None
        self.concept = None

    def _normalize_term(self):
        self.term = re.sub(r" 's", "'s", self.term)  # remove dd 's
        self.term = re.sub(r"\([^)]*$", "", self.term)  # remove wdew ( ff
        self.term = re.sub(r"^[^(]*\)", "", self.term)  # remove swd ) jdf
        adjectives = ['major', 'active', 'primary', 'other', 'desire', 'probable',
                      'additional', 'severe', 'significant', 'poor', 'poorer', 'suspected', 'prescribed',
                      'normal', 'chronic', 'serious', 'adequate', 'another', 'or', 'mild', 'mildly',
                      'uncontrolled', 'treated', 'untreated', 'controlled']
        adjectives.extend([x.capitalize() for x in adjectives])
        while True:
            temp = re.sub("( |^|$)(" + '|'.join(adjectives) + ")( |^|$)", '', self.term)
            if self.term == temp:
                break
            self.term = temp

    def map_concept(self, umls_concept_matcher: UMLSConceptSearcher):
        self.concept = umls_concept_matcher.search_term(self.term)
        self.matched_term = self.term
        if self.concept is None:
            new_term = re.sub(r'(grade)?\s?\d+', '', self.term)
            if new_term != self.term:
                self.matched_term = new_term
                self.concept = umls_concept_matcher.search_term(new_term)
        if self.concept is None:
            new_term = re.sub(r"non-?", '', new_term)
            if new_term != self.term:
                self.concept = umls_concept_matcher.search_term(new_term)
                self.negated = True
                self.matched_term = new_term
        if self.concept is None:
            words = tokenizer.tokenize(self.term)
            tags = nltk.pos_tag(words)
            if len(words) > 1 and tags[0][1][:2] in ['JJ', 'DT', 'RB']:
                new_term = detokenizer.detokenize(words[1:])
                self.concept = umls_concept_matcher.search_term(new_term)
                self.matched_term = new_term
        if self.concept is None:
            self.matched_term = None

    def add_relation(self, relation, value):
        if relation == 'has_temporal':
            self.temporal.add(value)
        elif relation == 'has_value':
            self.value.add(value)
        else:
            raise ValueError("relation must be temporal or value")

    def __str__(self):
        s = self.term + " ({}) | ".format(self.orig_term)
        if len(self.temporal):
            s += ", ".join(self.temporal)
            s += " "
        if len(self.value):
            s += ", ".join(self.value)
            s += " "
        if self.concept:
            s += " (Concept: {})".format(self.concept['name'])
        return s.strip()

    def __repr__(self):
        return str(self)


class CriteriaOutputParser:
    def __init__(self, basedir, pb=True):
        self.basedir = basedir
        self.pb = tqdm(total=None)

    @staticmethod
    def _parse_criteria(criteria, crit_dict, exclusion=False):
        for criterion in criteria:
            for sent in criterion['sents']:
                # pprint(sent)
                terms = {term['termId']: term for term in sent['terms']}
                # print(terms)
                relations = defaultdict(list)
                for relation in sent['relations']:
                    relations[relation['first']].append(relation)
                for termId, term in terms.items():
                    # print(term)
                    category = term['categorey']
                    # use relations to map temporal and value attributes
                    if category in ['Temporal', 'Value', 'Negation_cue']:
                        continue
                    neg = term['neg']
                    text = term['text']

                    for text in re.split(" or |/| and |,", text):
                        crit = Criteria(category, text)
                        # pprint(relations)
                        if termId in relations:
                            for relation in relations[termId]:
                                if relation['second'] in terms:
                                    crit.add_relation(relation['third'], terms[relation['second']]['text'])
                        inclusion = not exclusion
                        if neg:
                            inclusion = not inclusion
                        if inclusion:
                            crit_dict[category]['inclusion'].add(crit)
                        else:
                            crit_dict[category]['exclusion'].add(crit)

    def parse_trial(self, nctid):
        folder = nctid[:-4] + "x" * 4
        filename = os.path.join(self.basedir, folder, nctid + '.json')
        output = self.parse_crit_output_from_file(filename)
        self.pb.update()

    @staticmethod
    def parse_crit_output_from_file(filename: str):
        with open(filename, 'r') as f:
            criteria = json.load(f)['eligibility criteria']
        exclusion_criteria = criteria['exclusion_criteria']
        inclusion_criteria = criteria['inclusion_criteria']
        crit_dict = defaultdict(lambda: {'inclusion': set(), 'exclusion': set()})
        CriteriaOutputParser._parse_criteria(inclusion_criteria, crit_dict)
        CriteriaOutputParser._parse_criteria(exclusion_criteria, crit_dict, exclusion=True)

        return {k: v for k, v in crit_dict.items()}


def clean_nonascii(term):
    return re.sub(r'[^\x00-\x7F]', ' ', term).strip()


class UMLSTFIDFMatcher:
    def __init__(self, cuid2concept, savedir: str, df=None, tfidf_threshold=0.8):
        self.cuid2concept = cuid2concept
        self.tfidf_threshold = tfidf_threshold
        self.stopwords = list('!"#$%&\'*+,-./:;<=>?@\\^_`{|}~')

        if df is not None:
            self._build_term_map(df)
            self._fit_tfidf_model()
            self.build_cid2concept(df)
            self.save_state(savedir)
        else:
            self.load_state(savedir)

    def save_state(self, basedir: str) -> None:
        filepath = os.path.join(basedir, "tfidf_matcher_state.pkl")
        with open(filepath, "wb") as f:
            pickle.dump({
                'term2cid': self.term2cid,
                'cid2concept': self.cid2concept
            }, f)

    def load_state(self, basedir: str) -> None:
        filepath = os.path.join(basedir, "tfidf_matcher_state.pkl")
        with open(filepath, "rb") as f:
            state = pickle.load(f)
        self.term2cid = state['term2cid']
        self.cid2concept = state['cid2concept']

    def build_cid2concept(self, df):
        cid2concept = {}
        for criteria_all in tqdm(df.ec_umls):
            for category in criteria_all:
                for inclusion in criteria_all[category]:
                    for criterion in criteria_all[category][inclusion]:
                        if criterion.concept:
                            cid = criterion.concept['ui']
                            cid2concept[cid] = criterion.concept
        self.cid2concept = cid2concept

    def _build_term_map(self, df):
        term2cid = {}
        cid_not_found = set()
        for criteria_all in tqdm(df['ec_umls']):
            for category in criteria_all:
                for inclusion in criteria_all[category]:
                    for criterion in criteria_all[category][inclusion]:
                        if criterion.concept is not None:
                            cid = criterion.concept['ui']
                            if cid not in self.cuid2concept:
                                cid_not_found.add(cid)
                                # print(cid, criterion.term)
                            else:
                                term2cid[self.cuid2concept[cid]] = cid
                            term2cid[criterion.term] = cid
                            term2cid[clean_nonascii(criterion.orig_term)] = cid
                            for parent in criterion.parents:
                                term2cid[self.cuid2concept[parent]] = parent
                        else:
                            term2cid[criterion.term] = None
                            term2cid[clean_nonascii(criterion.orig_term)] = None
        print("Number of CIDs not found: ", len(cid_not_found))
        self.term2cid = term2cid

    def _fit_tfidf_model(self):
        stopwords = self.stopwords

        def preprocess(term):
            return [stemmer.stem(word) for word in tokenizer.tokenize(term.lower())
                    if word not in stopwords]

        term2cid = self.term2cid
        all_terms = list(term2cid.keys())
        print("Number of terms: ", len(all_terms))
        # settings that you use for count vectorizer will go here
        tfidf_vectorizer = TfidfVectorizer(use_idf=True, smooth_idf=True, max_df=0.95, analyzer=lambda x: x)
        # just send in all your docs here
        concept_terms = [term for term in all_terms if term2cid[term] is not None]
        concept_docs = [preprocess(term) for term in concept_terms]

        no_concept_terms = [term for term in all_terms if term2cid[term] is None]
        no_concept_docs = [preprocess(term) for term in no_concept_terms]

        print("Concept terms: ", len(concept_terms), "No concept terms", len(no_concept_terms))
        tfidf_vectorizer.fit(concept_docs + no_concept_docs)
        print("Model fit")

        concept_vectors = tfidf_vectorizer.transform(concept_docs)
        print(concept_vectors.shape)
        print(concept_vectors[:1], concept_docs[:1], concept_terms[:1])

        no_concept_vectors = tfidf_vectorizer.transform(no_concept_docs)
        print(no_concept_vectors.shape, concept_vectors.shape)

        sim = no_concept_vectors * concept_vectors.T
        print(sim.shape)

        sim_max, sim_argmax = scipy.sparse.csr_matrix.max(sim, axis=1).todense(), scipy.sparse.csr_matrix.argmax(sim,
                                                                                                                 axis=1)
        most_sim_term = np.array(concept_terms)[sim_argmax]

        import random
        for i in random.sample(range(len(no_concept_terms)), 100):
            if sim_max[i] >= self.tfidf_threshold:
                print(no_concept_terms[i], "|", most_sim_term[i][0])

        cnt = 0
        for idx, term in tqdm(enumerate(no_concept_terms)):
            if len(term) > 2 and sim_max[idx] >= self.tfidf_threshold:
                cnt += 1
                term2cid[term] = term2cid[most_sim_term[idx][0]]
        print(len(no_concept_terms), cnt)

    def populate_result_single(self, criteria_all):
        term2cid = self.term2cid
        cid2concept = self.cid2concept
        for category in criteria_all:
            for inclusion in criteria_all[category]:
                for criterion in criteria_all[category][inclusion]:
                    if not criterion.concept:
                        orig_term = clean_nonascii(criterion.orig_term)
                        term = criterion.term
                        if term2cid[orig_term] is not None:
                            cid = term2cid[orig_term]
                        else:
                            cid = term2cid[term]
                        if cid is not None:
                            if cid not in cid2concept:
                                concept = {
                                    'ui': cid,
                                    'rootSource': '',
                                    'name': self.cuid2concept[cid]
                                }
                            else:
                                concept = cid2concept[cid]
                            criterion.concept = concept

    def populate_results(self, df):
        for criteria_all in tqdm(df.ec_umls):
            self.populate_result_single(criteria_all)


class UMLSGraphClipper:
    def __init__(self, evaluate_parents, max_level=2):
        self.evaluate_parents = evaluate_parents
        self.max_level = max_level

    @staticmethod
    def _make_nct_count(df):
        cuid2nctids = defaultdict(set)
        for idx, trial in tqdm(df.iterrows()):
            nctid = trial['nct_id']
            criteria = trial['ec_umls']
            for category in criteria:
                for inclusion in criteria[category]:
                    for criterion in criteria[category][inclusion]:
                        if criterion.concept is not None:
                            cuid2nctids[criterion.concept['ui']].add(nctid)
        return cuid2nctids

    @staticmethod
    def _count_threshold(counter, thresholds):
        counts = [0] * len(thresholds)
        for cuid, nctids in counter.items():
            count = len(nctids)
            for idx, threshold in enumerate(thresholds):
                if count >= threshold:
                    counts[idx] += 1
        return counts

    def _parents_above_threshold(self, cuid, threshold, c2nct):
        invalid = {cuid}
        valid = set()
        for level in range(self.max_level + 1):
            new_invalid = set()
            for cui in invalid:
                if len(c2nct[cui]) >= threshold:
                    valid.add(cui)
                else:
                    new_invalid.add(cui)
            if level == self.max_level:
                if len(valid) > 0:
                    return valid
                else:
                    return new_invalid
            invalid = set()
            for cui in new_invalid:
                invalid.update(self.evaluate_parents(cui))

    def clip_graph(self, df, count_full_children=False,
                   thresholds=(5, 10, 20, 50, 100), verbose=False):
        cuid2nctids = self._make_nct_count(df)
        newc2parent = dict()
        c2countnew = dict()
        level2counts = dict()
        for threshold in thresholds:
            f_map = defaultdict(set)
            newc2parent[threshold] = dict()

            cuid2nctids_t = defaultdict(set)
            for k, v in cuid2nctids.items():
                cuid2nctids_t[k] = v

            for level in range(self.max_level):
                cuid2nctids_next = defaultdict(set)
                for cuid, nctids in cuid2nctids_t.items():
                    if count_full_children or len(nctids) < threshold:
                        for parent in self.evaluate_parents(cuid):
                            cuid2nctids_next[parent].update(nctids)
                    cuid2nctids_next[cuid].update(nctids)
                cuid2nctids_t = cuid2nctids_next
            if verbose:
                pprint(cuid2nctids_t)
            level2counts[threshold] = cuid2nctids_t
            for cuid, nctids in cuid2nctids.items():
                cuids = self._parents_above_threshold(cuid, threshold, cuid2nctids_t)
                newc2parent[threshold][cuid] = cuids
                for c in cuids:
                    f_map[c].update(nctids)
            if verbose:
                pprint(f_map)
            print(threshold, self._count_threshold(f_map, [1, 5, 10, 20, 50, 100, 500, 1000, 10000]))
            c2countnew[threshold] = f_map
        return newc2parent, c2countnew, level2counts
