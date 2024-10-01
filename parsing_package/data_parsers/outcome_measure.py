import re
from collections import defaultdict
from typing import List

import gensim
from nltk.corpus import stopwords
from stemming.porter2 import stem
import nltk

nltk.download('stopwords')


class OutcomeMeasureExtract:
    def __init__(self, cluster_file_path: str) -> None:
        self.stopwords = self._get_stopwords()
        self.bigram_mod = None
        self.trigram_mod = None
        self._load_cluster_info(cluster_file_path)

    def _load_cluster_info(self, cluster_file_path: str) -> None:
        # Add primary/secondary outcome measures
        token2clusterid = {}
        cid2cluster = {}
        with open(cluster_file_path, 'r') as f:
            for idx, line in enumerate(f.readlines()):
                cluster, count = line.strip().split("\t")
                cid2cluster[idx] = cluster
                for token in cluster.split(", "):
                    token = token.strip()
                    token2clusterid[token] = idx
        self.token2clusterid = token2clusterid
        self.cid2cluster = cid2cluster
        
    def save_phrase_models(self, base_dir: str) -> None:
        self.bigram_mod.save(f'{base_dir}/outcome_measures_phrase_bigram_model.pkl')
        self.trigram_mod.save(f'{base_dir}/outcome_measures_phrase_trigram_model.pkl')
        
    def load_phrase_models(self, base_dir: str) -> None:
        self.bigram_mod = gensim.models.phrases.Phraser.load(f'{base_dir}/outcome_measures_phrase_bigram_model.pkl')
        self.trigram_mod = gensim.models.phrases.Phraser.load(f'{base_dir}/outcome_measures_phrase_trigram_model.pkl')

    def build_phrase_models(self, data_df):
        def _get_measures():
            primary_outcome_measures = []
            secondary_outcome_measures = []
            for idx, trial in data_df.iterrows():
                poms = trial.get('primary_outcome', []) or []
                soms = trial.get('secondary_outcome', [])
                if type(poms) != list:
                    poms = []
                if type(soms) != list:
                    soms = []
                for pom in poms:
                    measure = pom.get("measure", "")
                    if len(measure) > 0:
                        primary_outcome_measures.append(measure)
                for pom in soms:
                    measure = pom.get("measure", "")
                    if len(measure) > 0:
                        secondary_outcome_measures.append(measure)
            return primary_outcome_measures + secondary_outcome_measures

        all_data = _get_measures()
        all_data = self._preprocess(all_data)
        all_data_words = list(self._sent_to_words(all_data))
        bigram = gensim.models.Phrases(all_data_words, min_count=5, threshold=5)  # higher threshold fewer phrases.
        trigram = gensim.models.Phrases(bigram[all_data_words], threshold=5)

        # Faster way to get a sentence clubbed as a trigram/bigram
        self.bigram_mod = gensim.models.phrases.Phraser(bigram)
        self.trigram_mod = gensim.models.phrases.Phraser(trigram)

    @staticmethod
    def _preprocess(data):
        data = [re.sub(r'\s+', ' ', sent) for sent in data]

        # Remove distracting single quotes
        data = [re.sub(r"\'", "", sent) for sent in data]

        return data

    @staticmethod
    def _get_stopwords():
        stop_words = stopwords.words('english')
        stop_words.remove('off')
        stop_words.remove('who')
        stop_words.remove('t')
        return stop_words

    @staticmethod
    def _sent_to_words(sentences):
        for sentence in sentences:
            yield (gensim.utils.simple_preprocess(str(sentence), deacc=True, min_len=1,
                                                  max_len=25))  # deacc=True removes punctuations

    def _remove_stopwords(self, texts):
        return [[word for word in doc if word not in self.stopwords] for doc in texts]

    def _make_bigrams(self, texts):
        return [self.bigram_mod[doc] for doc in texts]

    def _make_trigrams(self, texts):
        return [self.trigram_mod[self.bigram_mod[doc]] for doc in texts]

    @staticmethod
    def _stem_token(token):
        if "_" in token:
            return "_".join([stem(w) for w in token.split('_')])
        else:
            return stem(token)

    def _stemmer(self, texts):
        texts_out = []
        for sent in texts:
            texts_out.append([self._stem_token(token) for token in sent])
        return texts_out

    def _tokenize(self, text):
        words = list(self._sent_to_words(text))
        words_nostops = self._remove_stopwords(words)
        words_trigrams = self._make_trigrams(words_nostops)
        text_stemmed = self._stemmer(words_trigrams)
        return text_stemmed

    def _match_sentences_substring(self, text_stemmed: List):
        sent2token = defaultdict(set)
        for idx, text in enumerate(text_stemmed):
            sent = "_" + "_".join(text) + "_"
            for token, cid in self.token2clusterid.items():
                if "_" + token + "_" in sent:
                    sent2token[idx].add(cid)
        return sent2token

    def get_cluster_ids(self, text: str):
        try:
            cids = self._match_sentences_substring(self._tokenize(self._preprocess([text])))[0]
            return cids, {self.cid2cluster[cid] for cid in cids}
        except Exception as e:
            print(e, text)
            return set(), set()

    def populate_cids(self, trial):
        for outcome_key in ['primary_outcome', 'secondary_outcome']:
            outcomes = trial.get(outcome_key, []) or []
            if type(outcomes) != list:
                trial[outcome_key] = []
                outcomes = []
            for pom in outcomes:
                measure = pom.get("measure", "")
                if len(measure) > 0:
                    cids, clusters = self.get_cluster_ids(measure)
                    pom['cids'] = cids
                    pom['clusters'] = clusters
