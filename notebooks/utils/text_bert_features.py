import re

import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel


class TextBertFeatures:
    def __init__(self, bert_model='microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext', device='cpu'):
        self.bert_model = bert_model
        self.device = device
        self._load_bert_model()

    def _load_bert_model(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.bert_model)

        self.model = AutoModel.from_pretrained(self.bert_model).to(self.device).eval()

    def _embed(self, text):
        device = self.device
        model = self.model
        tokens = self.tokenizer(text, max_length=512, return_tensors="pt")
        output = model(input_ids=tokens['input_ids'].to(device), token_type_ids=tokens['token_type_ids'].to(device),
                       attention_mask=tokens['attention_mask'].to(device)).pooler_output
        return output.cpu().detach().numpy()[0]

    def disease_features(self, mesh_data):
        embs = {}
        for key, term in tqdm(mesh_data.items()):
            embs[key] = self._embed(term.description)
        embs['HEALTHY'] = self._embed('healthy')
        return pd.DataFrame(list(embs.items()), columns=['mesh_id', 'embs'])

    def drug_features(self, drug_data):
        def prop(x, name="SMILES"):
            if type(x) == float:
                return ""
            for e in x['property']:
                if e['kind'] == 'SMILES':
                    return e['value']
            return ""

        drug_data['smiles'] = drug_data['calculated-properties'].map(prop)
        df_drug = drug_data[['primary_id', 'description', 'smiles']]
        df_drug = df_drug.fillna("")
        df_drug['embs'] = df_drug['description'].map(lambda x: self._embed(x))
        return df_drug

    def trial_features(self, trials_data):
        df = trials_data[['nct_id', 'brief_summary', 'detailed_description']]
        df['brief_summary'] = df['brief_summary'].map(lambda x: re.sub(r"\n\s*", " ", x))
        df['embs'] = df['brief_summary'].map(lambda x: self._embed(x))
        return df
