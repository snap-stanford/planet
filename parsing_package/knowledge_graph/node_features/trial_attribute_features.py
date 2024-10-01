import math

import numpy as np


class TrialAttributeFeatures:
    def __init__(self, attributes=('age', 'gender', 'enrollment', 'phase')):
        self.attributes = attributes

    @staticmethod
    def _phase_feature_vec(phase):
        v = [0] * 5
        if phase in ['Early Phase 1', 'Phase 1']:
            v[1] = 1
        elif phase == 'N/A':
            v[0] = 1
        elif phase == 'Phase 2':
            v[2] = 1
        elif phase == 'Phase 3':
            v[3] = 1
        elif phase == 'Phase 4':
            v[4] = 1
        elif phase == 'Phase 1/Phase 2':
            v[1] = 1
            v[2] = 1
        elif phase == 'Phase 2/Phase 3':
            v[2] = 1
            v[3] = 1
        else:
            raise RuntimeError(f"Unknown phase: {phase}")
        return v

    @staticmethod
    def _enrollment_feat(enrollment):
        is_anticipated = False
        if type(enrollment) == dict:
            if enrollment['@type'] == 'Anticipated':
                is_anticipated = True
            return [math.log(1 + enrollment['$']), int(is_anticipated)]
        if np.isnan(enrollment):
            return [0, 0]
        return [math.log(1 + enrollment), 0]

    @staticmethod
    def _sex_vec(sex):
        if type(sex) == float:
            return [0, 0, 0]
        sex_to_feats = {
            'All': [1, 0, 0],
            "Male": [0, 1, 0],
            "Female": [0, 0, 1]
        }
        return sex_to_feats[sex]

    @staticmethod
    def _age_vec(age):
        if type(age) == float or age == 'N/A':
            return [1, 0, 0, 0]
        val, unit = age.split(" ")
        if unit == 'Years':
            return [0, int(val), 0, 0]
        if unit in ['Months', 'Year', 'Weeks']:
            if unit == "Year":
                val = int(val) * 12
            elif unit == 'Weeks':
                val = int(val) / 4
            return [0, 0, int(val), 0]
        if unit == 'Month':
            assert int(val) == 1
            return [0, 0, 0, 30]
        if unit == 'Week':
            assert int(val) == 1
            return [0, 0, 0, 7]
        if unit in ['Days', 'Day']:
            return [0, 0, 0, int(val)]
        if unit in ['Hours', 'Hour']:
            return [0, 0, 0, int(val) / 24]
        if unit in ['Minutes']:
            return [0, 0, 0, int(val) / (24 * 60)]
        else:
            raise RuntimeError(age)

    @staticmethod
    def _age_vec_2(row):
        minimum_age = row['minimum_age']
        maximum_age = row['maximum_age']
        if type(minimum_age) == float or minimum_age == 'N/A':
            minimum_age = '1 Day'
        if type(maximum_age) == float or maximum_age == 'N/A':
            maximum_age = '130 Years'
        cats = ['Child', 'Adult', 'Older Adult']
        min_val, unit = minimum_age.split(" ")
        min_val = int(min_val)
        if unit != 'Years':
            min_val = 1
        max_val, unit = maximum_age.split(" ")
        max_val = int(max_val)
        if unit != 'Years':
            max_val = 1

        if max_val <= 17:
            return [1, 0, 0]
        elif max_val <= 64:
            if min_val <= 17:
                return [1, 1, 0]
            else:
                return [0, 1, 0]
        else:
            if min_val <= 17:
                return [1, 1, 1]
            elif min_val <= 64:
                return [0, 1, 1]
            else:
                return [0, 0, 1]

    def features(self, trial_df):
        df_all = trial_df
        df_all['phase_vec'] = df_all['phase'].map(self._phase_feature_vec)
        df_all['enrollment_vec'] = df_all.enrollment.map(self._enrollment_feat)
        df_all['gender_sex_vec'] = df_all['gender_sex'].map(self._sex_vec)
        df_all['minimum_age_vec'] = df_all['minimum_age'].map(self._age_vec)
        df_all['maximum_age_vec'] = df_all['maximum_age'].map(self._age_vec)
        df_all['age_vec_2'] = df_all.apply(self._age_vec_2, axis=1)

        def merge_vecs(row):
            feats = []
            for attribute in self.attributes:
                if attribute == 'phase':
                    feats.extend(row['phase_vec'])
                elif attribute == 'enrollment':
                    feats.extend(row['enrollment_vec'])
                elif attribute == 'gender':
                    feats.extend(row['gender_sex_vec'])
                elif attribute == 'age':
                    feats.extend(row['minimum_age_vec'])
                    feats.extend(row['maximum_age_vec'])
                elif attribute == 'age_class':
                    feats.extend(row['age_vec_2'])
                else:
                    raise RuntimeError(f"Unknown attributes ({attribute}) for features")
            return np.array(feats)

        df_all['attribute_feats'] = df_all.apply(merge_vecs, axis=1)
        return df_all[['nct_id', 'attribute_feats']]
