from imblearn.over_sampling import SMOTENC
import pandas as pd
import numpy as np


class SMOTE_nc:
    def __init__(self, X_train, Y_train, category_columns=None, sampling_rate='balance'):
        if sampling_rate == 'balance':
            n_class = len(np.unique(Y_train))
            n_samples = [Y_train[Y_train == i].shape[0] for i in range(n_class)]
            min_samples = min(n_samples)
            model = SMOTENC(categorical_features=category_columns, sampling_strategy='not majority',
                            random_state=0, k_neighbors=min_samples-1)
            self.X_os, self.Y_os = model.fit_sample(X_train, Y_train)
        else:
            model = SMOTENC(categorical_features=category_columns, sampling_strategy=sampling_rate,
                            random_state=0, k_neighbors=n_neighbors)
            self.X_os, self.Y_os = model.fit_sample(X_train, Y_train)

        self.count_bf = {'Class 0': str(Y_train).count('0'), 'Class 1': str(Y_train).count('1'),
                         'Class 2': str(Y_train).count('2'), 'Class 3': str(Y_train).count('3')}
        self.count_af = {'Class 0': str(self.Y_os).count('0'), 'Class 1': str(self.Y_os).count('1'),
                         'Class 2': str(self.Y_os).count('2'), 'Class 3': str(self.Y_os).count('3')}

    def output(self):
        return self.X_os, self.Y_os

    def count(self):
        count_bf = pd.DataFrame.from_dict(self.count_bf, orient='index', columns=['Before'])
        count_af = pd.DataFrame.from_dict(self.count_af, orient='index', columns=['After'])
        output_count = pd.concat((count_bf, count_af), axis=1)
        return output_count

