from __future__ import division, print_function
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold, ShuffleSplit
from mastml.legos.data_splitters import Bootstrap
from sklearn import preprocessing
import hdbscan
from sklearn.manifold import TSNE
import seaborn as sns
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split
from scipy.spatial.distance import cdist


class Data_transformer:
    def __init__(self, load_data, unused_columns=None, discrete_columns=None,
                 target_column=None):
        if isinstance(load_data, pd.DataFrame):
            self.discrete_columns = discrete_columns
            self.clean_data = load_data
            # Data preparing (cleaning)
            if unused_columns is not None:
                self.clean_data = self.clean_data.drop(columns=unused_columns)
            if target_column is not None:
                self.target = self.clean_data[target_column]
            else:
                raise ValueError('Target column must be specified!')
            if discrete_columns is not None:
                self.dis_data = self.clean_data[discrete_columns]
                self.num_data = self.clean_data.drop(columns=discrete_columns + [target_column])
                self.data = pd.concat((self.num_data, self.dis_data, self.target), axis=1)
            else:
                self.num_data = self.clean_data.drop(columns=[target_column])
                self.data = pd.concat((self.num_data, self.target), axis=1)
        else:
            raise ValueError('Input data must be a DataFrame!')

    def data_transform(self, onehot=True):
        # Normalize numerical features
        self.scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
        num_nomalized = self.scaler.fit_transform(self.num_data)

        # One-hot encoding discrete features
        if onehot is True:
            onehot_data = self.dis_data
            for i in self.discrete_columns:
                dum = pd.get_dummies(self.dis_data[i], prefix=i)
                onehot_data = onehot_data.drop(columns=i)
                onehot_data = pd.concat((onehot_data, dum), axis=1)
            trans_data = pd.concat((pd.DataFrame(num_nomalized, columns=self.num_data.columns), onehot_data, self.target), axis=1)
        else:
            if self.discrete_columns is not None:
                trans_data = pd.concat((pd.DataFrame(num_nomalized, columns=self.num_data.columns), self.dis_data, self.target), axis=1)
            else:
                trans_data = pd.concat((pd.DataFrame(num_nomalized, columns=self.num_data.columns), self.target), axis=1)
        return trans_data

    def data_inverse(self, data, discrete_indx=None, target_indx=None):
        if isinstance(data, pd.DataFrame):
            # Data preparing (cleaning)
            if target_indx is not None:
                target = data.iloc[:, target_indx]
            else:
                raise ValueError('Target column must be specified!')
            if discrete_indx is not None:
                dis_data = data.iloc[:, discrete_indx]
                num_data = data.drop(data.columns[discrete_indx + [target_indx]], axis=1)
                num_inverse = self.scaler.inverse_transform(num_data)
                invs_data = pd.concat((pd.DataFrame(num_inverse), dis_data, target), axis=1)
            else:
                num_data = data.drop(data.columns[target_indx], axis=1)
                num_inverse = self.scaler.inverse_transform(num_data)
                invs_data = pd.concat((pd.DataFrame(num_inverse), target), axis=1)
        else:
            raise ValueError('Input data must be a DataFrame!')
        return invs_data


class Data_splitting:
    def __init__(self, data, n_splits, test_size=0.2, type='StratifiedKFold'):
        global sss
        if isinstance(data, pd.DataFrame):
            self.X = data.iloc[:, 0:-1]
            self.Y = data.iloc[:, -1]
            n_class = len(np.unique(np.asarray(self.Y)))
            self.TRAIN_idx = []
            self.TEST_idx = []
            if type == 'StratifiedKFold':
                sss = StratifiedKFold(n_splits=n_splits, random_state=0)
                for train_index, test_index in sss.split(self.X, self.Y):
                    self.TRAIN_idx.append(train_index)
                    self.TEST_idx.append(test_index)
            elif type == 'ShuffleSplit':
                i = 0
                while True:
                    sss = ShuffleSplit(n_splits=n_splits, test_size=test_size, random_state=i)
                    done = 0
                    self.TRAIN_idx = []
                    self.TEST_idx = []
                    for train_index, test_index in sss.split(self.X, self.Y):
                        self.TRAIN_idx.append(train_index)
                        self.TEST_idx.append(test_index)
                        y_test = np.asarray(self.Y.iloc[test_index])
                        if len(np.unique(y_test)) < n_class:
                            i += 1
                            break
                        else:
                            done += 1
                    if done == n_splits:
                        break
            elif type == 'StratifiedShuffleSplit':
                sss = StratifiedShuffleSplit(n_splits=n_splits, test_size=test_size, random_state=0)
                for train_index, test_index in sss.split(self.X, self.Y):
                    self.TRAIN_idx.append(train_index)
                    self.TEST_idx.append(test_index)
            elif type == 'Bootstrap':
                i = 0
                count = {}
                for i in range(n_class):
                    count[i] = str(self.Y).count(f'{i}')
                min_class = list(count.keys())[list(count.values()).index(min(count.values()))]

                while True:
                    sss = Bootstrap(self.X.shape[0], n_bootstraps=n_splits, train_size=(1 - test_size),
                                    test_size=test_size, random_state=i)
                    done = 0
                    self.TRAIN_idx = []
                    self.TEST_idx = []
                    for train_index, test_index in sss:
                        self.TRAIN_idx.append(train_index)
                        self.TEST_idx.append(test_index)
                        y_test = np.asarray(self.Y.iloc[test_index])
                        y_train = np.asarray(self.Y.iloc[train_index])

                        min_class_num_train = str(y_train).count(f'{min_class}')
                        if (len(np.unique(y_test)) < n_class) or (len(np.unique(y_train)) < n_class) \
                                or (min_class_num_train < n_class + 3):
                            i += 1
                            break
                        else:
                            done += 1
                    if done == n_splits:
                        print('DONE', done)
                        break
        else:
            raise ValueError('Input data must be a DataFrame!')


class Cluster_based_splitting:
    def __init__(self, clustered_data, n_splits, test_size=0.2):
        if isinstance(clustered_data, pd.DataFrame):
            self.X = clustered_data.iloc[:, 0:-2]
            self.label = clustered_data.iloc[:, -2:-1]
            self.Y = clustered_data.iloc[:, -1]
            self.TRAIN_idx = []
            self.TEST_idx = []
            for i in range(n_splits):
                test_index = np.empty(0)
                train_index = np.empty(0)
                for j in range(self.Y.nunique()):
                    inner_label = self.label[(self.Y == j) & (self.label['Cluster_label'] != -1)]
                    class_idx = np.empty(0)
                    if inner_label.empty:
                        getidx = clustered_data[(self.Y == j)].index
                        frac = test_size
                        sample = pd.DataFrame(getidx).sample(frac=frac, replace=False, random_state=i)
                        class_idx = np.append(class_idx, sample.values).astype('int')
                    else:
                        for k in range(inner_label.nunique().values[0]):
                            # Get all index of the k-th inner-class cluster
                            getidx = clustered_data[(self.Y == j)
                                                    & (self.label['Cluster_label'] != -1)
                                                    & (self.label['Cluster_label'] == k)].index
                            # Randomly select x% of samples from each cluster
                            frac = test_size * len(self.label[self.Y == j]) / len(inner_label)
                            sample = pd.DataFrame(getidx).sample(frac=frac, replace=False, random_state=i)
                            class_idx = np.append(class_idx, sample.values).astype('int')

                    test_index = np.append(test_index, class_idx).astype('int')
                    train_index = np.asarray(clustered_data.drop(index=test_index).index)
                self.TEST_idx.append(test_index)
                self.TRAIN_idx.append(train_index)
        else:
            raise ValueError('Input data must be a DataFrame!')


class Data_inner_clustering:
    def __init__(self, data, checkTSNE=True, balance=False):
        if isinstance(data, pd.DataFrame):
            out_dir = os.path.join(os.getcwd(), 'Output')
            if os.path.exists(out_dir):
                output_path = out_dir
            else:
                os.mkdir('Output')
                output_path = out_dir

            self.X = data.iloc[:, 0:-1]
            self.Y = data.iloc[:, -1]
            stored_data = {}
            for i in range(self.Y.nunique()):
                x_eachclass = data[self.Y == i].iloc[:, 0:-1]
                if balance:
                    min_cluster_size = 5
                else:
                    min_cluster_size = int(np.ceil(len(x_eachclass.index)/20))
                    if min_cluster_size <= 1:
                        min_cluster_size = 2
                clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size)
                clusterer.fit(x_eachclass)
                x_core = x_eachclass[clusterer.labels_ != -1]
                labels_core = clusterer.labels_[clusterer.labels_ != -1]
                # silhouette = metrics.silhouette_score(x_core, labels_core)

                cluster_label = pd.DataFrame(clusterer.labels_, columns=['Cluster_label'], index=x_eachclass.index)
                df_cluster = pd.concat([x_eachclass, cluster_label], axis=1)
                stored_data['Class {}'.format(i)] = df_cluster

                if checkTSNE:
                    # Plot the cluster with TSNE
                    if len(x_eachclass.index) < 20:
                        perplexity = 5
                    elif 20 <= len(x_eachclass.index) < 50:
                        perplexity = 18
                    elif 50 <= len(x_eachclass.index) < 100:
                        perplexity = 30
                    else:
                        perplexity = 50
                    m = TSNE(n_components=2, learning_rate=50, random_state=0,
                             perplexity=perplexity, n_iter=5000, method='exact')
                    tsne_features = m.fit_transform(x_eachclass)
                    df_tsne = pd.DataFrame(cluster_label, columns=['Cluster_label'])
                    df_tsne['x'] = tsne_features[:, 0]
                    df_tsne['y'] = tsne_features[:, 1]
                    print('Kullback-Leibler divergence:', m.kl_divergence_)

                    '''Plot the t-SNE graph'''
                    sns.scatterplot(x='x', y='y', data=df_tsne, hue='Cluster_label', palette="deep")
                    plt.savefig(os.path.join(output_path, 'Class_{}.jpg'.format(i)), dpi=600)
                    plt.show()
                else:
                    pass
            temp_concat = pd.concat([i for i in list(stored_data.values())], axis=0)
            self.clustered_data = pd.concat([temp_concat, self.Y], axis=1)
        else:
            raise ValueError('Input data must be a DataFrame!')







