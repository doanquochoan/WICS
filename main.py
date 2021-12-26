from Data_processing import Data_transformer, Data_splitting, Data_inner_clustering, Cluster_based_splitting
from Classifiers import Randomforest, SVM, XGBoost, MLP
from Oversampling import SMOTE_nc
import pandas as pd
import numpy as np
import time
from BayesianOpt_hyperopt import OPTI
from sklearn.svm import SVC
import xgboost as xgb
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import make_classification

pd.set_option("display.max_rows", None, "display.max_columns", None)
begin_time = time.time()

# Create toy example
X, Y = make_classification(n_samples=150, n_features=10, n_informative=3, n_clusters_per_class=2, n_classes=3,
                           weights=(0.1, 0.3, 0.6), class_sep=1.3, random_state=1)


load_data = pd.concat([pd.DataFrame(X), pd.DataFrame(Y)],axis=1, ignore_index=True)
target_column = 10

# Cleaning data
trans = Data_transformer(load_data, target_column=target_column)
data = trans.data  # Get data
# Normalize data
trans_data = trans.data_transform(onehot=False)

# Controlling variables
n_splits = 10
tune = 1
scoring = 'AUC'  # Working only when tune = 1
split_type = 'StratifiedShuffleSplit'  # [ShuffleSplit, StratifiedShuffleSplit, Bootstrap, StratifiedKFold, WICS]
getwhat = 0  # 0: Raw; 1: Oversampling
os_method = 'SMOTE-NC'
clf = [SVM]  # [Randomforest, SVM, XGBoost, MLP]

if split_type == 'WICS':
    # Clustering data
    cltr = Data_inner_clustering(trans_data, checkTSNE=True)
    clustered_data = cltr.clustered_data
    split = Cluster_based_splitting(clustered_data, n_splits=n_splits, test_size=0.2)
else:
    split = Data_splitting(trans_data, n_splits=n_splits, test_size=0.2, type=split_type)

# Get X and Y
X = split.X
Y = split.Y
# Split data
# Get train and text index matrix
trainidx_matrix = split.TRAIN_idx
testidx_matrix = split.TEST_idx


full_dict = { }
full_dict_os = { }
full_dict_os_RC = { }
for k in clf:
    full_dict[k.__name__] = pd.DataFrame(columns=['0'])
    full_dict_os[k.__name__] = pd.DataFrame(columns=['0'])
    full_dict_os_RC[k.__name__] = pd.DataFrame(columns=['0'])

num = 1
params = []
score = []
for n_run in range(n_splits):
    print(f'----n_run = {n_run}------')
    # Train and test set
    x_train_df, y_train_df = X.iloc[trainidx_matrix[n_run]], Y.iloc[trainidx_matrix[n_run]]
    x_test_df, y_test_df = X.iloc[testidx_matrix[n_run]], Y.iloc[testidx_matrix[n_run]]

    x_train, y_train = np.asarray(x_train_df), np.asarray(y_train_df)
    x_test, y_test = np.asarray(x_test_df), np.asarray(y_test_df)

    if getwhat == 0:
        # Tuning hyper-parameters
        if tune != 0:
            start_time = time.time()
            opt = OPTI(x_train, y_train, classifier=clf[0], num=num, score=scoring)
            print('{}. Running time: {} seconds'.format(num, np.round(time.time() - start_time, 1)))
            params.append(np.append(opt.bestparams, opt.bestfun))

        # Model training
        outdict = { }
        for i in clf:
            model = i(x_train, y_train, x_test, y_test, iter=n_run, savecm=False, opti='Original', hyperpa=opt.bestparams)
            output = model.performance(case='Original')
            outdict[i.__name__] = output
            full_dict[i.__name__] = pd.concat([full_dict[i.__name__], output], axis=1, ignore_index=True, sort=False)

            output_recall = model.output_recall()
            full_dict_os_RC[i.__name__] = pd.concat([full_dict_os_RC[i.__name__], output_recall], axis=1,
                                                    ignore_index=True, sort=False)

    if getwhat == 1:
        global x_train_os, y_train_os
        if os_method == 'SMOTE-NC':
            # -------------SMOTE-NC-----------------------
            smotenc = SMOTE_nc(x_train, y_train, category_columns=None, sampling_rate='balance')
            x_train_os, y_train_os = smotenc.output()

        # Tuning hyper-parameters
        if tune != 0:
            start_time = time.time()
            opt = OPTI(x_train_os, y_train_os, classifier=clf[0], num=num, score=scoring)
            print('{}. Running time: {} seconds'.format(num, np.round(time.time() - start_time, 1)))
            params.append(opt.bestparams)

        # Model training
        join_outdict = { }
        outdict_os = { }
        for i in clf:
            model = i(x_train_os, y_train_os, x_test, y_test, savecm=False,
                      iter='os_{}'.format(n_run), opti=os_method, hyperpa=opt.bestparams)
            output = model.performance(case='Oversampling')
            outdict_os[i.__name__] = output
            full_dict_os[i.__name__] = pd.concat([full_dict_os[i.__name__], output], axis=1, ignore_index=True,
                                                 sort=False)

            output_recall = model.output_recall()
            full_dict_os_RC[i.__name__] = pd.concat([full_dict_os_RC[i.__name__], output_recall], axis=1,
                                                    ignore_index=True, sort=False)

    num += 1

# Display results
for j in clf:
    if getwhat == 0:
        print(j.__name__)
        origin = pd.concat([full_dict[j.__name__], full_dict[j.__name__].mean(axis=1), full_dict[j.__name__].std(axis=1)], axis=1,
                           names=['Values', 'Avr', 'Std'])
        print('Origin:\n', origin)
        print('------------------------------')
        if tune != 0:
            print('All best params:\n')
            print(pd.DataFrame(params))
            print('------------------------------')

        rc = pd.concat([full_dict_os_RC[j.__name__].mean(axis=1), full_dict_os_RC[j.__name__].std(axis=1)], axis=1)
        rc = pd.concat([rc, pd.DataFrame(rc.std(axis=0)).T], axis=0)
        print('Recall of each class:\n', rc)
        print('------------------------------')

    if getwhat == 1:
        print(j.__name__)
        ovs = pd.concat([full_dict_os[j.__name__], full_dict_os[j.__name__].mean(axis=1), full_dict_os[j.__name__].std(axis=1)], axis=1,
                       names=['Values', 'Avr', 'Std'])
        print('Oversampling: {}\n'.format(os_method), ovs)
        print('------------------------------')
        if tune != 0:
            print('All best params:\n')
            print(pd.DataFrame(params))
            print('------------------------------')

        rc = pd.concat([full_dict_os_RC[j.__name__].mean(axis=1), full_dict_os_RC[j.__name__].std(axis=1)], axis=1)
        rc = pd.concat([rc, pd.DataFrame(rc.std(axis=0)).T], axis=0)
        print('Recall each class:\n', rc)
        print('------------------------------')

print('Total running time:', time.time() - begin_time)

