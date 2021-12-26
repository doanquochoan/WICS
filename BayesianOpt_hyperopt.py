from __future__ import print_function
from Classifiers import Randomforest, SVM, XGBoost, MLP
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import xgboost as xgb
from sklearn.model_selection import cross_validate
import numpy as np
import os
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import roc_curve, auc, confusion_matrix
import pandas as pd
import traceback
from skelm import ELMClassifier
from hyperopt import fmin, tpe, hp, Trials
from wrapt_timeout_decorator import *
import sys
import threading
from sklearn.neural_network import MLPClassifier


try:
    import thread
except ImportError:
    import _thread as thread

def quit_function(fn_name):
    print('{0} took too long'.format(fn_name), file=sys.stderr)
    sys.stderr.flush()
    thread.interrupt_main() # raises KeyboardInterrupt
    raise KeyboardInterrupt

def exit_after(s):
    def outer(fn):
        def inner(*args, **kwargs):
            timer = threading.Timer(s, quit_function, args=[fn.__name__])
            timer.start()
            try:
                result = fn(*args, **kwargs)
            finally:
                timer.cancel()
            return result
        return inner
    return outer

import multiprocessing.pool
import functools

def timeout(max_timeout):
    """Timeout decorator, parameter in seconds."""
    def timeout_decorator(item):
        """Wrap the original function."""
        @functools.wraps(item)
        def func_wrapper(*args, **kwargs):
            """Closure for function."""
            pool = multiprocessing.pool.ThreadPool(processes=1)
            async_result = pool.apply_async(item, args, kwargs)
            # raises a TimeoutError if execution exceeds max_timeout
            return async_result.get(max_timeout)
        return func_wrapper
    return timeout_decorator


seed_value = 0
# -----------------Create stored folder--------------------
out_dir = os.path.join(os.getcwd(), 'Optimization')
if os.path.exists(out_dir):
    output_path = out_dir
else:
    os.mkdir('Optimization')
    output_path = out_dir
# -----------------Create stored folder [End]--------------------


class OPTI:
    def __init__(self, x_train_val, y_train_val, classifier, num=1, score='AUC'):
        global nuum
        try:
            global search_space, best, trials, obj
            if classifier == Randomforest:
                search_space = { 'bootstrap': hp.choice("bootstrap", [True, False]),
                                 'criterion': hp.choice('criterion', ['entropy', 'gini']),
                                 'max_depth': hp.choice('max_depth', list(range(3, 30))),
                                 'max_features': hp.choice('max_features', ['sqrt', 'log2']),
                                 'min_samples_leaf': hp.choice('min_samples_leaf', list(range(1, 10))),
                                 'min_samples_split': hp.choice('min_samples_split', list(range(2, 10))),
                                 'n_estimators': hp.choice('n_estimators', list(range(20, 500))),
                                 }

            elif classifier == SVM:
                search_space = { 'C': hp.uniform("C", 1, 1000),
                                 'kernel': hp.choice('kernel', ['linear', 'poly', 'rbf']),
                                 'degree': hp.choice('degree', [2, 3]),
                                 'gamma': hp.uniform('gamma', 1e-4, 10)
                                 }

            elif classifier == XGBoost:
                search_space = { 'colsample_bytree': hp.uniform('colsample_bytree', 0.1, 0.9),
                                 'gamma': hp.uniform('gamma', 0, 0.4),
                                 'learning_rate': hp.uniform('learning_rate', 0.001, 0.1),
                                 'max_depth': hp.choice('max_depth', list(range(3, 50))),
                                 'min_child_weight': hp.choice('min_child_weight', list(range(2, 7))),
                                 'n_estimators': hp.choice('n_estimators', list(range(20, 500))),
                                 'subsample': hp.uniform('subsample', 0.2, 0.9),
                                 }

            elif classifier == MLP:
                search_space = { 'activation': hp.choice('activation', ['tanh', 'relu']),
                                 'alpha': hp.uniform('alpha', 0.001, 1),
                                 'hidden_layer_sizes': hp.choice('hidden_layer_sizes', [(20,), (25,), (30,),
                                                                                        (20, 20,), (25, 25,), (30, 30,),
                                                                                        (20, 20, 20,), (25, 25, 25,), (30, 30, 30,)]),
                                 'learning_rate_init': hp.uniform('learning_rate_init', 0.001, 0.1),
                                 }

            if score == 'AUC':
                def scoring(estimator, x_val, y_val):
                    probs = estimator.predict_proba(x_val)
                    y_val = np.asarray(pd.get_dummies(y_val))
                    # Compute micro-average ROC curve and ROC area
                    fpr, tpr, _ = roc_curve(y_val.ravel(), probs.ravel())
                    roc_auc = np.round(auc(fpr, tpr), 3)
                    return roc_auc
            else:
                def scoring(estimator, x_val, y_val):
                    y_val_pred = estimator.predict(x_val)
                    cm = confusion_matrix(y_val, y_val_pred)
                    class_acc = np.round(cm.diagonal() / cm.sum(axis=1), 3)
                    avr = np.round(np.average(class_acc), 3)
                    return avr

            # @use_named_args(search_space)
            def objective(params):
                global model
                if classifier == Randomforest:
                    model = RandomForestClassifier(random_state=0)
                elif classifier == SVM:
                    model = SVC(probability=True, random_state=0)
                elif classifier == XGBoost:
                    model = xgb.XGBClassifier(objective='multi:softprob', random_state=0, eval_metric='mlogloss')
                elif classifier == MLP:
                    model = MLPClassifier(random_state=0, max_iter=5000)

                model.set_params(**params)

                cv = StratifiedShuffleSplit(n_splits=4, random_state=0)
                result = cross_validate(model, x_train_val, y_train_val,
                                         cv=cv, n_jobs=-1, scoring=scoring, return_train_score=True)
                estimate = np.mean(result['test_score'])
                return -estimate

            @timeout(310)
            def opt(objective, search_space, rs):
                trials = Trials()
                best = fmin(fn=objective,
                            space=search_space,
                            algo=tpe.suggest,
                            max_evals=30,
                            timeout=300,
                            trials=trials,
                            rstate=np.random.RandomState(rs))
                obj = np.asarray(sorted(trials.losses())[0])
                return best, obj

            i = 0
            while i < 15:
                try:
                    best, obj = opt(objective, search_space, i)
                except Exception as e:
                    i += 1
                    print(traceback.format_exc())
                    print('Try another run!')
                    print(e)
                    continue
                else:
                    break

        except Exception:
            print(traceback.format_exc())
        else:
            self.bestparams = np.asarray(list(best.values()))
            print('BEST:', best)
            self.bestfun = obj
            print('Best parameters:', self.bestparams)
            print('Max {}:'.format(score), self.bestfun)



