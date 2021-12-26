from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from Test_performance import Metrics
import pandas as pd
import xgboost as xgb
from sklearn.neural_network import MLPClassifier


class Randomforest:
    def __init__(self, X_train, Y_train, X_test, Y_test, iter=1, savecm=False, opti=True, hyperpa=None):
        # global individual
        individual = hyperpa

        bootstrap = individual[0]
        if bootstrap == 0:
            bootstrap = True
        elif bootstrap == 1:
            bootstrap = False

        crit = individual[1]
        if crit == 0:
            crit = 'entropy'
        elif crit == 1:
            crit = 'gini'

        depth = individual[2] + 3
        features = individual[3]
        if features == 0:
            features = 'sqrt'
        elif features == 1:
            features = 'log2'

        samples_leaf = individual[4] + 1
        samples_split = individual[5] + 2
        n_estimator = individual[6] + 20

        model = RandomForestClassifier(n_estimators=n_estimator, bootstrap=bootstrap, criterion=crit,
                                       max_depth=depth, min_samples_split=samples_split,
                                       min_samples_leaf=samples_leaf, max_features=features, random_state=0)

        model.fit(X_train, Y_train)
        Y_pred = model.predict(X_test)
        probs = model.predict_proba(X_test)
        metrics = Metrics(X_train, Y_train, X_test, Y_test, Y_pred)
        self.Fmeasure = metrics.Fmeasure()
        self.Gmean = metrics.Gmean()
        self.Class_acc, self.avr = metrics.ConfusionMatrix(name='RF', iter=iter, save=savecm)
        self.AUC, self.fpr, self.tpr = metrics.AUC(probs)
        self.Precision = metrics.Precision()
        self.Precision_each = metrics.Precision_each()
        self.Recall = metrics.Recall()
        self.Recall_each = metrics.Recall_each()
        self.Kappa = metrics.Kappa()

    def performance(self, case='Original'):
        output = { 'Precision': self.Precision, 'Recall': self.Recall, 'Fmeasure': self.Fmeasure,
                   'Kappa': self.Kappa, 'Gmean': self.Gmean, 'Average class acc': self.avr, 'AUC': self.AUC}
        output = pd.DataFrame.from_dict(output, orient='index', columns=[case])
        return output

    def output_recall(self):
        output_recall = pd.DataFrame(self.Recall_each)
        return output_recall


class XGBoost:
    def __init__(self, X_train, Y_train, X_test, Y_test, iter=1, savecm=False, opti=True, hyperpa=None):

        individual = hyperpa
        colsample_bytree = individual[0]
        gamma = individual[1]
        learning_rate = individual[2]
        max_depth = int(individual[3]) + 3
        min_child_weight = int(individual[4]) + 2
        n_estimator = int(individual[5]) + 20
        subsample = individual[6]

        model = xgb.XGBClassifier(n_estimators=n_estimator, max_depth=max_depth, learning_rate=learning_rate,
                                  objective='multi:softprob', booster='gbtree', gamma=gamma,
                                  min_child_weight=min_child_weight, subsample=subsample,
                                  colsample_bytree=colsample_bytree, random_state=0, use_label_encoder=False)

        model.fit(X_train, Y_train, eval_metric='mlogloss')
        Y_pred = model.predict(X_test)
        probs = model.predict_proba(X_test)
        metrics = Metrics(X_train, Y_train, X_test, Y_test, Y_pred)
        self.f_im = model.feature_importances_
        self.Fmeasure = metrics.Fmeasure()
        self.Gmean = metrics.Gmean()
        self.Class_acc, self.avr = metrics.ConfusionMatrix(name='XGBoost', iter=iter, save=savecm)
        self.AUC, self.fpr, self.tpr = metrics.AUC(probs)
        self.Precision = metrics.Precision()
        self.Precision_each = metrics.Precision_each()
        self.Recall = metrics.Recall()
        self.Recall_each = metrics.Recall_each()
        self.Kappa = metrics.Kappa()

        feature_important = model.get_booster().get_score(importance_type='weight')
        keys = list(feature_important.keys())
        values = list(feature_important.values())
        # data = pd.DataFrame(data=values, index=keys, columns=["score"]).sort_values(by="score", ascending=False)
        data = pd.DataFrame(data=values, index=keys, columns=["score"])
        # print(data)
        # data.plot(kind='barh')
        # print(self.f_im)
        # xgb.plot_importance(model)
        # plt.show()

    def performance(self, case='Original'):
        output = { 'Precision': self.Precision, 'Recall': self.Recall, 'Fmeasure': self.Fmeasure,
                   'Kappa': self.Kappa, 'Gmean': self.Gmean, 'Average class acc': self.avr, 'AUC': self.AUC }
        output = pd.DataFrame.from_dict(output, orient='index', columns=[case])
        return output

    def output_recall(self):
        output_recall = pd.DataFrame(self.Recall_each)
        return output_recall

    def out_f_im(self):
        out_f_im = pd.DataFrame(self.f_im)
        return out_f_im


class SVM:
    def __init__(self, X_train, Y_train, X_test, Y_test, iter=1, savecm=False, opti=False, hyperpa=None):

        individual = hyperpa
        C = float(individual[0])
        degree = int(individual[1]) + 2
        gamma = float(individual[2])
        kernel = individual[3]
        if kernel == 0:
            kernel = 'linear'
        elif kernel == 1:
            kernel = 'poly'
        elif kernel == 2:
            kernel = 'rbf'

        model = SVC(C=C, kernel=kernel, degree=degree, gamma=gamma,
                    probability=True, random_state=0)

        model.fit(X_train, Y_train)
        Y_pred = model.predict(X_test)
        probs = model.predict_proba(X_test)
        metrics = Metrics(X_train, Y_train, X_test, Y_test, Y_pred)
        self.Fmeasure = metrics.Fmeasure()
        self.Gmean = metrics.Gmean()
        self.Class_acc, self.avr = metrics.ConfusionMatrix(name='SVM', iter=iter, save=savecm)
        self.AUC, self.fpr, self.tpr = metrics.AUC(probs)
        self.Precision = metrics.Precision()
        self.Precision_each = metrics.Precision_each()
        self.Recall = metrics.Recall()
        self.Recall_each = metrics.Recall_each()
        self.Kappa = metrics.Kappa()

    def performance(self, case='Original'):
        output = { 'Precision': self.Precision, 'Recall': self.Recall, 'Fmeasure': self.Fmeasure,
                   'Kappa': self.Kappa, 'Gmean': self.Gmean, 'Average class acc': self.avr, 'AUC': self.AUC }
        output = pd.DataFrame.from_dict(output, orient='index', columns=[case])
        return output

    def output_recall(self):
        output_recall = pd.DataFrame(self.Recall_each)
        return output_recall


class MLP:
    def __init__(self, X_train, Y_train, X_test, Y_test, iter=1, savecm=True, opti=False, hyperpa=None):

        individual = hyperpa
        activation = individual[0]
        if activation == 0:
            activation = 'tanh'
        elif activation == 1:
            activation = 'relu'

        alpha = individual[1]

        hidden_layer_sizes = int(individual[2]) + 10

        learning_rate_init = individual[3]

        model = MLPClassifier(random_state=0, activation=activation, alpha=alpha, learning_rate_init=learning_rate_init,
                              solver='adam', hidden_layer_sizes=(hidden_layer_sizes,))

        model.fit(X_train, Y_train)
        Y_pred = model.predict(X_test)
        probs = model.predict_proba(X_test)
        metrics = Metrics(X_train, Y_train, X_test, Y_test, Y_pred)
        self.Fmeasure = metrics.Fmeasure()
        self.Gmean = metrics.Gmean()
        self.Class_acc, self.avr = metrics.ConfusionMatrix(name='MLP', iter=iter, save=savecm)
        self.AUC, self.fpr, self.tpr = metrics.AUC(probs)
        self.Precision = metrics.Precision()
        self.Precision_each = metrics.Precision_each()
        self.Recall = metrics.Recall()
        self.Recall_each = metrics.Recall_each()
        self.Kappa = metrics.Kappa()

    def performance(self, case='Original'):
        output = { 'Precision': self.Precision, 'Recall': self.Recall, 'Fmeasure': self.Fmeasure,
                   'Kappa': self.Kappa, 'Gmean': self.Gmean, 'Average class acc': self.avr, 'AUC': self.AUC }
        output = pd.DataFrame.from_dict(output, orient='index', columns=[case])
        return output

    def output_recall(self):
        output_recall = pd.DataFrame(self.Recall_each)
        return output_recall


