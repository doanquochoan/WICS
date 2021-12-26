from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
from sklearn.metrics import roc_curve, auc, cohen_kappa_score
import matplotlib.pyplot as plt
import matplotlib
from mlxtend.plotting import plot_confusion_matrix
import pandas as pd
from imblearn.metrics import geometric_mean_score
import os
import numpy as np


class Metrics:
    def __init__(self, X_train, Y_train, X_test, Y_test, Y_pred):
        self.X_train = X_train
        self.Y_train = Y_train
        self.X_test = X_test

        self.Y_test = Y_test
        self.Y_pred = Y_pred
        # Create an output folder and its path
        out_dir = os.path.join(os.getcwd(), 'Output')
        if os.path.exists(out_dir):
            self.output_path = out_dir
        else:
            os.mkdir('Output')
            self.output_path = out_dir

    def Fmeasure(self):
        f1 = f1_score(self.Y_test, self.Y_pred, average='macro')
        return np.round(f1, 3)

    def Precision(self):
        precision = precision_score(self.Y_test, self.Y_pred, average='macro')
        return np.round(precision, 3)

    def Precision_each(self):
        precision = precision_score(self.Y_test, self.Y_pred, average=None)
        return np.round(precision, 3)

    def Recall(self):
        recall = recall_score(self.Y_test, self.Y_pred, average='macro')
        return np.round(recall, 3)

    def Recall_each(self):
        recall = recall_score(self.Y_test, self.Y_pred, average=None)
        return np.round(recall, 3)

    def Gmean(self):
        gmean = geometric_mean_score(self.Y_test, self.Y_pred)
        return np.round(gmean, 3)

    def ConfusionMatrix(self, name='one', iter=1, save=True):
        cm = confusion_matrix(self.Y_test, self.Y_pred)
        if save:
            matplotlib.rcParams.update({'font.size': 18})
            fig, ax = plot_confusion_matrix(conf_mat=cm, colorbar=True, show_normed=True)
            plt.pause(1e-13)
            plt.savefig(self.output_path + "/cm_{}_{}.jpg".format(name, iter), dpi=600, bbox_inches='tight')
            plt.close(fig)
        else:
            pass
        # Class accuracy
        class_acc = np.round(cm.diagonal() / cm.sum(axis=1), 3)
        return class_acc, np.round(np.average(class_acc), 3)

    def AUC(self, probs):
        Y_test = np.asarray(pd.get_dummies(self.Y_test))
        # Compute micro-average ROC curve and ROC area
        fpr, tpr, _ = roc_curve(Y_test.ravel(), probs.ravel())

        roc_auc = np.round(auc(fpr, tpr), 3)
        return roc_auc, fpr, tpr

    def Kappa(self):
        kappa = cohen_kappa_score(self.Y_test, self.Y_pred)
        return np.round(kappa, 3)







