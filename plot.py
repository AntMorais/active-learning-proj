#!/usr/bin/env python3
"""
The script helps guide the users to quickly understand how to use
libact by going through a simple active learning task with clear
descriptions.
"""

import copy
import os

import numpy as np
import matplotlib.pyplot as plt
try:
    from sklearn.model_selection import train_test_split
except ImportError:
    from sklearn.cross_validation import train_test_split

# libact classes
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc
from scipy import interp
from itertools import cycle
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score 
from sklearn.utils import indexable
from sklearn.utils.validation import _num_samples
from libact.base.dataset import Dataset, import_libsvm_sparse
from libact.models import *
from libact.query_strategies import *
from libact.labelers import IdealLabeler


def run(trn_ds, tst_ds, lbr, model, qs, quota, title):
    E_f1 = []
    for _ in range(quota):
       
        # Standard usage of libact objects
        ask_id = qs.make_query()
        #f = open("test.txt", "a")
        X, _ = zip(*trn_ds.data)

        lb = lbr.label(X[ask_id])
        
        # f.write("\nda classe:"+ str(lb))
        # f.write("\n")

        trn_ds.update(ask_id, lb)

        model.train(trn_ds)
        #E_in = np.append(E_in, 1 - model.score(trn_ds))

        #F1 SCORE
        #E_out = np.append(E_out, 1 - model.score(tst_ds))
        E_f1 = np.append(E_f1, f1_score(tst_ds._y, model.predict(tst_ds._X), average='weighted'))
        #E_auroc = np.append(E_auroc, roc_auc_score(tst_ds._y.astype(int), model.predict(tst_ds._X).astype(int)), multiclass = 'ovr')
        #f.close()
    return E_f1


def split_train_test(dataset_filepath, test_size, n_labeled):
    X, y = import_libsvm_sparse(dataset_filepath).format_sklearn()
    
    X_train, X_test, y_train, y_test = \
        train_test_split(X, y, test_size=test_size)
    
    ###########################################################################
    
    #preciso retirar um classe
    
    #escolher as samples todas as classes menos uma (por agora)
    #print("classes em teste: "+str(np.unique(y_test))+"\n"+"classes em treino: "+str(np.unique(y_train)))
    itemsNotUsed = list(range(n_labeled, len(y_train)))
    class2remove = float(max(y_train)) #guarda a classe a nao ter no dataset

    if(len(np.unique(y_train[:n_labeled])) == len(np.unique(y))):
        #print("tenho todas")
        for i in range(n_labeled):
            #print(y_train[i])
            if y_train[i] == class2remove:
                #trocar i com outro elemento de outra classe
                features2remove = X_train[i]
                for j in range(n_labeled, len(y_train)):
                    #encontra primeira instancia (ainda nao trocada) de uma classe diferente daquela a retirar
                    if y_train[j] != class2remove:
                        #remover item da lista (para nao poder ser reutilizado)
                        #faz as trocas
                        itemsNotUsed.remove(j)
                        X_train[i] = X_train[j] 
                        X_train[j] = features2remove
                        y_train[i] = y_train[j]
                        y_train[j] = class2remove
                        break
    print(np.unique(y_train[:n_labeled]))

    ###########################################################################
    trn_ds = Dataset(X_train, np.concatenate(
        [y_train[:n_labeled], [None] * (len(y_train) - n_labeled)]))
    # print(trn_ds[:][1])
    tst_ds = Dataset(X_test, y_test)
    fully_labeled_trn_ds = Dataset(X_train, y_train)
    # print(fully_labeled_trn_ds[:][1])
    return trn_ds, tst_ds, y_train, fully_labeled_trn_ds


def main():
    # Specifiy the parameters here:
    # path to your binary classification dataset
    #'winequality-white_libsvm.txt'
    dataset_filepath = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'winequality-white_libsvm.txt')
    test_size = 0.15   # the percentage of samples in the dataset that will be
    # randomly selected and assigned to the test set
    n_labeled = 30    # number of samples that are initially labeled

#iterar 5 vezes e calcular media

    # Load dataset
    trn_ds, tst_ds, y_train, fully_labeled_trn_ds = \
        split_train_test(dataset_filepath, test_size, n_labeled)
    trn_ds2 = copy.deepcopy(trn_ds)
    trn_ds3 = copy.deepcopy(trn_ds)
    lbr = IdealLabeler(fully_labeled_trn_ds)

    quota = len(y_train) - n_labeled    # number of samples to query
    # print("quota = "+ str(quota))
    # Comparing EUCLIDEAN(interest) strategy with RandomSampling.
    # model is the base learner, e.g. LogisticRegression, SVM ... etc.
    qs = UncertaintySampling(trn_ds, method='euclidean', model=LogisticRegression())
    model = LogisticRegression()
    E_f1_1= run(trn_ds, tst_ds, lbr, model, qs, quota, 'Euclidean')
    #AUROC
    auroc(model.predict(tst_ds._X), tst_ds._y, 'Euclidean')

    #RANDOM
    qs2 = RandomSampling(trn_ds2)
    model2 = LogisticRegression()
    E_f1_2 = run(trn_ds2, tst_ds, lbr, model2, qs2, quota, 'Random')
    #AUROC
    auroc(model2.predict(tst_ds._X), tst_ds._y, 'Random')

    #UNCERTAINTY
    qs3 = UncertaintySampling(trn_ds3, method='entropy', model=LogisticRegression())
    model3 = LogisticRegression()
    E_f1_3 = run(trn_ds3, tst_ds, lbr, model3, qs3, quota, 'Uncertainty')
    #AUROC
    auroc(model3.predict(tst_ds._X), tst_ds._y, 'Uncertainty')

    if model.predict(tst_ds._X).all() == (model2.predict(tst_ds._X).all() == model3.predict(tst_ds._X).all()) :
        print("tinganei")
    # Plot the learning curve of UncertaintySampling to RandomSampling
    # The x-axis is the number of queries, and the y-axis is the corresponding
    # error rate
    query_num = np.arange(1, quota + 1)
    
    #F1 SCOREq
    plt.figure()
    plt.plot(query_num, E_f1_1, 'g', label='euclidean Eout')
    plt.plot(query_num, E_f1_2, 'k', label='random Eout')
    plt.plot(query_num, E_f1_3, 'c', label='entropy Eout')
    
    plt.xlabel('Number of Queries')
    plt.ylabel('F1')
    plt.title('F1 - Wine')
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
               fancybox=True, shadow=True, ncol=5)
    plt.show()

def auroc(y_test_predict, y_test_true, titulo):
    #output -> binario
    y_test_predict = label_binarize(y_test_predict, np.unique(y_test_true))
    y_test_true = label_binarize(y_test_true, np.unique(y_test_true))
    n_classes = y_test_true.shape[1]
    
    #ROC
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test_true[:, i], y_test_predict[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    #macro average
    #false positives
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

    #Interpolate all ROC curves
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= n_classes
    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
    plt.figure()
    
    plt.plot(fpr["macro"], tpr["macro"],
            label='macro-average ROC curve (area = {0:0.2f})'
                ''.format(roc_auc["macro"]),
            color='navy', linestyle=':', linewidth=4)

    colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                label='ROC curve of class {0} (area = {1:0.2f})'
                ''.format(i, roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('AUROC - '+titulo)
    plt.legend(loc="lower right")
    

if __name__ == '__main__':
    main()