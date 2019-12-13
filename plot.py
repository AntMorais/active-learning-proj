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

from sklearn.utils import indexable
from sklearn.utils.validation import _num_samples
from libact.base.dataset import Dataset, import_libsvm_sparse
from libact.models import *
from libact.query_strategies import *
from libact.labelers import IdealLabeler


def run(trn_ds, tst_ds, lbr, model, qs, quota):
    E_in, E_out = [], []

    for _ in range(quota):
        # Standard usage of libact objects
        ask_id = qs.make_query()
        X, _ = zip(*trn_ds.data)
        lb = lbr.label(X[ask_id])
        trn_ds.update(ask_id, lb)

        model.train(trn_ds)
        #print("score: " + str(model.score(tst_ds)))
        # E_in =1
        # E_in =1
        E_in = np.append(E_in, 1 - model.score(trn_ds))
        E_out = np.append(E_out, 1 - model.score(tst_ds))
    
    return E_in, E_out


def split_train_test(dataset_filepath, test_size, n_labeled):
    X, y = import_libsvm_sparse(dataset_filepath).format_sklearn()
    
    X_train, X_test, y_train, y_test = \
        train_test_split(X, y, test_size=test_size)
    
    ###########################################################################
    #escolher as samples apenas da classe 0 e 1
    itemsNotUsed = list(range(n_labeled, len(y_train)))
    class2remove = float(max(y_train)) #guarda a classe a nao ter no dataset
    
    for i in range(n_labeled):
        if y_train[i] == class2remove:
            #trocar i com outro elemento de outra classe
            features2remove = X_train[i]
            for j in range(n_labeled, len(y_train)):
                #encontra primeira instancia (ainda nao trocada) de uma classe diferente daquela a retirar
                if y_train[j] != class2remove:
                    #remover item da lista (para nao poder ser reutilizado)
                    if j in itemsNotUsed:
                        #faz as trocas
                        itemsNotUsed.remove(j)
                        X_train[i] = X_train[j] 
                        X_train[j] = features2remove
                        y_train[i] = y_train[j]
                        y_train[j] = class2remove
                        break
                    else:
                        continue


    ###########################################################################
    trn_ds = Dataset(X_train, np.concatenate(
        [y_train[:n_labeled], [None] * (len(y_train) - n_labeled)]))
    print(trn_ds[:][1])
    tst_ds = Dataset(X_test, y_test)
    fully_labeled_trn_ds = Dataset(X_train, y_train)
    print(fully_labeled_trn_ds[:][1])
    return trn_ds, tst_ds, y_train, fully_labeled_trn_ds


def main():
    # Specifiy the parameters here:
    # path to your binary classification dataset
    dataset_filepath = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'libsvmcopy.txt')
    test_size = 0.33   # the percentage of samples in the dataset that will be
    # randomly selected and assigned to the test set
    n_labeled = 10     # number of samples that are initially labeled

    # Load dataset
    trn_ds, tst_ds, y_train, fully_labeled_trn_ds = \
        split_train_test(dataset_filepath, test_size, n_labeled)
    trn_ds2 = copy.deepcopy(trn_ds)
    lbr = IdealLabeler(fully_labeled_trn_ds)

    quota = len(y_train) - n_labeled    # number of samples to query
    print("quota = "+ str(quota))
    # Comparing UncertaintySampling strategy with RandomSampling.
    # model is the base learner, e.g. LogisticRegression, SVM ... etc.
    qs = UncertaintySampling(trn_ds, method='euclidean', model=LogisticRegression())
    model = LogisticRegression()
    E_in_1, E_out_1 = run(trn_ds, tst_ds, lbr, model, qs, quota)
    # print("E_out_1 = "+str(E_out_1))
    qs2 = RandomSampling(trn_ds2)
    model = LogisticRegression()
    E_in_2, E_out_2 = run(trn_ds2, tst_ds, lbr, model, qs2, quota)


    # Plot the learning curve of UncertaintySampling to RandomSampling
    # The x-axis is the number of queries, and the y-axis is the corresponding
    # error rate.
    query_num = np.arange(1, quota + 1)
    plt.plot(query_num, E_in_1, 'b', label='euclidean Ein')
    plt.plot(query_num, E_in_2, 'r', label='random Ein')
    plt.plot(query_num, E_out_1, 'g', label='euclidean Eout')
    plt.plot(query_num, E_out_2, 'k', label='random Eout')
    plt.xlabel('Number of Queries')
    plt.ylabel('Error')
    plt.title('Experiment Result')
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
               fancybox=True, shadow=True, ncol=5)
    plt.show()

# def train_test_split_OSR(*arrays, **options):
    
#     n_arrays = len(arrays)
#     if n_arrays == 0:
#         raise ValueError("At least one array required as input")

#     test_size = options.pop('test_size', None) 
#     train_size = options.pop('train_size', None)
#     random_state = options.pop('random_state', None)
    
#     if options:
#         raise TypeError("Invalid parameters passed: %s" % str(options))
 
#     if test_size is None and train_size is None:
#         test_size = 0.25
#     arrays = indexable(*arrays)
    
#     n_samples = _num_samples(arrays[0])
#     print(n_samples)
#     #selecionar apenas  as samples com duas classes e prosseguir
#     cv = ShuffleSplit(n_samples, test_size=0,
#                         train_size=train_size,
#                         random_state=random_state)
#     train, test = next(iter(cv))

#     return list(chain.from_iterable((safe_indexing(a, train),
#                                      safe_indexing(a, test)) for a in arrays))

if __name__ == '__main__':
    main()