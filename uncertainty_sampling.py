""" Uncertainty Sampling

This module contains a class that implements two of the most well-known
uncertainty sampling query strategies: the least confidence method and the
smallest margin method (margin sampling).

"""
import numpy as np
import scipy as sc

from libact.base.interfaces import QueryStrategy, ContinuousModel, \
    ProbabilisticModel
from libact.utils import inherit_docstring_from, zip


class UncertaintySampling(QueryStrategy):

    """Uncertainty Sampling

    This class implements Uncertainty Sampling active learning algorithm [1]_.

    Parameters
    ----------
    model: :py:class:`libact.base.interfaces.ContinuousModel` or :py:class:`libact.base.interfaces.ProbabilisticModel` object instance
        The base model used for training.

    method: {'lc', 'sm', 'entropy'}, optional (default='lc')
        least confidence (lc), it queries the instance whose posterior
        probability of being positive is nearest 0.5 (for binary
        classification);
        smallest margin (sm), it queries the instance whose posterior
        probability gap between the most and the second probable labels is
        minimal;
        entropy, requires :py:class:`libact.base.interfaces.ProbabilisticModel`
        to be passed in as model parameter;


    Attributes
    ----------
    model: :py:class:`libact.base.interfaces.ContinuousModel` or :py:class:`libact.base.interfaces.ProbabilisticModel` object instance
        The model trained in last query.


    Examples
    --------
    Here is an example of declaring a UncertaintySampling query_strategy
    object:

    .. code-block:: python

       from libact.query_strategies import UncertaintySampling
       from libact.models import LogisticRegression

       qs = UncertaintySampling(
                dataset, # Dataset object
                model=LogisticRegression(C=0.1)
            )

    Note that the model given in the :code:`model` parameter must be a
    :py:class:`ContinuousModel` which supports predict_real method.


    References
    ----------

    .. [1] Settles, Burr. "Active learning literature survey." University of
           Wisconsin, Madison 52.55-66 (2010): 11.
    """

    def __init__(self, *args, **kwargs):
        super(UncertaintySampling, self).__init__(*args, **kwargs)

        self.model = kwargs.pop('model', None)
        if self.model is None:
            raise TypeError(
                "__init__() missing required keyword-only argument: 'model'"
            )
        if not isinstance(self.model, ContinuousModel) and \
                not isinstance(self.model, ProbabilisticModel):
            raise TypeError(
                "model has to be a ContinuousModel or ProbabilisticModel"
            )

        self.model.train(self.dataset)

        self.method = kwargs.pop('method', 'lc')
        if self.method not in ['lc', 'sm', 'entropy', 'euclidean']:
            raise TypeError(
                "supported methods are ['lc', 'sm', 'entropy', 'euclidean'], the given one "
                "is: " + self.method
            )

        if self.method=='entropy' and \
                not isinstance(self.model, ProbabilisticModel):
            raise TypeError(
                "method 'entropy' requires model to be a ProbabilisticModel"
            )

    def _get_scores(self):
        dataset = self.dataset
        self.model.train(dataset)
        unlabeled_entry_ids, X_pool = dataset.get_unlabeled_entries()

        if isinstance(self.model, ProbabilisticModel):
            dvalue = self.model.predict_proba(X_pool)
        elif isinstance(self.model, ContinuousModel):
            dvalue = self.model.predict_real(X_pool)

        if self.method == 'lc':  # least confident
            score = -np.max(dvalue, axis=1)

        elif self.method == 'sm':  # smallest margin
            if np.shape(dvalue)[1] > 2:
                # Find 2 largest decision values
                dvalue = -(np.partition(-dvalue, 2, axis=1)[:, :2])
            score = -np.abs(dvalue[:, 0] - dvalue[:, 1])

        elif self.method == 'entropy':
            
            score = np.sum(-dvalue * np.log(dvalue), axis=1)
        return zip(unlabeled_entry_ids, score)


    def make_query(self, return_score=False):
        """Return the index of the sample to be queried and labeled and
        selection score of each sample. Read-only.

        No modification to the internal states.

        Returns
        -------
        ask_id : int
            The index of the next unlabeled sample to be queried and labeled.

        score : list of (index, score) tuple
            Selection score of unlabled entries, the larger the better.

        """
        dataset = self.dataset
        
        # unlabeled_entry_ids, _ = dataset.get_unlabeled_entries()
        
        ############################################################################
        if(self.method == 'euclidean'):
            self.model.train(dataset)
            #caso usemos a distancia euclidiana e metricas de interesse
            unlabeled_entry_ids, X_pool = dataset.get_unlabeled_entries()
            X_labeled, y_labeled = dataset.get_labeled_entries()
            
            #prototipos (media)
            classes = np.unique(dataset._y[dataset.get_labeled_mask()])
            #lista para acumular valores de cada feature
            list_acum = []
            #list acumulador de isntancias de classes
            counter_classes = []
            for i in range(len(classes)):
                counter_classes.append(0)
                list_acum.append([0] * len(X_labeled[0]))

            #lista que guarda medias das features de cada classe
            prot = []
            acum = 0

            #percorrer as instancias labeled e acumular os valores e dividir no fim
            for i in range(len(X_labeled)):
                for j in classes:
                    if y_labeled[i] == j:
                        counter_classes[int(j)] += 1
                        j = int(j)
                        #acumular para fazer media (cada feature)
                        for k in range(len(X_labeled[0])):
                            list_acum[j][k] += X_labeled[i][k]
                print(list_acum)
            print(list_acum)
            for i in range(len(list_acum)):
                prot.append([x / counter_classes[i] for x in list_acum[i]])
            
            #interesse - METRICAS
            print("inst atual: " +str(X_pool[0]))
            print("prot 0: " +str(prot[0]))
            corr = np.corrcoef(X_pool[0], prot[0])
            print("corr: ")
            print(corr)
            print("prot 1: " +str(prot[1]))
            corr = np.corrcoef(X_pool[0], prot[1])
            print("corr: ")
            print(corr)
            

            # isto foi o broche do toni que adicionou
            pearson,_ = sc.stats.pearsonr(X_pool[0], prot[1])
            print("pearson: ")
            print(pearson)

            #unlabeled_entry_ids, scores = zip(*self._get_scores())
            #ask_id = np.argmax(scores)
        else:
            unlabeled_entry_ids, scores = zip(*self._get_scores())
            ask_id = np.argmax(scores)
            
        
        ###########################################################################
        
#        unlabeled_entry_ids, scores = zip(*self._get_scores())
#        ask_id = np.argmax(scores)

        if return_score:
            return unlabeled_entry_ids[ask_id], \
                   list(zip(unlabeled_entry_ids, scores))
        else:
            return unlabeled_entry_ids[ask_id]
