""" Uncertainty Sampling

This module contains a class that implements two of the most well-known
uncertainty sampling query strategies: the least confidence method and the
smallest margin method (margin sampling).

"""
import numpy as np
import scipy
import math

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

        self.metric = kwargs.pop('metric', 'Max')
        if self.metric not in ['Max', 'Weight', 'Sum']:
            raise TypeError(
                "supported metrics are ['Max', 'Weight', 'Sum'], the given one "
                "is: " + self.metric
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
        
        ############################################################################
        #EUCLIDEANA + METRICAS DE INTERESSE
        if(self.method == 'euclidean'):
            self.model.train(dataset)

            #caso usemos a distancia euclidiana e metricas de interesse
            unlabeled_entry_ids, X_pool = dataset.get_unlabeled_entries()
            X_labeled, y_labeled = dataset.get_labeled_entries()
            
            #CALCULAR prototipos (media das features)
            classes = np.unique(dataset._y[dataset.get_labeled_mask()])
            #lista para acumular valores de cada feature
            list_acum = []
            #lista guarda numero de instancias de cada classe (dar return para poder usar no plot????)
            counter_classes = []
            for i in range(len(classes)):
                counter_classes.append(0)
                list_acum.append([0] * len(X_labeled[0]))

            #lista que guarda medias das features de cada classe (prototipos)
            prot = []
            acum = 0

            #percorrer as instancias labeled e acumular os valores (cada feature da mesma classe) e dividir no fim _> prototipos
            for i in range(len(X_labeled)):
                for j in range(len(classes)):
                    if y_labeled[i] == classes[j]:
                        counter_classes[j] += 1 #atualiza o numero de instancias
                        #acumular para fazer media (cada feature)
                        #X_labeled[0] _> numero de features
                        for k in range(len(X_labeled[0])):
                            list_acum[j][k] += X_labeled[i][k]
            
            #guardar em prot os prototipos (medias das features)
            for i in range(len(list_acum)):
                prot.append([x / counter_classes[i] for x in list_acum[i]])
            
            #interesse - METRICAS
            euclidean_dist = []
            acum = []
            #similarity for each instance to be from an unknown class
            sim_unknown_global =[]
            interest_unknow_incertainty = []
            #euclidean distance
            for i in range(len(X_pool)):
                for j in range(len(classes)):
                    #acum - distancias euclidianas entre features
                    acum.append(abs(scipy.spatial.distance.euclidean(X_pool[i], prot[j])))

                #calcula semelhanca 
                #guarda distancia entre a nova e uma classe unknown ()
                sim_unknown_global.append(abs(max(acum)-min(acum)))
                
                #calcula probabilidade
                euclidean_dist.append(acum)
                euclidean_dist_temp = sim_unknown_global[i]

                #probabilidade de ser unknown (similarity)
                #anteriormente -> sim_unknown_global[i] = euclidean_dist_temp/(sum(acum)+euclidean_dist_temp)
                sim_unknown_global[i] = 1/(1+euclidean_dist_temp)

                #calcula interesses
                #unknown = np.tanh(2*sim_unknown_global[i])
                #uncertainty: -sum(p(new = pi)*log(p(new = pi))) 
                prob_classes = []
                
                #calcular semelhancas (sim)
                for j in range(len(classes)):
                    #anteriormente -> prob_classes.append(float(acum[j]/(sum(acum)+euclidean_dist_temp)))
                    prob_classes.append(1/(1+acum[j]))
                #adicionar probabilidade da classe desconhecida
                prob_classes.append(sim_unknown_global[i])
                #calcular probabilidades (normalizar)
                soma_sim = sum(prob_classes)
                for j in range(len(classes)):
                    prob_classes[j] = prob_classes[j]/soma_sim

                #interesse uncertainty (prob_classes[j]*math.log(prob_classes[j])
                prob_classes = [prob_classes[j]*math.log(prob_classes[j]) for j in range(len(prob_classes))]
                #interesse unknown e uncertainty
                if(self.metric=='Max'):
                    #MAX - mudar em baixo, nao esquecer
                    interest_unknow_incertainty.append([np.tanh(2*sim_unknown_global[i]),  -(sum(prob_classes) + (sim_unknown_global[i]*math.log(sim_unknown_global[i])))])
                if(self.metric=='Sum'):
                    #SOMA - mudar em baixo, nao esquecer
                    interest_unknow_incertainty.append(np.tanh(2*sim_unknown_global[i]) +  -(sum(prob_classes) + (sim_unknown_global[i]*math.log(sim_unknown_global[i]))))

                if(self.metric=='Weight'):
                    #WEIGHTED - mudar em baixo, nao esquecer
                    interest_unknow_incertainty.append(np.tanh(2*sim_unknown_global[i])*15 +  -(sum(prob_classes) + (sim_unknown_global[i]*math.log(sim_unknown_global[i]))))
                acum = []
                prob_classes = []
    
            interest_unknow_incertainty_array = np.asarray(interest_unknow_incertainty)
            
            if(self.metric=='Max'):
                ask_id = np.unravel_index(np.argmax(interest_unknow_incertainty_array, axis=None), interest_unknow_incertainty_array.shape)
                #reset no array
                interest_unknow_incertainty = []
                
                #caso seja SOMA ou WEIGHTED -> return unlabeled_entry_ids[ask_id]
                return unlabeled_entry_ids[ask_id[0]]
            else:
                ask_id = np.unravel_index(np.argmax(interest_unknow_incertainty_array, axis=None), interest_unknow_incertainty_array.shape)
                #reset no array
                interest_unknow_incertainty = []
                
                #caso seja SOMA ou WEIGHTED -> return unlabeled_entry_ids[ask_id]
                return unlabeled_entry_ids[ask_id]
        else:
            unlabeled_entry_ids, scores = zip(*self._get_scores())
            ask_id = np.argmax(scores)
            if return_score:
                return unlabeled_entry_ids[ask_id], \
                    list(zip(unlabeled_entry_ids, scores))
            else:
                return unlabeled_entry_ids[ask_id]
            
        
        ###########################################################################

        
