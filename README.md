# active-learning-proj

## Abstract 
Recent surges in the quantity and variety of data lead
to a constant search to find techniques that optimize
the process of training machine learning classifiers.
The cases where there are a lot of unlabelled data
and the cost of annotating is not insignificant are par-
ticularly meaningful. Because of this, there has been
progress in the area of active learning, which is related
to minimizing the cost (computational and otherwise)
of sample annotation. Additionally, the ever-growing
spread of machine learning to real-life cases makes it
important to steer away from static (closed-set) mod-
els and move into dynamic (open-set) models, which
do not limit the classifier to a fixed number of classes.
Since these are 2 heavily researched areas, it is perti-
nent to seek different querying metrics to optimize the
performance of such classification algorithms. On this
paper, we present a continuation of past research on
the usage of unknown interest as described in Macedo
et al. (2011) [1] to bring insight into the comparison
of the performance of several querying metrics.

## Requirements
• Python 2.7 installed
• libact framework installed (https://pypi.org/project/libact/)
• Dataset files in the project directory (abalonelibsvm.txt, satlibsvm.txt,
irislibsvm.txt, winelibsvm.txt)
• uncertaintysampling.py and teste_plot.py files
The teste_plot.py plots the F1-Score value and AUROC (Area Under Receiver
Operating Characteristics) by the number of queries of 3 querying strategies:
Uncertainty, Random and Interest (combination of unknown and uncertainty).
The program runs n_samples times returning the average values in the end.
Consider the size and computational effort required to query so many instances
since we are updating the classes prototypes, calculating distances, probabilities
as-well as the F1-Score and AUROC every time an instance is queried.
Steps:
• Copy the uncertaintysampling.py and replace on the libact directory (libact
> query_strategies) and reinstall libact in order to update the makequery()
function
• run:
o
python teste_plot.py interest_combination dataset_name
n_samples
with:
• interest_combination - WeightN (weighed sum of metrics, N - value of
weight to give to the interest of the unknown, ex. Weight10), Sum (sum of
the interest metrics), Max (max value of the interests)
• dataset_name - iris (150 entries, 3 classes, 4 features), abalone (4177
entries, 3 classes, 8 features), wine (4898 entries, 7 classes, 11 features),
sat (6435 entries, 7 classes, 36 features)
• n_samples - number of times the program runs (it gives an average value
afterwards)
additionally:
• For changing the value of training and testing sizes, open teste_plot.py, on
line 139 and change the value of tst_size (training size will be 1-tst_size)
