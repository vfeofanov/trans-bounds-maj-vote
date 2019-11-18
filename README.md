# Transductive Bounds for the Multi-class Majority Vote Classifier
This repository is devoted to implementation of the approach proposed in:
Vasilii Feofanov, Emilie Devijver and Massih-Reza Amini - <a href="https://aaai.org/ojs/index.php/AAAI/article/view/4236" target="_blank">Transductive Bounds for the Multi-class Majority Vote Classifier</a>. In Proceedings of the AAAI Conference on Artificial Intelligence 33, 3566-3573.

## Multi-class Self-learning Algorithm (MSLA)
The multi-class semi-supervised framework is considered. The goal is to infer a model based on given few labeled examples and lots of unlabeled ones. The proposed algorithm iteratively assigns pseudo-labels to a subset of unlabeled training examples that have their associated class margin above a threshold obtained from the transductive bound proposed in the <a href="https://aaai.org/ojs/index.php/AAAI/article/view/4236" target="_blank"> paper</a>. The algorithm is based on a supervised approach that can be any classifier that outputs posteriors. In our implementation, we use the <a href="https://www.stat.berkeley.edu/~breiman/randomforest2001.pdf" target="_blank"> random forest</a> approach. 

## Code
The algorithm is implemented in Python 3 and can be found at [self_learning.py](https://github.com/vfeofanov/trans-bounds-maj-vote/blob/master/self_learning.py). Some functions are re-written in Cython to reduce runtime and are located at [self_learning_cython.pyx](https://github.com/vfeofanov/trans-bounds-maj-vote/blob/master/self_learning_cython.pyx). By default, the msla function makes use of Cython, which can be manually changed if there is no possibility to install Cython package.

#### Dependencies

To run succesfully the code, it requires:

* Python 3
* scikit-learn
* NumPy
* Pandas
* Matplotlib (only for plotting)
* Cython (optional)

## Experiments
To validate our approach, we compare the MSLA algorithm with the following methods:
1. Purely supervised approach. It is a scikit-learn implementation of the [Random Forest](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html).
2. The [Label Propagation](https://scikit-learn.org/dev/modules/generated/sklearn.semi_supervised.LabelPropagation.html#sklearn.semi_supervised.LabelPropagation) by scikit-learn.
3. [Transductive SVM](http://svmlight.joachims.org) extended to the multi-class case by the one-versus-all approach.
4. [FSLA](https://github.com/vfeofanov/trans-bounds-maj-vote/blob/master/self_learning.py): the self-learning algorithm with a fixed threshold equal to 0.7.

#### 1. Simple Test
This test can be executed to verify that the code works on a machine. It performs classification using all classifiers under consideration on the DNA dataset with one random split on labeled and ulabeled data. In a terminal, the code is executed in the following way:
```bash
python3 simple_test.py
```
In case of success, the basic information will be displayed as well as the following graph:
<img src="https://github.com/vfeofanov/trans-bounds-maj-vote/blob/master/plots/performance_plot.jpg" alt="The performance results of the simple test" width="700"/>

In addition, the folder with TSVM input and output for each class will be created. 

#### 2. Experiment Test
This test performs experiments with the setup described in the <a href="https://aaai.org/ojs/index.php/AAAI/article/view/4236" target="_blank"> paper</a>. 20 random splits on labeled and unlabeled parts are performed for a dataset. To run the test in a terminal you specify two arguments: name of a dataset, and the labeled/unlabeled split. For instance, for the dataset pendigits with the split 0.99, one types the following: 
```bash
python3 experiment_test.py pendigits 0.99
```
The output will create several file in the output folder.
