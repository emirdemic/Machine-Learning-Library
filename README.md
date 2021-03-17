# Machine Learning Library

## Table of Content
[About The Project](#goal)<br/>
[Tools](#tools)<br/>
[Algorithms](#algorithms)<br/>
[Theoretical Explanations](#theory)<br/>
[Roadmap](#roadmap)<br/>
[References](#references)<br/>

## About The Project <a name="goal"/>
The goal of the project is to create a personal machine learning library populated with different machine learning algorithms, written in Python using only NumPy library for quick and efficient computations. More specifically, the idea is to implement both common and uncommon machine learning algorithms.<br/><br/>
Furthermore, this page contains theoretical and practical explanations of each machine learning algorithm currently populating this repository. This allows anyone using code from this repository to dive into theory behind the code.


## Tools <a name="tools"/>
Python 3.8.5
-  NumPy



## Algorithms<a name="algorithms"/>
This is the full list of algorithms this repository currently holds. Note that not all of them are fully implemented yet :)
-  Regression<br/>
[Linear Regression](https://github.com/emirdemic/Machine-Learning-Library/blob/main/Regression/LinearRegression.py)<br/><br/>
-  Classification<br/>
[Perceptron](https://github.com/emirdemic/Machine-Learning-Library/blob/main/Classification/Perceptron.py)<br/>
[Adaptive Linear Neuron](https://github.com/emirdemic/Machine-Learning-Library/blob/main/Classification/AdaptiveLinearNeuron.py)<br/>
[Logistic Regression](https://github.com/emirdemic/Machine-Learning-Library/blob/main/Classification/LogisticRegression.py)<br/>
[Gaussian Discriminative Analysis](https://github.com/emirdemic/Machine-Learning-Library/blob/main/Classification/GaussianDiscriminantAnalysis.py)<br/>
-  Unsupervised Learning<br/>
[COOLCAT Clustering Algorithm](https://github.com/emirdemic/Machine-Learning-Library/blob/main/UnsupervisedLearning/COOLCAT.py)<br/>
[Multidimensional Scaling](https://github.com/emirdemic/Machine-Learning-Library/blob/main/UnsupervisedLearning/MDS.py)

## Theoretical Explanations<a name="theorys"/><br/>
###COOLCAT Clustering Algorithm<br/>!


The clustering analysis used here is COOLCAT clustering algorithm proposed by 
[Barbara, Couto, & Li (2002)](https://dl.acm.org/doi/abs/10.1145/584792.584888). 
COOLCAT algorithm is used for clustering categorical datasets and is based on a notion of *entropy*. 
More specifically, the entropy of one categorical variable is:

![entropy](https://user-images.githubusercontent.com/57667464/111456984-e7ed0580-8717-11eb-8bfc-042c8a77e70e.png)

Authors assume variable independence, which means that multivariate entropy is equal to
the sum of each variable's entropy. The minimization criterion of the algorithm is the *expected 
entropy of the whole system*:

![expected_entropy](https://user-images.githubusercontent.com/57667464/111457023-f4715e00-8717-11eb-86a5-30d28fc602f2.png)<br/><br/>
<img align="right" width="100" height="100" src="https://user-images.githubusercontent.com/57667464/111457023-f4715e00-8717-11eb-86a5-30d28fc602f2.png")>

where $|C_{k}|$ is the size of cluster $k$ and $|D|$ is the size of dataset. 
In other words, algorithm finds clusters which minimize expected entropy of all clusters.

In order to find such clusters, algorithm goes through two steps: *initialization step* and *incremental step*. 
The initialization step finds two datapoints $p_{1}$ and $p_{2}$ such that their multivariate entropy 
$E(p_{1}, p_{2})$ is the highest possible. These two datapoints will be assigned to clusters $C_{1}$ and $C_{2}$.
Afterwards, algorithm finds new datapoint $p_{j}$ which maximizes $min_{i=1,...,j-1}(E(p_{i}, p_{j}))$, 
until it finds all $k$ points and initiate a total of $k$ clusters.

During the incremental step, the algorithm finds the appropriate cluster for point $p_{j}$ while minimizing the 
expected entropy of the whole system. More specifically, the incremental step is:

![Screenshot_4](https://user-images.githubusercontent.com/57667464/111457041-fe935c80-8717-11eb-9c23-595ac319485d.png)


Since the order of processing points may have an impact on the clustering quality, authors propose one heuristic
to solve this problem: reprocessing the worst fitted datapoints. After a batch of datapoints is clustered, for each
datapoint we can calculate the probability of clusters having values datapoint's values, i.e.
we calculate $p_{i} = \prod_{j} (p_{ij})$ for each datapoint. Afterwards, we find $m$ datapoints for which 
the calculated probability is the lowest and reprocess those datapoints again.


## Roadmap<a name="roadmap"/>



## References<a name="references"/>
I have used multiple references while implementing these algorithms. The most important and comprehensive ones are:
*  [The Elements of Statistical Learning]() by T. Hastie, R.Tibshirani, and J. Friedman
*  [Mathematics for Machine Learning]() by M. P. Deisenroth, A. A. Faisal, and C. S. Ong
*  [NumPy documentation]()
