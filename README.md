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

### COOLCAT Clustering Algorithm<br/>


The clustering analysis used here is COOLCAT clustering algorithm proposed by 
[Barbara, Couto, & Li (2002)](https://dl.acm.org/doi/abs/10.1145/584792.584888). 
COOLCAT algorithm is used for clustering categorical datasets and is based on a notion of *entropy*. 
More specifically, the entropy of one categorical variable is:

<div align="center"><img src="https://render.githubusercontent.com/render/math?math=%5CLARGE%0A%5Cbegin%7Baligned%7D%0AE(X)%20%3D%20-%5Csum_%7Bx%20%5Cin%20S%7D%20p(x)log(p(x))%0A%5Cend%7Baligned%7D%0A"></div>

Authors assume variable independence, which means that multivariate entropy is equal to
the sum of each variable's entropy. The minimization criterion of the algorithm is the *expected 
entropy of the whole system*:

<div align="center"><img src="https://render.githubusercontent.com/render/math?math=%5CLARGE%0A%5Cbegin%7Baligned%7D%0AE(C)%20%3D%20%5Csum_%7Bk%7D%20(%5Cfrac%7B%7CC_%7Bk%7D%7C%7D%7B%7CD%7C%7D(E(C_%7Bk%7D)))%0A%5Cend%7Baligned%7D%0A"></div>

where <img src="https://render.githubusercontent.com/render/math?math=%7CC_%7Bk%7D%7C"> is the size of cluster <img src="https://render.githubusercontent.com/render/math?math=k"> and <img src="https://render.githubusercontent.com/render/math?math=%7CD%7C"> is the size of dataset. 
In other words, algorithm finds clusters which minimize expected entropy of all clusters.
In order to find such clusters, algorithm goes through two steps: *initialization step* and *incremental step*. 
The initialization step finds two datapoints <img src="https://render.githubusercontent.com/render/math?math=p_%7B1%7D"> and 
<img src="https://render.githubusercontent.com/render/math?math=p_%7B2%7D"> such that their multivariate entropy 
<img src="https://render.githubusercontent.com/render/math?math=E(p_%7B1%7D%2C%20p_%7B2%7D)"> is the highest possible. 
These two datapoints will be assigned to clusters <img src="https://render.githubusercontent.com/render/math?math=C_%7B1%7D"> and 
<img src="https://render.githubusercontent.com/render/math?math=C_%7B2%7D">.
Afterwards, algorithm finds new datapoint <img src="https://render.githubusercontent.com/render/math?math=p_%7Bj%7D"> 
which maximizes <img src="https://render.githubusercontent.com/render/math?math=min_%7Bi%3D1%2C...%2Cj-1%7D(E(p_%7Bi%7D%2C%20p_%7Bj%7D))">, 
until it finds all <img src="https://render.githubusercontent.com/render/math?math=k"> points and initiate a total of <img src="https://render.githubusercontent.com/render/math?math=k"> clusters.

During the incremental step, the algorithm finds the appropriate cluster for point $p_{j}$ while minimizing the 
expected entropy of the whole system. More specifically, the incremental step is:

* given an initial set of clusters <img src="https://render.githubusercontent.com/render/math?math=C_%7B1%7D%24%20to%20%24C_%7Bk%7D">:
  * for each datapoint <img src="https://render.githubusercontent.com/render/math?math=p"> do:
    * for <img src="https://render.githubusercontent.com/render/math?math=i%20%3D%201%2C...%2Ck">:
      * temporarily place <img src="https://render.githubusercontent.com/render/math?math=p"> in <img src="https://render.githubusercontent.com/render/math?math=C%5E%7Bi%7D"> and calculate the expected entropy of the system <img src="https://render.githubusercontent.com/render/math?math=E(C%5E%7Bi%7D)">
      * let <img src="https://render.githubusercontent.com/render/math?math=j%20%3D%20argmin_%7Bi%7D(E(C%5E%7Bi%7D))">
    * place <img src="https://render.githubusercontent.com/render/math?math=p"> in <img src="https://render.githubusercontent.com/render/math?math=C_%7Bj%7D">
  * until all points have been clustered

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
