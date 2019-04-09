## Environment
* Ubuntu 18.04
* Python 3.6.8
* numpy 1.16.0, pandas 0.24.0, matplotlib 3.0.2


## Maintainer Info
* Chin-Ho, Lin
* tainvecs@gmail.com


## Overview

In homework 2, we aim to solve a binary classification problem by implementing a logistic regression model and a probabilistic generative model. We are given training features including age, workplace, and education etc., and tried to predict whether the personal income of people in test data exceeds $50K/yr.


## Dataset
Predict whether income exceeds $50K/yr based on census data. Also known as "Census Income" dataset. Extraction was done by Barry Becker from the 1994 Census database. ([more info](https://archive.ics.uci.edu/ml/datasets/Adult))

* **Training Data**
	- Raw Data
		+ /raw_data/train.csv
		+ size: (32561, 15)
	- Feature
		+ /feature/X_train
		+ size: (32561, 106)
	- Ground Truth
		+ /feature/Y_train

* **Test Data**
	- Raw Data
		+ /raw_data/test.csv
		+ size: (16281, 14)
	- Feature
		+ /feature/X_test
		+ size: (16281, 106)


## Logistic Regression Model


* **--train_feature** (str)
	- path of training feature
	- If both **--train_feature** and **--train_answer** are specified, new model will be trained and output.
* **--train_answer** (str)
	- path of training ground truth
	- If both **--train_feature** and **--train_answer** are specified, new model will be trained and output.
* **--test_feature** (str)
	- path of test feature
	- If specified, prediction will be output to **--out_predict**.
* **--validate** (float or int)
	- float 0~1: the proportions of validation set split from training dataset
	- int > 1: number of validation data slice from training dataset
	- validation data should not be more than 30\% of training dataset
* **--in_model** (str)
	- path of the model to load
* **--random_seed** (int)
	- random seed for splitting training and validation data
* **--epoch** (int)
	- number of training epoch
* **--batch_size** (int)
	- sgd mini batch size
* **--eta** (float)
	- learning rate
* **--l2_lambda** (float)
	- l2 norm lambda value
* **--optimizer** (str)
	- option: **\"adam\"**, **\"ada\"**
		* adam optimizer
		* ada gradient
* **--beta_m** (float)
	- bata value of momentum
	- The value should be specified if the optimizer is \"adam\".
* **--beta_v** (float)
	- bata value of velocity
	- The value should be specified if the optimizer is \"adam\".
* **--epsilon** (float)
	- The small value for avoiding divide by zero error while calculating gradient.
* **--norm** (str)
	- feature normalization method
	- options: **\"none\"**, **\"standard\"**, **\"min_max\"**, **\"mean\"**
* **--early_stop** (bool)
	- early stopping based on validation value
* **--out_log** (str)
	- path to output log file
* **--out_model** (str)
	- path to output model
* **--out_predict** (str)
	- path to output test prediction file
* **--debug** (bool)
	- print debug message


## Probabilistic Generative Model


* **--train_x** (str)
	- path of training feature
* **--train_y** (str)
	- path of training ground truth
* **--test_x** (str)
	- path of test feature
* **--validate** (float or int)
	- float 0~1: the proportions of validation set split from training dataset
	- int > 1: number of validation data slice from training dataset
	- validation data should not be more than 30\% of training dataset
* **--random_seed** (int)
	- random seed for splitting training and validation data
* **--norm** (str)
	- feature normalization method
	- options: **\"none\"**, **\"standard\"**, **\"min_max\"**, **\"mean\"**
* **--early_stop** (bool)
	- early stopping based on validation value
* **--out_predict** (str)
	- path to output test prediction file
* **--debug** (bool)
	- print debug message


## Model

* a 107-dimensional numpy array
* the first 106 dimension is the training feature weight **w**
* the last dimension is the bias term **b**


## Parameters

* **Logistic Regression Model**
	- Optimizer
		+ \"adam\", \"ada\"
	- Feature Normalization
		+ \"standard\", \"min_max\", \"mean\", \"none\"
	- Mini Batch Size
		+ 10, 50, 100, 500
	- L2 Regularization Lambda
		+ 0.1, 0.01, 0.001, 0.0001
	- Learning Rate (eta)
		+ 10 1 0.1 0.01 0.001 0.0001 0.00001
	- In general, model trained with **standard score** normalized features and **adam** optimizer gain better performance.
	- The best parameter set is adam optimizer with mini batch size 10, L2 regularization lambda 0.0001 and learning rate 0.01. The following table shows part of the validation result with different scaling of L2 regularization lambda and  learning rate, and compares the performance on different feature normalization method.


| Optimizer | Feature Normalization | Mini Batch Size | L2 Regularization Lambda | Learning Rate (eta) |   Accuracy (Validation) |
| :---: | :------: | -----: | ------: |---------: | ----------: |
|  adam | standard |     10 |   0.0001 | 0.01   |  **0.854115** |
|  adam |  min_max |     10 |   0.0001 | 0.01   |  0.851658 |
|  adam |     mean |     10 |   0.0001 | 0.01   |  0.851966 |
|  adam |     none |     10 |   0.0001 | 0.01   |  0.844902 |
|  adam | standard |     10 |   0.0001 | 0.0001 |  0.851658 |
|  adam |  min_max |     10 |   0.0001 | 0.0001 |  0.850123 |
|  adam |     mean |     10 |   0.0001 | 0.0001 |  0.849509 |
|  adam |     none |     10 |   0.0001 | 0.0001 |  0.847052 |
|  adam | standard |     10 |   0.0001 | 1 |  0.837531 |
|  adam |  min_max |     10 |   0.0001 | 1 |  0.833231 |
|  adam |     mean |     10 |   0.0001 | 1 |  0.836916 |
|  adam |     none |     10 |   0.0001 | 1 |  0.809275 |
|  adam | standard |     10 |    0.01  | 0.01 |  0.851351 |
|  adam |  min_max |     10 |    0.01  | 0.01 |  0.839681 |
|  adam |     mean |     10 |    0.01  | 0.01 |  0.840295 |
|  adam |     none |     10 |    0.01  | 0.01 |  0.808968 |


* As the tables shown, training with standard score normalized features gets consistently better validation accuracy than training with none normalized features.
* In addition, feature normalization shows significant improvement if learning rate eta or L2 regularization lambda is relatively large.
* Besides, we also found that training with none normalized features is more vulnerable to overflow error when computing exponential value in sigmoid function.

![](https://github.com/tainvecs/MachineLearning-2017/blob/master/hw2/png/eta-opt_adam_l2_lambda_0.0001_batch_size_10_norm_standard.png?raw=true)
* Learning rate eta tested with 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1 and 10.
* Optimizer: adam, Feature Normalization: standard, Mini Batch Size: 10, L2 Regularization Lambda: 0.0001

![](https://github.com/tainvecs/MachineLearning-2017/blob/master/hw2/png/l2_lambda-opt_adam_eta_0.01_batch_size_10_norm_standard.png?raw=true)
* L2 regularization lambda tested with 1e-4, 1e-3, 1e-2 and 1e-1.
* Optimizer: adam, Feature Normalization: standard, Mini Batch Size: 10, Learning Rate Eta: 0.01


* **Probabilistic Generative model**
* On the contrary, feature normalization show no difference on validation accuracy.


## Reference

* [Adult Data Set](https://archive.ics.uci.edu/ml/datasets/Adult)
* [Logistic Regression Model Slide](http://speech.ee.ntu.edu.tw/~tlkagk/courses/ML_2016/Lecture/Logistic%20Regression%20(v3).pdf)
* [Gradient Descent Slide](http://speech.ee.ntu.edu.tw/~tlkagk/courses/ML_2016/Lecture/Gradient%20Descent%20(v2).pdf)
* [Probabilistic Generative Model Slide](http://speech.ee.ntu.edu.tw/~tlkagk/courses/ML_2016/Lecture/Classification%20(v3).pdf)
* [Feature Normalization](https://en.wikipedia.org/wiki/Feature_scaling)
* [Machine Learning: Regularization](https://murphymind.blogspot.com/2017/05/machine.learning.regularization.html)
* [Machine Learning note: SGD, Momentum, AdaGrad, Adam Optimizer](https://medium.com/%E9%9B%9E%E9%9B%9E%E8%88%87%E5%85%94%E5%85%94%E7%9A%84%E5%B7%A5%E7%A8%8B%E4%B8%96%E7%95%8C/%E6%A9%9F%E5%99%A8%E5%AD%B8%E7%BF%92ml-note-sgd-momentum-adagrad-adam-optimizer-f20568c968db)
* [Beyond SGD: Gradient Descent with Momentum and Adaptive Learning Rate](https://wiseodd.github.io/techblog/2016/06/22/nn-optimization/)
* [Early Stopping Implementation](https://gist.github.com/ryanpeach/9ef833745215499e77a2a92e71f89ce2)
