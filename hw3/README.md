## Environment
* Ubuntu 18.04
* Python 3.6.8
* numpy 1.16.0, pandas 0.24.0, matplotlib 3.0.2
* tensorflow-gpu 1.13.1, keras==2.2.4


## Maintainer Info
* Chin-Ho, Lin
* tainvecs@gmail.com
* [GitLab](https://gitlab.com/tainvecs/MachineLearning-2017/)


## Outline
* [Overview](https://github.com/tainvecs/MachineLearning-2017/blob/master/hw3/README.md#overview)
* [Dataset](https://github.com/tainvecs/MachineLearning-2017/blob/master/hw3/README.md#dataset)
* [Model Architecture](https://github.com/tainvecs/MachineLearning-2017/blob/master/hw3/README.md#model-architecture)
* [CNN Model](https://github.com/tainvecs/MachineLearning-2017/blob/master/hw3/README.md#cnn-model)
* [Learning Curve](https://github.com/tainvecs/MachineLearning-2017/blob/master/hw3/README.md#learning-curve)
* [Confusion Matrix](https://github.com/tainvecs/MachineLearning-2017/blob/master/hw3/README.md#confusion-matrix)
* [Saliency Map](https://github.com/tainvecs/MachineLearning-2017/blob/master/hw3/README.md#saliency-map)
* [Convolutional Filter Visualization](https://github.com/tainvecs/MachineLearning-2017/blob/master/hw3/README.md#convolutional-filter-visualization)


## Overview
In homework 3, we are given a dataset of human face images and aim to solve a seven-class sentiment classification problem. By training convolutional neural network model, we build a classifier to predict the class of input images. We can further observe training result with convolutional filter visualization and saliency map.


## Dataset
Human face images consist of 7 different sentiment classes, "**Angry**", "**Disgust**", "**Fear**", "**Happy**", "**Sad**", "**Surprise**" and "**Neutral**".

* **Training Data**
	- /data/train.csv
	- size: 28709 images
	- |  | **Angry** | **Disgust** | **Fear** | **Happy** | **Sad** | **Surprise** | **Neutral** |
		| - | -: | -: | -: | -: | -: | -: | -: |
		| **class** | 0 | 1 |  2 |  3 |  4 |  5 | 6 |
		| **images** | 3995 |  436 | 4097 | 7215 | 4830 | 3171 | 4965 |
* **Test Data**
	- /data/test.csv
	- size: 7178 images

* **Validation set**
	- 10\% images randomly splitted and held out from **Training Data**
	- random seed 1234


## Model Architecture

![](https://github.com/tainvecs/MachineLearning-2017/blob/master/hw3/plot/model/896.png?raw=true)

* **First Part**
	* Layers \"**Convolution**\", \"**Activation Function**\", \"**Batch Normalization**\", \"**Max Pooling**\", \"**Dropout**\" are repeated for 4 times.
    * **Convolution**
    	- input shape
            + channels last
            + [ (48, 48, 1), (24, 24, 1), (12, 12, 1), (6, 6, 1) ]
        - filters
        	+ [ 56, 112, 224, 448 ] or [ 112, 224, 448, 896 ]
        - kernel size: (3, 3)
        - strides: (1, 1)
        - kernel initializer: \"Glorot Uniform\"
    * **Activation Function**
        - \"LeakyReLU\", \"ELU\", \"PReLU\", \"ReLU\"
    * **Batch Normalization**
    * **Max Pooling**
        - pool size: (2, 2)
    * **Dropout**
        - dropout rate: ( 0.2, 0.25, 0.3, 0.3 )
* **Flatten Layer**
	* flatten the output from the first part and connect the second part
* **Second Part**
	* Layers \"**Dense Layer**\", \"**Activation Function**\", \"**Batch Normalization**\" and \"**Dropout**\" are repeated for 2 times.
    * **Dense Layer**
        - fully connected layer
    * **Activation Function**
        - \"LeakyReLU\", \"ELU\", \"PReLU\", \"ReLU\"
    * **Batch Normalization**
    * **Dropout**
        - dropout rate: ( 0.5 )
* **Output Layer**
	* Dense Layer and Softmax activation function


## CNN Model
* **--train** (str)
	- path of training data
	- If **--train** specified, new model will be trained and output to **--out_model**
* **--validate** (float)
	- float 0~1: the proportions of validation set split from training dataset'
* **--random_seed** (int)
	- random seed for splitting training and validation data
* **--epoch** (int)
	- number of training epoch
* **--batch_size** (int)
	- sgd mini batch size
* **--optimizer** (str)
	- option: \"adam\"
* **--filters** (int)
	- number of filters in first convolutional layer
	- the number of filters will increase by 2 times in preceeding convolutional layers
* **--cnn_activation** (str)
	- activation function option: \"leakyrelu\", \"elu\", \"prelu\", \"relu\"
* **--cnn_activation_alpha** (float)
	- alpha for activation function
* **--units** (int)
	- number of units for dnn input and hidden layers
* **--dnn_activation** (str)
	- activation function option: \"leakyrelu\", \"elu\", \"prelu\", \"relu\"
* **--dnn_activation_alpha** (float)
	- alpha for activation function
* **--out_log** (str)
	- path to output log file
* **--out_model** (str)
	- path to output model


## Learning Curve

![](https://github.com/tainvecs/MachineLearning-2017/blob/master/hw3/plot/loss_acc_curves/val_0.1_seed_1234_epoch_2000_batch_64_opt_adam_cnn_filter_56_leakyrelu_0.3_dnn_unit_448_leakyrelu_0.3.loss_acc_curves.png?raw=true)

![](https://github.com/tainvecs/MachineLearning-2017/blob/master/hw3/plot/loss_acc_curves/val_0.1_seed_1234_epoch_2000_batch_64_opt_adam_cnn_filter_56_elu_1.0_dnn_unit_448_elu_1.0.loss_acc_curves.png?raw=true)

![](https://github.com/tainvecs/MachineLearning-2017/blob/master/hw3/plot/loss_acc_curves/val_0.1_seed_1234_epoch_2000_batch_64_opt_adam_cnn_filter_56_prelu_0.0_dnn_unit_448_prelu_0.0.loss_acc_curves.png?raw=true)

![](https://github.com/tainvecs/MachineLearning-2017/blob/master/hw3/plot/loss_acc_curves/val_0.1_seed_1234_epoch_2000_batch_64_opt_adam_cnn_filter_56_relu_0.0_dnn_unit_448_relu_0.0.loss_acc_curves.png?raw=true)

* Parameters
	- **Epoch**
		+ 2000
	- **Batch Size**
		+ 64
	- **Optimizer**
		+ \"adam\"
	- **Filters**
		+ [ 56, 112, 224, 448 ]
	- **Activation Function**
		+ | **\"cnn_activation\"** | **\"cnn_activation_alpha\"** | **\"units\"** | **\"dnn_activation\"** | **\"dnn_activation_alpha\"** |			
			| :-: | :-: | :-: | :-: | :-: |
			| \"leakyrelu\" | 0.3 | 448 |  \"leakyrelu\" |  0.3 |
            | \"elu\" | 1.0 |  448 | \"elu\" | 1.0 |
            | \"prelu\" | - |  448 | \"prelu\" | - |            
            | \"relu\" | - |  448 | \"relu\" | - |

* Applying theses activation functions in the CNN model are able to perform about 0.7 accuracy on validation set.
* From learning curve of LeakyReLU using keras default alpha 0.3, we may found that the training improvement of loss and accuracy dramatically slow down after 250 epochs.

![](https://github.com/tainvecs/MachineLearning-2017/blob/master/hw3/plot/loss_acc_curves/val_0.1_seed_1234_epoch_4000_batch_64_opt_adam_cnn_filter_56_leakyrelu_0.3_dnn_unit_448_leakyrelu_0.3.loss_acc_curves.png?raw=true)

* Parameters
	- **Epoch**
		+ 4000
	- **Batch Size**
		+ 64
	- **Optimizer**
		+ \"adam\"
	- **Filters**
		+ [ 56, 112, 224, 448 ]
	- **Activation Function**
		+ | **\"cnn_activation\"** | **\"cnn_activation_alpha\"** | **\"units\"** | **\"dnn_activation\"** | **\"dnn_activation_alpha\"** |		
			| :-: | :-: | :-: | :-: | :-: |
			| \"leakyrelu\" | 0.3 | 448 |  \"leakyrelu\" |  0.3 |

* We plot learning curves of 4000 epochs. Compare to 2000 epochs, the training performance is still improving but with an extremely slow speed.

![](https://github.com/tainvecs/MachineLearning-2017/blob/master/hw3/plot/loss_acc_curves/val_0.1_seed_1234_epoch_2000_batch_32_opt_adam_cnn_filter_56_leakyrelu_0.3_dnn_unit_448_leakyrelu_0.3.loss_acc_curves.png?raw=true)

* Parameters
	- **Epoch**
		+ 2000
	- **Batch Size**
		+ 32
	- **Optimizer**
		+ \"adam\"
	- **Filters**
		+ [ 56, 112, 224, 448 ]
	- **Activation Function**
		+ | **\"cnn_activation\"** | **\"cnn_activation_alpha\"** | **\"units\"** | **\"dnn_activation\"** | **\"dnn_activation_alpha\"** |			
			| :-: | :-: | :-: | :-: | :-: |
			| \"leakyrelu\" | 0.3 | 448 |  \"leakyrelu\" |  0.3 |

* We tried a batch size 32 model but the model converge even slightly slower.

![](https://github.com/tainvecs/MachineLearning-2017/blob/master/hw3/plot/loss_acc_curves/val_0.1_seed_1234_epoch_2000_batch_32_opt_adam_cnn_filter_56_leakyrelu_0.1_dnn_unit_448_leakyrelu_0.1.loss_acc_curves.png?raw=true)

* Parameters
	- **Epoch**
		+ 2000
	- **Batch Size**
		+ 32
	- **Optimizer**
		+ \"adam\"
	- **Filters**
		+ [ 56, 112, 224, 448 ]
	- **Activation Function**
		+ | **\"cnn_activation\"** | **\"cnn_activation_alpha\"** | **\"units\"** | **\"dnn_activation\"** | **\"dnn_activation_alpha\"** |		
			| :-: | :-: | :-: | :-: | :-: |
			| \"leakyrelu\" | 0.1 | 448 |  \"leakyrelu\" |  0.1 |

* Activation function "LeakyReLU" is a special case of "PReLU" with constant alpha. As the models with "ReLU" and "PReLU" activation function are able to converge much faster, model with "LeakyReLU" activation function should able to converge in same speed.
* We tried a smaller alpha model and training speed indeed speed up.

![](https://github.com/tainvecs/MachineLearning-2017/blob/master/hw3/plot/loss_acc_curves/val_0.1_seed_1234_epoch_2000_batch_64_opt_adam_cnn_filter_56_elu_1.0_dnn_unit_448_elu_1.0.loss_acc_curves.png?raw=true)

![](https://github.com/tainvecs/MachineLearning-2017/blob/master/hw3/plot/loss_acc_curves/val_0.1_seed_1234_epoch_2000_batch_64_opt_adam_cnn_filter_56_elu_0.7_dnn_unit_448_elu_0.7.loss_acc_curves.png?raw=true)

* Parameters
	- **Epoch**
		+ 2000
	- **Batch Size**
		+ 64
	- **Optimizer**
		+ \"adam\"
	- **Filters**
		+ [ 56, 112, 224, 448 ]
	- **Activation Function**
		+ tested with different activation function
		+ | **\"cnn_activation\"** | **\"cnn_activation_alpha\"** | **\"units\"** | **\"dnn_activation\"** | **\"dnn_activation_alpha\"** |			
			| :-: | :-: | :-: | :-: | :-: |
			| \"elu\" | 1.0 |  448 | \"elu\" | 1.0 |
			| \"elu\" | 0.7 | 448 |  \"elu\" |  0.7 |

* Besides, we tried a smaller alpha with \"ELU\" activation function, and the training performance and time to converge shows no significant difference.

![](https://github.com/tainvecs/MachineLearning-2017/blob/master/hw3/plot/loss_acc_curves/val_0.1_seed_1234_epoch_2000_batch_64_opt_adam_cnn_filter_112_elu_1.0_dnn_unit_896_elu_1.0.loss_acc_curves.png?raw=true)

* Furthermore, we tried a model with more filters, and the training loss converge to 0 even much faster.
* Parameters
	- **Epoch**
		+ 2000
	- **Batch Size**
		+ 64
	- **Optimizer**
		+ \"adam\"
	- **Filters**
		+ [ 112, 224, 448, 896 ]
    - **Activation Function**
    	+ | **\"cnn_activation\"** | **\"cnn_activation_alpha\"** | **\"units\"** | **\"dnn_activation\"** | **\"dnn_activation_alpha\"** |		
			| :-: | :-: | :-: | :-: | :-: |
  			\"elu\" | 1.0 |  896 | \"elu\" | 1.0 |
* The model attain the best performance, 0.7052 of validation accuracy, at the 722th epoch.
* We further ensemble 11 model that trained with different parameters and achieve 0.7181 of validation accuracy.


## Confusion Matrix

![](https://github.com/tainvecs/MachineLearning-2017/blob/master/hw3/plot/confusion_matrix/val_0.1_seed_1234_epoch_2000_batch_64_opt_adam_cnn_filter_112_elu_1.0_dnn_unit_896_elu_1.0_0722-0.7052.confusion_matrix.png?raw=true)

* From the confusion matrix, the model has the best performance on class \"happy\".
* Images of classes \"Disgust\" and \"Surprise\" have more facial expressions and seem to be easier cases for the model.
* The model tend to predict \"Neural\" on negative emotions such as \"Angry\", \"Fear\" and \"Sad\".
* The model has poor performance on classes \"Fear\" and \"Sad\".

| **True:Fear, Predict:Angry** | **True:Fear, Predict:Sad** | **True:Angry, Predict:Neutral** | **True:Sad, Predict:Neutral** |
| :-: | :-: | :-: | :-: |
| ![](https://github.com/tainvecs/MachineLearning-2017/blob/master/hw3/example/386.png?raw=true) | ![](https://github.com/tainvecs/MachineLearning-2017/blob/master/hw3/example/8.png?raw=true) | ![](https://github.com/tainvecs/MachineLearning-2017/blob/master/hw3/example/135.png?raw=true) | ![](https://github.com/tainvecs/MachineLearning-2017/blob/master/hw3/example/167.png?raw=true)|
| ![](https://github.com/tainvecs/MachineLearning-2017/blob/master/hw3/example/311.png?raw=true) | ![](https://github.com/tainvecs/MachineLearning-2017/blob/master/hw3/example/215.png?raw=true) | ![](https://github.com/tainvecs/MachineLearning-2017/blob/master/hw3/example/742.png?raw=true) | ![](https://github.com/tainvecs/MachineLearning-2017/blob/master/hw3/example/154.png?raw=true) |

* Some images might be mislabeled or ambiguous.
* Class \"Fear\" seem to be mispredicted as classes \"Angry\" and \"Sad\".
* The table also lists some ambiguous examples of classes \"Neural\", \"Angry\" and \"Sad\".


## Saliency Map

* From the saliency map, cnn model mainly focus on the eyes, mouth, cheek and forehead of the face. The model seems to capture sentiment information from the line or wrinkle of those parts.

|  |  |  |  |
| :-: | :-: | :-: | :-: |
| **Class** | **Original Image** | **Heat Map** | **Focus** |
| Angry | ![](https://github.com/tainvecs/MachineLearning-2017/blob/master/hw3/plot/saliency_map/val_0.1_seed_1234_epoch_2000_batch_64_opt_adam_cnn_filter_112_elu_1.0_dnn_unit_896_elu_1.0_0722-0.7052.ori_img.image_25704.png?raw=true) | ![](https://github.com/tainvecs/MachineLearning-2017/blob/master/hw3/plot/saliency_map/val_0.1_seed_1234_epoch_2000_batch_64_opt_adam_cnn_filter_112_elu_1.0_dnn_unit_896_elu_1.0_0722-0.7052.heatmp.image_25704.png?raw=true) | ![](https://github.com/tainvecs/MachineLearning-2017/blob/master/hw3/plot/saliency_map/val_0.1_seed_1234_epoch_2000_batch_64_opt_adam_cnn_filter_112_elu_1.0_dnn_unit_896_elu_1.0_0722-0.7052.machine_focus.image_25704.png?raw=true) |
| Disgust | ![](https://github.com/tainvecs/MachineLearning-2017/blob/master/hw3/plot/saliency_map/val_0.1_seed_1234_epoch_2000_batch_64_opt_adam_cnn_filter_112_elu_1.0_dnn_unit_896_elu_1.0_0722-0.7052.ori_img.image_25717.png?raw=true) | ![](https://github.com/tainvecs/MachineLearning-2017/blob/master/hw3/plot/saliency_map/val_0.1_seed_1234_epoch_2000_batch_64_opt_adam_cnn_filter_112_elu_1.0_dnn_unit_896_elu_1.0_0722-0.7052.heatmp.image_25717.png?raw=true) | ![](https://github.com/tainvecs/MachineLearning-2017/blob/master/hw3/plot/saliency_map/val_0.1_seed_1234_epoch_2000_batch_64_opt_adam_cnn_filter_112_elu_1.0_dnn_unit_896_elu_1.0_0722-0.7052.machine_focus.image_25717.png?raw=true) |
| Fear | ![](https://github.com/tainvecs/MachineLearning-2017/blob/master/hw3/plot/saliency_map/val_0.1_seed_1234_epoch_2000_batch_64_opt_adam_cnn_filter_112_elu_1.0_dnn_unit_896_elu_1.0_0722-0.7052.ori_img.image_25707.png?raw=true) | ![](https://github.com/tainvecs/MachineLearning-2017/blob/master/hw3/plot/saliency_map/val_0.1_seed_1234_epoch_2000_batch_64_opt_adam_cnn_filter_112_elu_1.0_dnn_unit_896_elu_1.0_0722-0.7052.heatmp.image_25707.png?raw=true) | ![](https://github.com/tainvecs/MachineLearning-2017/blob/master/hw3/plot/saliency_map/val_0.1_seed_1234_epoch_2000_batch_64_opt_adam_cnn_filter_112_elu_1.0_dnn_unit_896_elu_1.0_0722-0.7052.machine_focus.image_25707.png?raw=true) |
| Happy | ![](https://github.com/tainvecs/MachineLearning-2017/blob/master/hw3/plot/saliency_map/val_0.1_seed_1234_epoch_2000_batch_64_opt_adam_cnn_filter_112_elu_1.0_dnn_unit_896_elu_1.0_0722-0.7052.ori_img.image_25705.png?raw=true) | ![](https://github.com/tainvecs/MachineLearning-2017/blob/master/hw3/plot/saliency_map/val_0.1_seed_1234_epoch_2000_batch_64_opt_adam_cnn_filter_112_elu_1.0_dnn_unit_896_elu_1.0_0722-0.7052.heatmp.image_25705.png?raw=true) | ![](https://github.com/tainvecs/MachineLearning-2017/blob/master/hw3/plot/saliency_map/val_0.1_seed_1234_epoch_2000_batch_64_opt_adam_cnn_filter_112_elu_1.0_dnn_unit_896_elu_1.0_0722-0.7052.machine_focus.image_25705.png?raw=true) |
| Sad | ![](https://github.com/tainvecs/MachineLearning-2017/blob/master/hw3/plot/saliency_map/val_0.1_seed_1234_epoch_2000_batch_64_opt_adam_cnn_filter_112_elu_1.0_dnn_unit_896_elu_1.0_0722-0.7052.ori_img.image_25700.png?raw=true) | ![](https://github.com/tainvecs/MachineLearning-2017/blob/master/hw3/plot/saliency_map/val_0.1_seed_1234_epoch_2000_batch_64_opt_adam_cnn_filter_112_elu_1.0_dnn_unit_896_elu_1.0_0722-0.7052.heatmp.image_25700.png?raw=true) | ![](https://github.com/tainvecs/MachineLearning-2017/blob/master/hw3/plot/saliency_map/val_0.1_seed_1234_epoch_2000_batch_64_opt_adam_cnn_filter_112_elu_1.0_dnn_unit_896_elu_1.0_0722-0.7052.machine_focus.image_25700.png?raw=true) |
| Surprise | ![](https://github.com/tainvecs/MachineLearning-2017/blob/master/hw3/plot/saliency_map/val_0.1_seed_1234_epoch_2000_batch_64_opt_adam_cnn_filter_112_elu_1.0_dnn_unit_896_elu_1.0_0722-0.7052.ori_img.image_25699.png?raw=true) | ![](https://github.com/tainvecs/MachineLearning-2017/blob/master/hw3/plot/saliency_map/val_0.1_seed_1234_epoch_2000_batch_64_opt_adam_cnn_filter_112_elu_1.0_dnn_unit_896_elu_1.0_0722-0.7052.heatmp.image_25699.png?raw=true) | ![](https://github.com/tainvecs/MachineLearning-2017/blob/master/hw3/plot/saliency_map/val_0.1_seed_1234_epoch_2000_batch_64_opt_adam_cnn_filter_112_elu_1.0_dnn_unit_896_elu_1.0_0722-0.7052.machine_focus.image_25699.png?raw=true) |
| Neutral | ![](https://github.com/tainvecs/MachineLearning-2017/blob/master/hw3/plot/saliency_map/val_0.1_seed_1234_epoch_2000_batch_64_opt_adam_cnn_filter_112_elu_1.0_dnn_unit_896_elu_1.0_0722-0.7052.ori_img.image_25718.png?raw=true) | ![](https://github.com/tainvecs/MachineLearning-2017/blob/master/hw3/plot/saliency_map/val_0.1_seed_1234_epoch_2000_batch_64_opt_adam_cnn_filter_112_elu_1.0_dnn_unit_896_elu_1.0_0722-0.7052.heatmp.image_25718.png?raw=true) | ![](https://github.com/tainvecs/MachineLearning-2017/blob/master/hw3/plot/saliency_map/val_0.1_seed_1234_epoch_2000_batch_64_opt_adam_cnn_filter_112_elu_1.0_dnn_unit_896_elu_1.0_0722-0.7052.machine_focus.image_25718.png?raw=true) |


## Convolutional Filter Visualization

* In the first and second layers, the filters seem to capture the information of training images through some sort of line and stripe.
* As the layers go deeper, the filters are more activated by complex texture.

| | |
| :-: | :-: |
| **Filters of First Conv2D Layer** | **Output of First Conv2D Layer** |
| ![](https://github.com/tainvecs/MachineLearning-2017/blob/master/hw3/plot/activate_filter/val_0.1_seed_1234_epoch_2000_batch_64_opt_adam_cnn_filter_112_elu_1.0_dnn_unit_896_elu_1.0_0722-0.7052.activate_filter.layer_conv2d_1.png?raw=true) | ![](https://github.com/tainvecs/MachineLearning-2017/blob/master/hw3/plot/activate_filter/val_0.1_seed_1234_epoch_2000_batch_64_opt_adam_cnn_filter_112_elu_1.0_dnn_unit_896_elu_1.0_0722-0.7052.filtered_image.layer_conv2d_1.png?raw=true) |
| **Filters of First Conv2D Layer + Batch Norm + Activation** | **Output of First Conv2D Layer + Batch Norm + Activation** |
| ![](https://github.com/tainvecs/MachineLearning-2017/blob/master/hw3/plot/activate_filter/val_0.1_seed_1234_epoch_2000_batch_64_opt_adam_cnn_filter_112_elu_1.0_dnn_unit_896_elu_1.0_0722-0.7052.activate_filter.layer_elu_1.png?raw=true) | ![](https://github.com/tainvecs/MachineLearning-2017/blob/master/hw3/plot/activate_filter/val_0.1_seed_1234_epoch_2000_batch_64_opt_adam_cnn_filter_112_elu_1.0_dnn_unit_896_elu_1.0_0722-0.7052.filtered_image.layer_elu_1.png?raw=true) |
| **Filters of Second Conv2D Layer** | **Output of Second Conv2D Layer** |
| ![](https://github.com/tainvecs/MachineLearning-2017/blob/master/hw3/plot/activate_filter/val_0.1_seed_1234_epoch_2000_batch_64_opt_adam_cnn_filter_112_elu_1.0_dnn_unit_896_elu_1.0_0722-0.7052.activate_filter.layer_conv2d_2.png?raw=true) | ![](https://github.com/tainvecs/MachineLearning-2017/blob/master/hw3/plot/activate_filter/val_0.1_seed_1234_epoch_2000_batch_64_opt_adam_cnn_filter_112_elu_1.0_dnn_unit_896_elu_1.0_0722-0.7052.filtered_image.layer_conv2d_2.png?raw=true) |
| **Filters of Second Conv2D Layer + Batch Norm + Activation** | **Output of Second Conv2D Layer + Batch Norm + Activation** |
| ![](https://github.com/tainvecs/MachineLearning-2017/blob/master/hw3/plot/activate_filter/val_0.1_seed_1234_epoch_2000_batch_64_opt_adam_cnn_filter_112_elu_1.0_dnn_unit_896_elu_1.0_0722-0.7052.activate_filter.layer_elu_2.png?raw=true) | ![](https://github.com/tainvecs/MachineLearning-2017/blob/master/hw3/plot/activate_filter/val_0.1_seed_1234_epoch_2000_batch_64_opt_adam_cnn_filter_112_elu_1.0_dnn_unit_896_elu_1.0_0722-0.7052.filtered_image.layer_elu_2.png?raw=true) |
| **Filters of Third Conv2D Layer** | **Output of Third Conv2D Layer** |
| ![](https://github.com/tainvecs/MachineLearning-2017/blob/master/hw3/plot/activate_filter/val_0.1_seed_1234_epoch_2000_batch_64_opt_adam_cnn_filter_112_elu_1.0_dnn_unit_896_elu_1.0_0722-0.7052.activate_filter.layer_conv2d_3.png?raw=true) | ![](https://github.com/tainvecs/MachineLearning-2017/blob/master/hw3/plot/activate_filter/val_0.1_seed_1234_epoch_2000_batch_64_opt_adam_cnn_filter_112_elu_1.0_dnn_unit_896_elu_1.0_0722-0.7052.filtered_image.layer_conv2d_3.png?raw=true) |
| **Filters of Third Conv2D Layer + Batch Norm + Activation** | **Output of Third Conv2D Layer + Batch Norm + Activation** |
| ![](https://github.com/tainvecs/MachineLearning-2017/blob/master/hw3/plot/activate_filter/val_0.1_seed_1234_epoch_2000_batch_64_opt_adam_cnn_filter_112_elu_1.0_dnn_unit_896_elu_1.0_0722-0.7052.activate_filter.layer_elu_3.png?raw=true) | ![](https://github.com/tainvecs/MachineLearning-2017/blob/master/hw3/plot/activate_filter/val_0.1_seed_1234_epoch_2000_batch_64_opt_adam_cnn_filter_112_elu_1.0_dnn_unit_896_elu_1.0_0722-0.7052.filtered_image.layer_elu_3.png?raw=true) |
| **Filters of Fourth Conv2D Layer** | **Output of Fourth Conv2D Layer** |
| ![](https://github.com/tainvecs/MachineLearning-2017/blob/master/hw3/plot/activate_filter/val_0.1_seed_1234_epoch_2000_batch_64_opt_adam_cnn_filter_112_elu_1.0_dnn_unit_896_elu_1.0_0722-0.7052.activate_filter.layer_conv2d_4.png?raw=true) | ![](https://github.com/tainvecs/MachineLearning-2017/blob/master/hw3/plot/activate_filter/val_0.1_seed_1234_epoch_2000_batch_64_opt_adam_cnn_filter_112_elu_1.0_dnn_unit_896_elu_1.0_0722-0.7052.filtered_image.layer_conv2d_4.png?raw=true) |
| **Filters of Fourth Conv2D Layer + Batch Norm + Activation** | **Output of Fourth Conv2D Layer + Batch Norm + Activation** |
| ![](https://github.com/tainvecs/MachineLearning-2017/blob/master/hw3/plot/activate_filter/val_0.1_seed_1234_epoch_2000_batch_64_opt_adam_cnn_filter_112_elu_1.0_dnn_unit_896_elu_1.0_0722-0.7052.activate_filter.layer_elu_4.png?raw=true) | ![](https://github.com/tainvecs/MachineLearning-2017/blob/master/hw3/plot/activate_filter/val_0.1_seed_1234_epoch_2000_batch_64_opt_adam_cnn_filter_112_elu_1.0_dnn_unit_896_elu_1.0_0722-0.7052.filtered_image.layer_elu_4.png?raw=true) |


## Reference
* [CNN Slide](http://speech.ee.ntu.edu.tw/~tlkagk/courses/ML_2017/Lecture/CNN.pdf)
* [Tips for Deep Learning](http://speech.ee.ntu.edu.tw/~tlkagk/courses/ML_2017/Lecture/DNN%20tip.pdf)
* [Understanding Activation Functions in Neural Networks](https://medium.com/the-theory-of-everything/understanding-activation-functions-in-neural-networks-9491262884e0)
* [Activation Functions](https://ml-cheatsheet.readthedocs.io/en/latest/activation_functions.html#id7)
* [ReLU](https://www.tinymind.com/learn/terms/relu)
* [How convolutional neural networks see the world](https://blog.keras.io/how-convolutional-neural-networks-see-the-world.html)
* [Saliency Map](https://github.com/raghakot/keras-vis)
* [Code Reference](https://github.com/orbxball/ML2017/tree/master/hw3)
