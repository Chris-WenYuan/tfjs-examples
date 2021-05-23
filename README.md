# tfjs-examples

This is a project that contains several examples for classification and regression using Node.js Tensorflow.

## Introduction

### 1. iris-classification

This is an iris classification example. In `iris.csv`, there are 150 data with three kinds of different spicies. Each data have 4 features, namely `sepal_length`, `sepal_width`, `petal_length`, and `petal_width`. I build a simple DNN model to classify the spicie according to the input features using Node.js Tensorflow.

### 2. hearDisease-classification

This is an heart disease classification example. In `heart.csv`, there are 303 data with two targets (0 and 1). Each data have 13 features. If `target=0`, it means that the patient doesn't have heart disease; If `target=1`, it means that the patient have heart disease. Obviously, this example is a classic binary classification problem. I build a simple DNN model to classify whether a patient have heart disease or not.

## How to run?

First, you should download this repository.

```shell
$ git clone https://github.com/tensorflow/tfjs-examples.git
```

Then choose the example you wanna execute, for exmaple (iris-classification):

```shell
$ cd tfjs-exmaples/iris-classification
$ sudo npm install
$ node index.js
```

