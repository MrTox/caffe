---
title: Complex MNIST Tutorial
description: Train and test a complex-valued network on the MNIST handwritten digit data.
category: example
include_in_docs: true
priority: 5
---

# Training complex-valued network on MNIST with Caffe

We will assume that you have Caffe successfully compiled. If not, please refer to the [Installation page](/installation.html). In this tutorial, we will assume that your Caffe installation is located at `CAFFE_ROOT`.

## Prepare Datasets

You will first need to download and convert the data format from the MNIST website. To do this, simply run the following commands:

    cd $CAFFE_ROOT
    ./data/mnist/get_mnist.sh
    ./examples/complex/create_mnist_complex.sh

If it complains that `wget` or `gunzip` are not installed, you need to install them respectively. After running the script there should be two datasets, `mnist_train_data.h5`, and `mnist_test_data.h5`.

## Training and Testing the Model

Training the model is simple after you have written the network definition protobuf and solver protobuf files. Simply run `train_mnist_complex.sh`, or the following command directly:

    cd $CAFFE_ROOT
    ./examples/mnist/train_mnist_complex.sh

`train_mnist_complex.sh` is a simple script, but here is a quick explanation: the main tool for training is `caffe` with action `train` and the solver protobuf text file as its argument.

