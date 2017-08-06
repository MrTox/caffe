#!/usr/bin/env sh
# This script converts the mnist data in to complex-valued data,
# with all zeros in the imaginary channel, and saves the data
# in HDF5 files with key values 'data' and 'labels'.
set -e

EXAMPLE=examples/complex
DATA=data/mnist

echo "Creating HDF5 dataset..."

rm -rf $EXAMPLE/mnist_train_data.h5
rm -rf $EXAMPLE/mnist_test_data.h5

python $EXAMPLE/convert_mnist_complex_data.py $DATA/train-images-idx3-ubyte \
  $DATA/train-labels-idx1-ubyte $EXAMPLE/mnist_train_data
python $EXAMPLE/convert_mnist_complex_data.py $DATA/t10k-images-idx3-ubyte \
  $DATA/t10k-labels-idx1-ubyte $EXAMPLE/mnist_test_data

echo "Done."
