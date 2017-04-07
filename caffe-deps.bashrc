#!/bin/bash

export CAFFE_DEP_ROOT=/u/vis/software/caffe-deps

# CUDA related
#CUDA_ROOT=/usr/local64/lang/cuda-6.5
#CUDA_ROOT=/usr/local64/lang/cuda-7.0
CUDA_ROOT=/usr/local64/lang/cuda-7.5
if [ -e $CUDA_ROOT ] ; then
    export CUDA_ROOT
    export PATH=$CUDA_ROOT/bin:$PATH
    export LD_LIBRARY_PATH=$CUDA_ROOT/lib64:$LD_LIBRARY_PATH
fi
CUDA_DRIVER_PATH=/usr/lib64/libcuda.so
if [ -e $CUDA_DRIVER_PATH ]; then
    export CUDA_DRIVER_PATH
fi

# boost
BOOST_ROOT=/u/vis/software/boost_1_55_0
if [ -e $BOOST_ROOT ] ; then
    export BOOST_ROOT
    export PATH=$BOOST_ROOT/bin:$PATH
    export LD_LIBRARY_PATH=$BOOST_ROOT/lib:$LD_LIBRARY_PATH
fi

# ANACONDA LIB (for PythonLayer)
export ANACONDA_HOME=/u/vis/software/anaconda2
export PATH=$ANACONDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$ANACONDA_HOME/lib:$LD_LIBRARY_PATH

# OPENCV2.3, HDF5, GLOG, PROTOBUF, LMDB, GFLAGS, BLAS
#export OPENCV23_ROOT=$CAFFE_DEP_ROOT/software/opencv23
export LD_LIBRARY_PATH=$CAFFE_DEP_ROOT/lib:$LD_LIBRARY_PATH
export PATH=$CAFFE_DEP_ROOT/bin:$PATH
export PKG_CONFIG_PATH=$CAFFE_DEP_ROOT/lib/pkgconfig:$PKG_CONFIG_PATH

