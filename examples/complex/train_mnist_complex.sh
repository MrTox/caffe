#!/usr/bin/env sh
set -e

./build/tools/caffe train --solver=examples/complex/mnist_complex_solver.prototxt $@
