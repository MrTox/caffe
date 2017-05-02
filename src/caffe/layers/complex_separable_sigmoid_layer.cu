#include <vector>

#include "caffe/layers/complex_separable_sigmoid_layer.hpp"

namespace caffe {

__device__ void sigmoid(const float x, float& result) {
   result = 1.0 / (1 + exp(-x) + 1e-14);
}

__device__ void sigmoid(const double x, double& result) {
   result = 1.0 / (1 + exp(-x) + 1e-14);
}

__global__ void ComplexSeparableSigmoidForward(const int n, const cuComplex* bottom, cuComplex* top) {
  CUDA_KERNEL_LOOP(index, n) {

    cuComplex z = bottom[index];

    float g_real = 0;
    float g_imag = 0;
    sigmoid(cuCrealf(z), g_real);
    sigmoid(cuCimagf(z), g_imag);

    top[index] = make_cuFloatComplex(g_real, g_imag);
  }
}

__global__ void ComplexSeparableSigmoidForward(const int n, const cuDoubleComplex* bottom, cuDoubleComplex* top) {
  CUDA_KERNEL_LOOP(index, n) {

    cuDoubleComplex z = bottom[index];

    double g_real = 0;
    double g_imag = 0;
    sigmoid(cuCreal(z), g_real);
    sigmoid(cuCimag(z), g_imag);

    top[index] = make_cuDoubleComplex(g_real, g_imag);
  }
}

__global__ void ComplexSeparableSigmoidBackward(const int n, const cuComplex* top,
    const cuComplex* top_diff, cuComplex* bottom_diff) {
  CUDA_KERNEL_LOOP(index, n) {
  
    // We already computed g_real = sigmoid(Re(bottom))
    // and g_imag = sigmoid(Imag(bottom)) during forward pass
    cuComplex g = top[index];
    float g_real = cuCrealf(g);
    float g_imag = cuCimagf(g);

    float dg_real = g_real * (1-g_real);
    float dg_imag = g_imag * (1-g_imag);

    cuComplex dfdz = make_cuFloatComplex(0.5*(dg_real + dg_imag), 0);
    cuComplex dfdcz = make_cuFloatComplex(0.5*(dg_real - dg_imag), 0);
    
    bottom_diff[index] = cuCaddf(
        cuCmulf(cuConjf(top_diff[index]), dfdcz),
        cuCmulf(top_diff[index], cuConjf(dfdz))
        );
  }
}

__global__ void ComplexSeparableSigmoidBackward(const int n, const cuDoubleComplex* top,
    const cuDoubleComplex* top_diff, cuDoubleComplex* bottom_diff) {
  CUDA_KERNEL_LOOP(index, n) {

    // We already computed g_real = sigmoid(Re(bottom))
    // and g_imag = sigmoid(Imag(bottom)) during forward pass
    cuDoubleComplex g = top[index];
    double g_real = cuCreal(g);
    double g_imag = cuCimag(g);

    double dg_real = g_real * (1-g_real);
    double dg_imag = g_imag * (1-g_imag);

    cuDoubleComplex dfdz = make_cuDoubleComplex(0.5*(dg_real + dg_imag), 0);
    cuDoubleComplex dfdcz = make_cuDoubleComplex(0.5*(dg_real - dg_imag), 0);
 
    bottom_diff[index] = cuCadd(
        cuCmul(cuConj(top_diff[index]), dfdcz),
        cuCmul(top_diff[index], cuConj(dfdz))
        );
  }
}

template <>
void ComplexSeparableSigmoidLayer<float>::Forward_gpu(const vector<Blob<float>*>& bottom,
    const vector<Blob<float>*>& top) {
  const std::complex<float>* bottom_data = this->RealToComplexBottomData_gpu(bottom, 0);
  std::complex<float>* top_data = this->RealToComplexTopData_mutable_gpu(top,0);

  int count = top[0]->count()/2;
  ComplexSeparableSigmoidForward<<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(count,
		  (const cuComplex*)bottom_data, (cuComplex*)top_data);
  CUDA_POST_KERNEL_CHECK;

  this->SyncComplexTopData_gpu(top, 0);
}

template <>
void ComplexSeparableSigmoidLayer<double>::Forward_gpu(const vector<Blob<double>*>& bottom,
    const vector<Blob<double>*>& top) {
  const std::complex<double>* bottom_data = this->RealToComplexBottomData_gpu(bottom, 0);
  std::complex<double>* top_data = this->RealToComplexTopData_mutable_gpu(top,0);

  int count = top[0]->count()/2;
  ComplexSeparableSigmoidForward<<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(count,
		  (const cuDoubleComplex*)bottom_data, (cuDoubleComplex*)top_data);
  CUDA_POST_KERNEL_CHECK;

  this->SyncComplexTopData_gpu(top, 0);
}

template <>
void ComplexSeparableSigmoidLayer<float>::Backward_gpu(const vector<Blob<float>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<float>*>& bottom) {
  if (propagate_down[0]) {
    const std::complex<float>* top_diff = this->RealToComplexTopDiff_gpu(top,0);
    const std::complex<float>* top_data = this->RealToComplexTopData_gpu(top, 0);
    std::complex<float>* bottom_diff = this->RealToComplexBottomDiff_mutable_gpu(bottom, 0);

    const int count = top[0]->count()/2;
    ComplexSeparableSigmoidBackward<<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(count,
    		(const cuComplex*)top_data, (const cuComplex*)top_diff, (cuComplex*)bottom_diff);
    CUDA_POST_KERNEL_CHECK;

    this->SyncComplexBlobDiff_gpu(0);
  }
}

template <>
void ComplexSeparableSigmoidLayer<double>::Backward_gpu(const vector<Blob<double>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<double>*>& bottom) {
  if (propagate_down[0]) {
    const std::complex<double>* top_diff = this->RealToComplexTopDiff_gpu(top,0);
    const std::complex<double>* top_data = this->RealToComplexTopData_gpu(top, 0);
    std::complex<double>* bottom_diff = this->RealToComplexBottomDiff_mutable_gpu(bottom, 0);

    const int count = top[0]->count()/2;
    ComplexSeparableSigmoidBackward<<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(count,
    		(const cuDoubleComplex*)top_data, (const cuDoubleComplex*)top_diff, (cuDoubleComplex*)bottom_diff);
    CUDA_POST_KERNEL_CHECK;

    this->SyncComplexBlobDiff_gpu(0);
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(ComplexSeparableSigmoidLayer);

}  // namespace caffe
