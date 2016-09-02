#include <vector>

#include "caffe/layers/complex_igaussian_layer.hpp"

namespace caffe {

__global__ void ComplexIGaussianForward(const int n, const cuComplex* bottom, float* top, const float sigmaSq) {
  CUDA_KERNEL_LOOP(index, n) {
	  float bottomReal = bottom[index].x;
	  float bottomImag = bottom[index].y;
	  float bottomSq = bottomReal*bottomReal + bottomImag*bottomImag;

	  top[index] = 1-expf(-bottomSq/(2*sigmaSq));
  }
}

__global__ void ComplexIGaussianForward(const int n, const cuDoubleComplex* bottom, double* top, const double sigmaSq) {
  CUDA_KERNEL_LOOP(index, n) {
	  double bottomReal = bottom[index].x;
	  double bottomImag = bottom[index].y;
	  double bottomSq = bottomReal*bottomReal + bottomImag*bottomImag;

	  top[index] = 1-exp(-bottomSq/(2*sigmaSq));
  }
}

__global__ void ComplexIGaussianBackward(const int n, const cuComplex* bottom, const float* top, const float* top_diff,
		cuComplex* bottom_diff, const float sigmaSq) {
  CUDA_KERNEL_LOOP(index, n) {
	  bottom_diff[index].x = top_diff[index]*(1-top[index])*bottom[index].x/(2*sigmaSq);
	  bottom_diff[index].y = top_diff[index]*(1-top[index])*bottom[index].y/(2*sigmaSq);
  }
}

__global__ void ComplexIGaussianBackward(const int n, const cuDoubleComplex* bottom, const double* top, const double* top_diff,
		cuDoubleComplex* bottom_diff, const double sigmaSq) {
  CUDA_KERNEL_LOOP(index, n) {
	  bottom_diff[index].x = top_diff[index]*(1-top[index])*bottom[index].x/(2*sigmaSq);
	  bottom_diff[index].y = top_diff[index]*(1-top[index])*bottom[index].y/(2*sigmaSq);
  }
}

template <>
void ComplexIGaussianLayer<float>::Forward_gpu(const vector<Blob<float>*>& bottom,
    const vector<Blob<float>*>& top) {
  const std::complex<float>* bottom_data = this->RealToComplexBottomData_gpu(bottom, 0);
  float* top_data = top[0]->mutable_gpu_data();

  int count = top[0]->count();
  ComplexIGaussianForward<<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(count,
		  (const cuComplex*)bottom_data, top_data, this->sigmaSq);
  CUDA_POST_KERNEL_CHECK;

  this->SyncComplexTopData_gpu(top, 0);
}

template <>
void ComplexIGaussianLayer<double>::Forward_gpu(const vector<Blob<double>*>& bottom,
    const vector<Blob<double>*>& top) {
  const std::complex<double>* bottom_data = this->RealToComplexBottomData_gpu(bottom, 0);
  double* top_data = top[0]->mutable_gpu_data();

  int count = top[0]->count();
  ComplexIGaussianForward<<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(count,
		  (const cuDoubleComplex*)bottom_data, top_data, this->sigmaSq);
  CUDA_POST_KERNEL_CHECK;

  this->SyncComplexTopData_gpu(top, 0);
}

template <>
void ComplexIGaussianLayer<float>::Backward_gpu(const vector<Blob<float>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<float>*>& bottom) {
  if (propagate_down[0]) {
    const float* top_data = top[0]->gpu_data();
    const float* top_diff = top[0]->gpu_diff();
    const std::complex<float>* bottom_data = this->RealToComplexBottomData_gpu(bottom, 0);
    std::complex<float>* bottom_diff = this->RealToComplexBottomDiff_mutable_gpu(bottom, 0);

    const int count = top[0]->count();
    ComplexIGaussianBackward<<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(count,
    		(const cuComplex*)bottom_data, top_data, top_diff, (cuComplex*)bottom_diff,
    		this->sigmaSq);
    CUDA_POST_KERNEL_CHECK;

    this->SyncComplexBlobDiff_gpu(0);
  }
}

template <>
void ComplexIGaussianLayer<double>::Backward_gpu(const vector<Blob<double>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<double>*>& bottom) {
  if (propagate_down[0]) {
    const double* top_data = top[0]->gpu_data();
    const double* top_diff = top[0]->gpu_diff();
    const std::complex<double>* bottom_data = this->RealToComplexBottomData_gpu(bottom, 0);
    std::complex<double>* bottom_diff = this->RealToComplexBottomDiff_mutable_gpu(bottom, 0);

    const int count = top[0]->count();
    ComplexIGaussianBackward<<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(count,
    		(const cuDoubleComplex*)bottom_data, top_data, top_diff, (cuDoubleComplex*)bottom_diff,
    		this->sigmaSq);
    CUDA_POST_KERNEL_CHECK;

    this->SyncComplexBlobDiff_gpu(0);
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(ComplexIGaussianLayer);

}  // namespace caffe
