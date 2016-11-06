#include <vector>

#include "caffe/layers/complex_igaussian_phase_layer.hpp"

namespace caffe {

__global__ void ComplexIGaussianPhaseForward(const int n, const cuComplex* bottom, cuComplex* top, const float sigmaSq) {
  CUDA_KERNEL_LOOP(index, n) {
    cuComplex z = bottom[index];
	  float zReal = z.x;
	  float zImag = z.y;
	  float zSq = zReal*zReal + zImag*zImag;

	  cuComplex g = make_cuFloatComplex(1 - expf(-zSq/(2*sigmaSq)), 0);
	  cuComplex p = cuCdivf(z, make_cuFloatComplex(cuCabsf(z),0) );
	  top[index] = cuCmulf(g,p);
  }
}

__global__ void ComplexIGaussianPhaseForward(const int n, const cuDoubleComplex* bottom, cuDoubleComplex* top, const double sigmaSq) {
  CUDA_KERNEL_LOOP(index, n) {
    cuDoubleComplex z = bottom[index];
    double zReal = z.x;
    double zImag = z.y;
    double zSq = zReal*zReal + zImag*zImag;

    cuDoubleComplex g = make_cuDoubleComplex(1 - exp(-zSq/(2*sigmaSq)), 0);
    cuDoubleComplex p = cuCdiv(z, make_cuDoubleComplex(cuCabs(z),0) );
    top[index] = cuCmul(g,p);
  }
}

__global__ void ComplexIGaussianPhaseBackward(const int n, const cuComplex* bottom, const float* top, const float* top_diff,
		cuComplex* bottom_diff, const float sigmaSq) {
  CUDA_KERNEL_LOOP(index, n) {
	  bottom_diff[index].x = top_diff[index]*(1-top[index])*bottom[index].x/(2*sigmaSq);
	  bottom_diff[index].y = top_diff[index]*(1-top[index])*bottom[index].y/(2*sigmaSq);
  }
}

__global__ void ComplexIGaussianPhaseBackward(const int n, const cuDoubleComplex* bottom, const double* top, const double* top_diff,
		cuDoubleComplex* bottom_diff, const double sigmaSq) {
  CUDA_KERNEL_LOOP(index, n) {
	  bottom_diff[index].x = top_diff[index]*(1-top[index])*bottom[index].x/(2*sigmaSq);
	  bottom_diff[index].y = top_diff[index]*(1-top[index])*bottom[index].y/(2*sigmaSq);
  }
}

template <>
void ComplexIGaussianPhaseLayer<float>::Forward_gpu(const vector<Blob<float>*>& bottom,
    const vector<Blob<float>*>& top) {
  const std::complex<float>* bottom_data = this->RealToComplexBottomData_gpu(bottom, 0);
  std::complex<float>* top_data = this->RealToComplexTopData_mutable_gpu(top,0);

  int count = top[0]->count()/2;
  ComplexIGaussianPhaseForward<<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(count,
		  (const cuComplex*)bottom_data, (cuComplex*)top_data, this->sigmaSq);
  CUDA_POST_KERNEL_CHECK;

  this->SyncComplexTopData_gpu(top, 0);
}

template <>
void ComplexIGaussianPhaseLayer<double>::Forward_gpu(const vector<Blob<double>*>& bottom,
    const vector<Blob<double>*>& top) {
  const std::complex<double>* bottom_data = this->RealToComplexBottomData_gpu(bottom, 0);
  std::complex<double>* top_data = this->RealToComplexTopData_mutable_gpu(top,0);

  int count = top[0]->count()/2;
  ComplexIGaussianPhaseForward<<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(count,
		  (const cuDoubleComplex*)bottom_data, (cuDoubleComplex*)top_data, this->sigmaSq);
  CUDA_POST_KERNEL_CHECK;

  this->SyncComplexTopData_gpu(top, 0);
}

template <>
void ComplexIGaussianPhaseLayer<float>::Backward_gpu(const vector<Blob<float>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<float>*>& bottom) {
  if (propagate_down[0]) {
    const float* top_data = top[0]->gpu_data();
    const float* top_diff = top[0]->gpu_diff();
    const std::complex<float>* bottom_data = this->RealToComplexBottomData_gpu(bottom, 0);
    std::complex<float>* bottom_diff = this->RealToComplexBottomDiff_mutable_gpu(bottom, 0);

    const int count = top[0]->count()/2;
    ComplexIGaussianPhaseBackward<<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(count,
    		(const cuComplex*)bottom_data, top_data, top_diff, (cuComplex*)bottom_diff,
    		this->sigmaSq);
    CUDA_POST_KERNEL_CHECK;

    this->SyncComplexBlobDiff_gpu(0);
  }
}

template <>
void ComplexIGaussianPhaseLayer<double>::Backward_gpu(const vector<Blob<double>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<double>*>& bottom) {
  if (propagate_down[0]) {
    const double* top_data = top[0]->gpu_data();
    const double* top_diff = top[0]->gpu_diff();
    const std::complex<double>* bottom_data = this->RealToComplexBottomData_gpu(bottom, 0);
    std::complex<double>* bottom_diff = this->RealToComplexBottomDiff_mutable_gpu(bottom, 0);

    const int count = top[0]->count()/2;
    ComplexIGaussianPhaseBackward<<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(count,
    		(const cuDoubleComplex*)bottom_data, top_data, top_diff, (cuDoubleComplex*)bottom_diff,
    		this->sigmaSq);
    CUDA_POST_KERNEL_CHECK;

    this->SyncComplexBlobDiff_gpu(0);
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(ComplexIGaussianPhaseLayer);

}  // namespace caffe
