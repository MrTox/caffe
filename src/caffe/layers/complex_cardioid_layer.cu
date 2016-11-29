#include <vector>

#include "caffe/layers/complex_cardioid_layer.hpp"

namespace caffe {

__global__ void ComplexCardioidForward(const int n, const cuComplex* bottom, cuComplex* top) {
  CUDA_KERNEL_LOOP(index, n) {
    cuComplex z = bottom[index];

    float theta = atan2f(z.y, z.x);
    top[index] = cuCmulf( make_cuFloatComplex(0.5f*(1+cosf(theta)),0), z );
  }
}

__global__ void ComplexCardioidForward(const int n, const cuDoubleComplex* bottom, cuDoubleComplex* top) {
  CUDA_KERNEL_LOOP(index, n) {
    cuDoubleComplex z = bottom[index];

    double theta = atan2(z.y, z.x);
    top[index] = cuCmul( make_cuDoubleComplex(0.5*(1+cos(theta)),0), z );
  }
}

__global__ void ComplexCardioidBackward(const int n, const cuComplex* bottom,
    const cuComplex* top_diff, cuComplex* bottom_diff) {
  CUDA_KERNEL_LOOP(index, n) {
    cuComplex z = bottom[index];

    float theta = atan2f(z.y, z.x);

    cuComplex dfdz = make_cuFloatComplex(0.5f + 0.5f*cosf(theta), 0.25f*sinf(theta));

    cuComplex dfdcz = cuCdivf(
        cuCmulf(
            make_cuFloatComplex(0, -0.25f*sinf(theta)),
            z
            ),
        cuCaddf(cuConjf(z), make_cuFloatComplex(1e-14,0))
        );

    bottom_diff[index] = cuCaddf(
        cuCmulf(cuConjf(top_diff[index]), dfdcz),
        cuCmulf(top_diff[index], cuConjf(dfdz))
        );
  }
}

__global__ void ComplexCardioidBackward(const int n, const cuDoubleComplex* bottom,
    const cuDoubleComplex* top_diff, cuDoubleComplex* bottom_diff) {
  CUDA_KERNEL_LOOP(index, n) {
    cuDoubleComplex z = bottom[index];

    double theta = atan2(z.y, z.x);

    cuDoubleComplex dfdz = make_cuDoubleComplex(0.5 + 0.5*cos(theta), 0.25*sin(theta));

    cuDoubleComplex dfdcz = cuCdiv(
        cuCmul(
            make_cuDoubleComplex(0, -0.25*sin(theta)),
            z
            ),
        cuCadd(cuConj(z), make_cuDoubleComplex(1e-14,0))
        );

    bottom_diff[index] = cuCadd(
        cuCmul(cuConj(top_diff[index]), dfdcz),
        cuCmul(top_diff[index], cuConj(dfdz))
        );
  }
}

template <>
void ComplexCardioidLayer<float>::Forward_gpu(const vector<Blob<float>*>& bottom,
    const vector<Blob<float>*>& top) {
  const std::complex<float>* bottom_data = this->RealToComplexBottomData_gpu(bottom, 0);
  std::complex<float>* top_data = this->RealToComplexTopData_mutable_gpu(top,0);

  int count = top[0]->count()/2;
  ComplexCardioidForward<<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(count,
		  (const cuComplex*)bottom_data, (cuComplex*)top_data);
  CUDA_POST_KERNEL_CHECK;

  this->SyncComplexTopData_gpu(top, 0);
}

template <>
void ComplexCardioidLayer<double>::Forward_gpu(const vector<Blob<double>*>& bottom,
    const vector<Blob<double>*>& top) {
  const std::complex<double>* bottom_data = this->RealToComplexBottomData_gpu(bottom, 0);
  std::complex<double>* top_data = this->RealToComplexTopData_mutable_gpu(top,0);

  int count = top[0]->count()/2;
  ComplexCardioidForward<<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(count,
		  (const cuDoubleComplex*)bottom_data, (cuDoubleComplex*)top_data);
  CUDA_POST_KERNEL_CHECK;

  this->SyncComplexTopData_gpu(top, 0);
}

template <>
void ComplexCardioidLayer<float>::Backward_gpu(const vector<Blob<float>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<float>*>& bottom) {
  if (propagate_down[0]) {
    const std::complex<float>* top_diff = this->RealToComplexTopDiff_gpu(top,0);
    const std::complex<float>* bottom_data = this->RealToComplexBottomData_gpu(bottom, 0);
    std::complex<float>* bottom_diff = this->RealToComplexBottomDiff_mutable_gpu(bottom, 0);

    const int count = top[0]->count()/2;
    ComplexCardioidBackward<<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(count,
    		(const cuComplex*)bottom_data, (const cuComplex*)top_diff, (cuComplex*)bottom_diff);
    CUDA_POST_KERNEL_CHECK;

    this->SyncComplexBlobDiff_gpu(0);
  }
}

template <>
void ComplexCardioidLayer<double>::Backward_gpu(const vector<Blob<double>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<double>*>& bottom) {
  if (propagate_down[0]) {
    const std::complex<double>* top_diff = this->RealToComplexTopDiff_gpu(top,0);
    const std::complex<double>* bottom_data = this->RealToComplexBottomData_gpu(bottom, 0);
    std::complex<double>* bottom_diff = this->RealToComplexBottomDiff_mutable_gpu(bottom, 0);

    const int count = top[0]->count()/2;
    ComplexCardioidBackward<<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(count,
    		(const cuDoubleComplex*)bottom_data, (const cuDoubleComplex*)top_diff, (cuDoubleComplex*)bottom_diff);
    CUDA_POST_KERNEL_CHECK;

    this->SyncComplexBlobDiff_gpu(0);
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(ComplexCardioidLayer);

}  // namespace caffe
