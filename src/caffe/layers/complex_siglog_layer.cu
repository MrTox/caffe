#include <vector>

#include "caffe/layers/complex_siglog_layer.hpp"

namespace caffe {

__global__ void ComplexSiglogForward(const int n, const cuComplex* bottom, cuComplex* top,
    float d, float s, float r, float c) {
  CUDA_KERNEL_LOOP(index, n) {

    cuComplex z = bottom[index];

    cuComplex sz = cuCmulf(make_cuFloatComplex(s,0), z);
    float abs_sz = cuCabsf(sz);
    float sz_to_d = powf(abs_sz, d);
    float denom = c + 1.0/r * sz_to_d + 1e-14;
    top[index] = cuCmulf(sz, make_cuFloatComplex(1.0/denom, 0) );

  }
}

__global__ void ComplexSiglogForward(const int n, const cuDoubleComplex* bottom, cuDoubleComplex* top,
    double d, double s, double r, double c) {
  CUDA_KERNEL_LOOP(index, n) {

    cuDoubleComplex z = bottom[index];

    cuDoubleComplex sz = cuCmul(make_cuDoubleComplex(s,0), z);
    double abs_sz = cuCabs(sz);
    double sz_to_d = pow(abs_sz, d);
    double denom = c + 1.0/r * sz_to_d + 1e-14;
    top[index] = cuCmul(sz, make_cuDoubleComplex(1.0/denom, 0) );
  }
}

__global__ void ComplexSiglogBackward(const int n, const cuComplex* bottom,
    const cuComplex* top_diff, cuComplex* bottom_diff,
    float d, float s, float r, float c) {
  CUDA_KERNEL_LOOP(index, n) {

    cuComplex z = bottom[index];

    float abs_z = cuCabsf(z);

    cuComplex sz = cuCmulf(make_cuFloatComplex(s,0), z);
    float abs_sz = cuCabsf(sz);

    float z_to_d = powf(abs_z, d);
    float z_to_d_minus_one = powf(abs_z, d-1);
    float sz_to_d = powf(abs_sz, d);
    float s_to_d = powf(abs(s), d);

    // Useful temp variable
    float c_r_sz_d = c + 1.0/r * sz_to_d;

    float dfdz_numer = (c_r_sz_d)*s + s*(d/(2*r) * s_to_d * z_to_d);
    float dfdz_denom = c_r_sz_d * c_r_sz_d + 1e-14;
    cuComplex dfdz = make_cuFloatComplex(dfdz_numer/dfdz_denom, 0);

    float dfdcz_coeff_numer = -s * s_to_d * d * z_to_d_minus_one;
    float dfdcz_coeff_denom = 2 * abs_z * r * c_r_sz_d * c_r_sz_d + 1e-14;
    float dfdcz_coeff = dfdcz_coeff_numer / dfdcz_coeff_denom;
    cuComplex dfdcz = cuCmulf(make_cuFloatComplex(dfdcz_coeff, 0), cuCmulf(z,z));

    bottom_diff[index] = cuCaddf(
        cuCmulf(cuConjf(top_diff[index]), dfdcz),
        cuCmulf(top_diff[index], cuConjf(dfdz))
        );
  }
}

__global__ void ComplexSiglogBackward(const int n, const cuDoubleComplex* bottom,
    const cuDoubleComplex* top_diff, cuDoubleComplex* bottom_diff,
    double d, double s, double r, double c) {
  CUDA_KERNEL_LOOP(index, n) {

    cuDoubleComplex z = bottom[index];

    double abs_z = cuCabs(z);

    cuDoubleComplex sz = cuCmul(make_cuDoubleComplex(s,0), z);
    double abs_sz = cuCabs(sz);

    double z_to_d = pow(abs_z, d);
    double z_to_d_minus_one = pow(abs_z, d-1);
    double sz_to_d = pow(abs_sz, d);
    double s_to_d = pow(abs(s), d);

    // Useful temp variable
    double c_r_sz_d = c + 1.0/r * sz_to_d;

    double dfdz_numer = (c_r_sz_d)*s + s*(d/(2*r) * s_to_d * z_to_d);
    double dfdz_denom = c_r_sz_d * c_r_sz_d + 1e-14;
    cuDoubleComplex dfdz = make_cuDoubleComplex(dfdz_numer/dfdz_denom, 0);

    double dfdcz_coeff_numer = -s * s_to_d * d * z_to_d_minus_one;
    double dfdcz_coeff_denom = 2 * abs_z * r * c_r_sz_d * c_r_sz_d + 1e-14;
    double dfdcz_coeff = dfdcz_coeff_numer / dfdcz_coeff_denom;
    cuDoubleComplex dfdcz = cuCmul(make_cuDoubleComplex(dfdcz_coeff, 0), cuCmul(z,z));

    bottom_diff[index] = cuCadd(
        cuCmul(cuConj(top_diff[index]), dfdcz),
        cuCmul(top_diff[index], cuConj(dfdz))
        );
  }
}

template <>
void ComplexSiglogLayer<float>::Forward_gpu(const vector<Blob<float>*>& bottom,
    const vector<Blob<float>*>& top) {
  const std::complex<float>* bottom_data = this->RealToComplexBottomData_gpu(bottom, 0);
  std::complex<float>* top_data = this->RealToComplexTopData_mutable_gpu(top,0);

  int count = top[0]->count()/2;
  ComplexSiglogForward<<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(count,
		  (const cuComplex*)bottom_data, (cuComplex*)top_data,
      this->d, this->s, this->r, this->c);
  CUDA_POST_KERNEL_CHECK;

  this->SyncComplexTopData_gpu(top, 0);
}

template <>
void ComplexSiglogLayer<double>::Forward_gpu(const vector<Blob<double>*>& bottom,
    const vector<Blob<double>*>& top) {
  const std::complex<double>* bottom_data = this->RealToComplexBottomData_gpu(bottom, 0);
  std::complex<double>* top_data = this->RealToComplexTopData_mutable_gpu(top,0);

  int count = top[0]->count()/2;
  ComplexSiglogForward<<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(count,
		  (const cuDoubleComplex*)bottom_data, (cuDoubleComplex*)top_data,
      this->d, this->s, this->r, this->c);
  CUDA_POST_KERNEL_CHECK;

  this->SyncComplexTopData_gpu(top, 0);
}

template <>
void ComplexSiglogLayer<float>::Backward_gpu(const vector<Blob<float>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<float>*>& bottom) {
  if (propagate_down[0]) {
    const std::complex<float>* top_diff = this->RealToComplexTopDiff_gpu(top,0);
    const std::complex<float>* bottom_data = this->RealToComplexBottomData_gpu(bottom, 0);
    std::complex<float>* bottom_diff = this->RealToComplexBottomDiff_mutable_gpu(bottom, 0);

    const int count = top[0]->count()/2;
    ComplexSiglogBackward<<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(count,
    		(const cuComplex*)bottom_data, (const cuComplex*)top_diff, (cuComplex*)bottom_diff,
        this->d, this->s, this->r, this->c);
    CUDA_POST_KERNEL_CHECK;

    this->SyncComplexBlobDiff_gpu(0);
  }
}

template <>
void ComplexSiglogLayer<double>::Backward_gpu(const vector<Blob<double>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<double>*>& bottom) {
  if (propagate_down[0]) {
    const std::complex<double>* top_diff = this->RealToComplexTopDiff_gpu(top,0);
    const std::complex<double>* bottom_data = this->RealToComplexBottomData_gpu(bottom, 0);
    std::complex<double>* bottom_diff = this->RealToComplexBottomDiff_mutable_gpu(bottom, 0);

    const int count = top[0]->count()/2;
    ComplexSiglogBackward<<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(count,
    		(const cuDoubleComplex*)bottom_data, (const cuDoubleComplex*)top_diff, (cuDoubleComplex*)bottom_diff,
        this->d, this->s, this->r, this->c);
    CUDA_POST_KERNEL_CHECK;

    this->SyncComplexBlobDiff_gpu(0);
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(ComplexSiglogLayer);

}  // namespace caffe
