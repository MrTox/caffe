#include <vector>

#include "caffe/layers/complex_igaussian_phase_layer.hpp"

namespace caffe {

__global__ void ComplexIGaussianPhaseForward(const int n, const cuComplex* bottom, cuComplex* top, const float sigmaSq) {
  CUDA_KERNEL_LOOP(index, n) {
    cuComplex z = bottom[index];

    float zMag = cuCabsf(z) + 1e-14;
    float zReal = z.x;
    float zImag = z.y;
    float zSq = zReal*zReal + zImag*zImag;

    cuComplex g = make_cuFloatComplex(1 - expf(-zSq/(2*sigmaSq)), 0);
    cuComplex p = cuCdivf(z, make_cuFloatComplex(zMag,0) );
    top[index] = cuCmulf(g,p);
  }
}

__global__ void ComplexIGaussianPhaseForward(const int n, const cuDoubleComplex* bottom, cuDoubleComplex* top, const double sigmaSq) {
  CUDA_KERNEL_LOOP(index, n) {
    cuDoubleComplex z = bottom[index];

    double zMag = cuCabs(z) + 1e-14;
    double zReal = z.x;
    double zImag = z.y;
    double zSq = zReal*zReal + zImag*zImag;

    cuDoubleComplex g = make_cuDoubleComplex(1 - exp(-zSq/(2*sigmaSq)), 0);
    cuDoubleComplex p = cuCdiv(z, make_cuDoubleComplex(zMag,0) );
    top[index] = cuCmul(g,p);
  }
}

__global__ void ComplexIGaussianPhaseBackward(const int n, const cuComplex* bottom, const cuComplex* top_diff,
		cuComplex* bottom_diff, const float sigmaSq) {
  CUDA_KERNEL_LOOP(index, n) {

    cuComplex z = bottom[index];

    float zMag = cuCabsf(z) + 1e-14;
    float zReal = z.x;
    float zImag = z.y;
    float zSq = zReal*zReal + zImag*zImag;

    float g = 1 - expf(-zSq/(2*sigmaSq));
    cuComplex p = cuCdivf(z, make_cuFloatComplex(zMag,0) );

    cuComplex dgdcz = cuCmulf( make_cuFloatComplex((1-g)/(2*sigmaSq),0), z );
    cuComplex dgdz = cuConjf(dgdcz);

    cuComplex dpdz = make_cuFloatComplex(1/(2*zMag), 0);

    cuComplex conj_z = cuCaddf(cuConjf(z), make_cuFloatComplex(1e-14,0));
    cuComplex dpdcz = cuCmulf(make_cuFloatComplex(-0.5,0), cuCdivf(p , conj_z));

    cuComplex dfdz = cuCaddf( cuCmulf(p,dgdz), cuCmulf( make_cuFloatComplex(g,0), dpdz ) );
    cuComplex dfdcz = cuCaddf( cuCmulf(p,dgdcz), cuCmulf( make_cuFloatComplex(g,0), dpdcz ) );
    bottom_diff[index] = cuCaddf( cuCmulf(cuConjf(top_diff[index]),dfdcz), cuCmulf(top_diff[index],cuConjf(dfdz)) );
  }
}

__global__ void ComplexIGaussianPhaseBackward(const int n, const cuDoubleComplex* bottom, const cuDoubleComplex* top_diff,
		cuDoubleComplex* bottom_diff, const double sigmaSq) {
  CUDA_KERNEL_LOOP(index, n) {
    cuDoubleComplex z = bottom[index];

    double zMag = cuCabs(z) + 1e-14;
    double zReal = z.x;
    double zImag = z.y;
    double zSq = zReal*zReal + zImag*zImag;

    double g = 1 - exp(-zSq/(2*sigmaSq));
    cuDoubleComplex p = cuCdiv(z, make_cuDoubleComplex(zMag,0) );

    cuDoubleComplex dgdcz = cuCmul( make_cuDoubleComplex((1-g)/(2*sigmaSq),0), z );
    cuDoubleComplex dgdz = cuConj(dgdcz);

    cuDoubleComplex dpdz = make_cuDoubleComplex(1/(2*zMag), 0);

    cuDoubleComplex conj_z = cuCadd(cuConj(z), make_cuDoubleComplex(1e-14,0));
    cuDoubleComplex dpdcz = cuCmul(make_cuDoubleComplex(-0.5,0), cuCdiv(p , conj_z));

    cuDoubleComplex dfdz = cuCadd( cuCmul(p,dgdz), cuCmul( make_cuDoubleComplex(g,0), dpdz ) );
    cuDoubleComplex dfdcz = cuCadd( cuCmul(p,dgdcz), cuCmul( make_cuDoubleComplex(g,0), dpdcz ) );
    bottom_diff[index] = cuCadd( cuCmul(cuConj(top_diff[index]),dfdcz), cuCmul(top_diff[index],cuConj(dfdz)) );
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
    const std::complex<float>* top_diff = this->RealToComplexTopDiff_gpu(top,0);
    const std::complex<float>* bottom_data = this->RealToComplexBottomData_gpu(bottom, 0);
    std::complex<float>* bottom_diff = this->RealToComplexBottomDiff_mutable_gpu(bottom, 0);

    const int count = top[0]->count()/2;
    ComplexIGaussianPhaseBackward<<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(count,
    		(const cuComplex*)bottom_data, (const cuComplex*)top_diff, (cuComplex*)bottom_diff,
    		this->sigmaSq);
    CUDA_POST_KERNEL_CHECK;

    this->SyncComplexBottomDiff_gpu(bottom, 0);
  }
}

template <>
void ComplexIGaussianPhaseLayer<double>::Backward_gpu(const vector<Blob<double>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<double>*>& bottom) {
  if (propagate_down[0]) {
    const std::complex<double>* top_diff = this->RealToComplexTopDiff_gpu(top,0);
    const std::complex<double>* bottom_data = this->RealToComplexBottomData_gpu(bottom, 0);
    std::complex<double>* bottom_diff = this->RealToComplexBottomDiff_mutable_gpu(bottom, 0);

    const int count = top[0]->count()/2;
    ComplexIGaussianPhaseBackward<<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(count,
    		(const cuDoubleComplex*)bottom_data, (const cuDoubleComplex*)top_diff, (cuDoubleComplex*)bottom_diff,
    		this->sigmaSq);
    CUDA_POST_KERNEL_CHECK;

    this->SyncComplexBottomDiff_gpu(bottom, 0);
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(ComplexIGaussianPhaseLayer);

}  // namespace caffe
