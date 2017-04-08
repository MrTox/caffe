#include <vector>
#include <limits>

#include "caffe/layers/complex_igaussian_phase_layer.hpp"

namespace caffe {

template <typename Dtype>
void ComplexIGaussianPhaseLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  CHECK_EQ(2, bottom[0]->shape(-1));

  vector<int> shape = bottom[0]->shape();
  top[0]->Reshape(shape);
}

template <typename Dtype>
void ComplexIGaussianPhaseLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const std::complex<Dtype>* bottom_data = this->RealToComplexBottomData_cpu(bottom, 0);
  std::complex<Dtype>* top_data = this->RealToComplexTopData_mutable_cpu(top,0);

  Dtype eps = std::numeric_limits<float>::min();

  const int count = top[0]->count()/2;

  for (int i = 0; i < count; ++i) {
    std::complex<Dtype> z = bottom_data[i];

    Dtype g = 1 - std::real(std::exp(-std::conj(z)*z/(2*this->sigmaSq)));

    Dtype z_mag = std::abs(z);
    std::complex<Dtype> p = z / (z_mag + eps);

    top_data[i] = g*p;
  }

  this->SyncComplexTopData_cpu(top, 0);
}

template <typename Dtype>
void ComplexIGaussianPhaseLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[0]) {
    const std::complex<Dtype>* top_diff = this->RealToComplexTopDiff_cpu(top,0);
    const std::complex<Dtype>* bottom_data = this->RealToComplexBottomData_cpu(bottom, 0);
    std::complex<Dtype>* bottom_diff = this->RealToComplexBottomDiff_mutable_cpu(bottom, 0);

    Dtype eps = std::numeric_limits<float>::min();

    const int count = top[0]->count()/2;

    for (int i = 0; i < count; ++i) {
      std::complex<Dtype> z = bottom_data[i];
      Dtype g = 1 - std::real(std::exp(-std::conj(z)*z/(2*this->sigmaSq)));

      Dtype abs_z = std::abs(z);
      std::complex<Dtype> p = z/(abs_z + eps);

      std::complex<Dtype> dgdcz = (1-g)*z/(2*this->sigmaSq);
      std::complex<Dtype> dgdz = std::conj(dgdcz);
      std::complex<Dtype> dpdz = 1 / (2*abs_z + eps);
      std::complex<Dtype> dpdcz = Dtype(-0.5) * z / (std::conj(z)*abs_z + eps);
      std::complex<Dtype> dfdz = p*dgdz + g*dpdz;
      std::complex<Dtype> dfdcz = p*dgdcz + g*dpdcz;
      bottom_diff[i] = std::conj(top_diff[i])*dfdcz + top_diff[i]*std::conj(dfdz);
    }

    this->SyncComplexBlobDiff_cpu(0);
  }
}


#ifdef CPU_ONLY
STUB_GPU(ComplexIGaussianPhaseLayer);
#endif

INSTANTIATE_CLASS(ComplexIGaussianPhaseLayer);
REGISTER_LAYER_CLASS(ComplexIGaussianPhase);

}  // namespace caffe
