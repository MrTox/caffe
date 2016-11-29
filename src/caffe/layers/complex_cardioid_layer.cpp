#include <vector>

#include "caffe/layers/complex_cardioid_layer.hpp"

namespace caffe {

template <typename Dtype>
void ComplexCardioidLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  CHECK_EQ(2, bottom[0]->shape(-1));

  vector<int> shape = bottom[0]->shape();
  top[0]->Reshape(shape);
}

template <typename Dtype>
void ComplexCardioidLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const std::complex<Dtype>* bottom_data = this->RealToComplexBottomData_cpu(bottom, 0);
  std::complex<Dtype>* top_data = this->RealToComplexTopData_mutable_cpu(top,0);

  const int count = top[0]->count()/2;

  for (int i = 0; i < count; ++i) {
    std::complex<Dtype> z = bottom_data[i];
    top_data[i] = 1/2 * (1 + std::cos(std::arg(z))) * z;
  }

  this->SyncComplexTopData_cpu(top, 0);
}

template <typename Dtype>
void ComplexCardioidLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[0]) {
    const std::complex<Dtype>* top_diff = this->RealToComplexTopDiff_cpu(top,0);
    const std::complex<Dtype>* bottom_data = this->RealToComplexBottomData_cpu(bottom, 0);
    std::complex<Dtype>* bottom_diff = this->RealToComplexBottomDiff_mutable_cpu(bottom, 0);

    const int count = top[0]->count()/2;

    for (int i = 0; i < count; ++i) {
      std::complex<Dtype> z = bottom_data[i];

      Dtype theta = std::arg(z);

      std::complex<Dtype> dfdz = Dtype(0.5) + Dtype(0.5*std::cos(theta)) +
          std::complex<Dtype>(0,1)/Dtype(4) * std::sin(theta);

      std::complex<Dtype> dfdcz = std::complex<Dtype>(0,-1)/Dtype(4) * std::sin(std::arg(z)) * 
          z / (std::conj(z) + Dtype(1e-14));

      bottom_diff[i] = std::conj(top_diff[i])*dfdcz + top_diff[i]*std::conj(dfdz);
    }

    this->SyncComplexBlobDiff_cpu(0);
  }
}


#ifdef CPU_ONLY
STUB_GPU(ComplexCardioidLayer);
#endif

INSTANTIATE_CLASS(ComplexCardioidLayer);
REGISTER_LAYER_CLASS(ComplexCardioid);

}  // namespace caffe
