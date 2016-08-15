#include <complex>

#include "caffe/layers/complex_layer.hpp"

namespace caffe {

template <typename Dtype>
const std::complex<Dtype>* ComplexLayer<Dtype>::RealToComplex_cpu(const Dtype* real_data) {
  // Living life on the edge: side stepping illegal static_cast from float* to complex<float>*
  return reinterpret_cast<const std::complex<Dtype>* >(real_data);
}

template <typename Dtype>
std::complex<Dtype>* ComplexLayer<Dtype>::RealToComplex_mutable_cpu(Dtype* real_data) {
  // Living life on the edge: side stepping illegal static_cast from float* to complex<float>*
  return reinterpret_cast<std::complex<Dtype>* >(real_data);
}

template <typename Dtype>
const std::complex<Dtype>* ComplexLayer<Dtype>::RealToComplexBottomData_cpu(const vector<Blob<Dtype>*>& bottom, int index) {
  return RealToComplex_cpu(bottom[index]->cpu_data());
}

template <typename Dtype>
const std::complex<Dtype>* ComplexLayer<Dtype>::RealToComplexBottomDiff_cpu(const vector<Blob<Dtype>*>& bottom, int index) {
  return RealToComplex_cpu(bottom[index]->cpu_diff());
}

template <typename Dtype>
std::complex<Dtype>* ComplexLayer<Dtype>::RealToComplexBottomDiff_mutable_cpu(const vector<Blob<Dtype>*>& bottom, int index) {
  return RealToComplex_mutable_cpu(bottom[index]->mutable_cpu_diff());
}

template <typename Dtype>
const std::complex<Dtype>* ComplexLayer<Dtype>::RealToComplexTopData_cpu(const vector<Blob<Dtype>*>& top, int index) {
  return RealToComplex_cpu(top[index]->cpu_data());
}

template <typename Dtype>
std::complex<Dtype>* ComplexLayer<Dtype>::RealToComplexTopData_mutable_cpu(const vector<Blob<Dtype>*>& top, int index) {
  return RealToComplex_mutable_cpu(top[index]->mutable_cpu_data());
}

template <typename Dtype>
const std::complex<Dtype>* ComplexLayer<Dtype>::RealToComplexTopDiff_cpu(const vector<Blob<Dtype>*>& top, int index) {
  return RealToComplex_cpu(top[index]->cpu_diff());
}

template <typename Dtype>
const std::complex<Dtype>* ComplexLayer<Dtype>::RealToComplexBlobData_cpu(int index) {
  return RealToComplex_cpu(this->blobs_[index]->cpu_data());
}

template <typename Dtype>
std::complex<Dtype>* ComplexLayer<Dtype>::RealToComplexBlobData_mutable_cpu(int index) {
  return RealToComplex_mutable_cpu(this->blobs_[index]->mutable_cpu_data());
}

template <typename Dtype>
const std::complex<Dtype>* ComplexLayer<Dtype>::RealToComplexBlobDiff_cpu(int index) {
  return RealToComplex_cpu(this->blobs_[index]->cpu_diff());
}

template <typename Dtype>
std::complex<Dtype>* ComplexLayer<Dtype>::RealToComplexBlobDiff_mutable_cpu(int index) {
  return RealToComplex_mutable_cpu(this->blobs_[index]->mutable_cpu_diff());
}

// We are casting the data in RealToComplex, so there is no action needed in SyncComplex

template <typename Dtype>
void ComplexLayer<Dtype>::SyncComplex_cpu(const std::complex<Dtype>* complex_data, Dtype* real_data) {}

template <typename Dtype>
void ComplexLayer<Dtype>::SyncComplexBottomDiff_cpu(const vector<Blob<Dtype>*>& bottom, int index) {}

template <typename Dtype>
void ComplexLayer<Dtype>::SyncComplexTopData_cpu(const vector<Blob<Dtype>*>& top, int index) {}

template <typename Dtype>
void ComplexLayer<Dtype>::SyncComplexBlobData_cpu(int index) {}

template <typename Dtype>
void ComplexLayer<Dtype>::SyncComplexBlobDiff_cpu(int index) {}



INSTANTIATE_CLASS(ComplexLayer);

}  // namespace caffe

