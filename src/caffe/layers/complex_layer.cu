#include <complex>

#include "caffe/layers/complex_layer.hpp"

namespace caffe {

template <typename Dtype>
const std::complex<Dtype>* ComplexLayer<Dtype>::RealToComplex_gpu(const Dtype* real_data) {
  // Living life on the edge: side stepping illegal static_cast from float* to complex<float>*
  return reinterpret_cast<const std::complex<Dtype>* >(real_data);
}

template <typename Dtype>
std::complex<Dtype>* ComplexLayer<Dtype>::RealToComplex_mutable_gpu(Dtype* real_data) {
  // Living life on the edge: side stepping illegal static_cast from float* to complex<float>*
  return reinterpret_cast<std::complex<Dtype>* >(real_data);
}

template <typename Dtype>
const std::complex<Dtype>* ComplexLayer<Dtype>::RealToComplexBottomData_gpu(const vector<Blob<Dtype>*>& bottom, int index) {
  return RealToComplex_gpu(bottom[index]->gpu_data());
}

template <typename Dtype>
const std::complex<Dtype>* ComplexLayer<Dtype>::RealToComplexBottomDiff_gpu(const vector<Blob<Dtype>*>& bottom, int index) {
  return RealToComplex_gpu(bottom[index]->gpu_diff());
}

template <typename Dtype>
std::complex<Dtype>* ComplexLayer<Dtype>::RealToComplexBottomDiff_mutable_gpu(const vector<Blob<Dtype>*>& bottom, int index) {
  return RealToComplex_mutable_gpu(bottom[index]->mutable_gpu_diff());
}

template <typename Dtype>
const std::complex<Dtype>* ComplexLayer<Dtype>::RealToComplexTopData_gpu(const vector<Blob<Dtype>*>& top, int index) {
  return RealToComplex_gpu(top[index]->gpu_data());
}

template <typename Dtype>
std::complex<Dtype>* ComplexLayer<Dtype>::RealToComplexTopData_mutable_gpu(const vector<Blob<Dtype>*>& top, int index) {
  return RealToComplex_mutable_gpu(top[index]->mutable_gpu_data());
}

template <typename Dtype>
const std::complex<Dtype>* ComplexLayer<Dtype>::RealToComplexTopDiff_gpu(const vector<Blob<Dtype>*>& top, int index) {
  return RealToComplex_gpu(top[index]->gpu_diff());
}

template <typename Dtype>
const std::complex<Dtype>* ComplexLayer<Dtype>::RealToComplexBlobData_gpu(int index) {
  return RealToComplex_gpu(this->blobs_[index]->gpu_data());
}

template <typename Dtype>
std::complex<Dtype>* ComplexLayer<Dtype>::RealToComplexBlobData_mutable_gpu(int index) {
  return RealToComplex_mutable_gpu(this->blobs_[index]->mutable_gpu_data());
}

template <typename Dtype>
const std::complex<Dtype>* ComplexLayer<Dtype>::RealToComplexBlobDiff_gpu(int index) {
  return RealToComplex_gpu(this->blobs_[index]->gpu_diff());
}

template <typename Dtype>
std::complex<Dtype>* ComplexLayer<Dtype>::RealToComplexBlobDiff_mutable_gpu(int index) {
  return RealToComplex_mutable_gpu(this->blobs_[index]->mutable_gpu_diff());
}

// We are casting the data in RealToComplex, so there is no action needed in SyncComplex

template <typename Dtype>
void ComplexLayer<Dtype>::SyncComplex_gpu(const std::complex<Dtype>* complex_data, Dtype* real_data) {}

template <typename Dtype>
void ComplexLayer<Dtype>::SyncComplexBottomDiff_gpu(const vector<Blob<Dtype>*>& bottom, int index) {}

template <typename Dtype>
void ComplexLayer<Dtype>::SyncComplexTopData_gpu(const vector<Blob<Dtype>*>& top, int index) {}

template <typename Dtype>
void ComplexLayer<Dtype>::SyncComplexBlobData_gpu(int index) {}

template <typename Dtype>
void ComplexLayer<Dtype>::SyncComplexBlobDiff_gpu(int index) {}

}  // namespace caffe

