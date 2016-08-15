#ifndef CAFFE_COMPLEX_LAYER_HPP_
#define CAFFE_COMPLEX_LAYER_HPP_

#include <complex>

#include "caffe/layer.hpp"

namespace caffe {

/**
 * @brief Base class for layers dealing with complex-values
 *
 * TODO(dox)
 */
template <typename Dtype>
class ComplexLayer : public Layer<Dtype> {
 public:
  explicit ComplexLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  static const std::complex<Dtype>* RealToComplex_cpu(const Dtype* real_data);
  static std::complex<Dtype>* RealToComplex_mutable_cpu(Dtype* real_data);

 protected:
  const std::complex<Dtype>* RealToComplexBottomData_cpu(const vector<Blob<Dtype>*>& bottom, int index);
  const std::complex<Dtype>* RealToComplexBottomDiff_cpu(const vector<Blob<Dtype>*>& bottom, int index);
  std::complex<Dtype>* RealToComplexBottomDiff_mutable_cpu(const vector<Blob<Dtype>*>& bottom, int index);
  const std::complex<Dtype>* RealToComplexTopData_cpu(const vector<Blob<Dtype>*>& top, int index);
  std::complex<Dtype>* RealToComplexTopData_mutable_cpu(const vector<Blob<Dtype>*>& top, int index);
  const std::complex<Dtype>* RealToComplexTopDiff_cpu(const vector<Blob<Dtype>*>& top, int index);
  const std::complex<Dtype>* RealToComplexBlobData_cpu(int index);
  std::complex<Dtype>* RealToComplexBlobData_mutable_cpu(int index);
  const std::complex<Dtype>* RealToComplexBlobDiff_cpu(int index);
  std::complex<Dtype>* RealToComplexBlobDiff_mutable_cpu(int index);

  static void SyncComplex_cpu(const std::complex<Dtype>* complex_data, Dtype* real_data);
  void SyncComplexBottomDiff_cpu(const vector<Blob<Dtype>*>& bottom, int index);
  void SyncComplexTopData_cpu(const vector<Blob<Dtype>*>& top, int index);
  void SyncComplexBlobData_cpu(int index);
  void SyncComplexBlobDiff_cpu(int index);

};

} // namespace caffe


#endif /* INCLUDE_CAFFE_LAYERS_COMPLEX_LAYER_Hpp_ */
