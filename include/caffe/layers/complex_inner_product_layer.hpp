#ifndef CAFFE_COMPLEX_INNER_PRODUCT_LAYER_HPP_
#define CAFFE_COMPLEX_INNER_PRODUCT_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layers/complex_layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

/**
 * @brief Complex-valued version of the inner product/fully-connectet layer, computes an inner product
 *        with a set of learned weights, and (optionally) adds biases.
 *
 * TODO(dox): thorough documentation for Forward, Backward, and proto params.
 */
template <typename Dtype>
class ComplexInnerProductLayer : public ComplexLayer<Dtype> {
 public:
  explicit ComplexInnerProductLayer(const LayerParameter& param)
      : ComplexLayer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "ComplexInnerProduct"; }
  virtual inline int ExactNumBottomBlobs() const { return 1; }
  virtual inline int ExactNumTopBlobs() const { return 1; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
//  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
//      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
// TODO Complex
//  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
//      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  int M_;
  int K_;
  int N_;
  bool bias_term_;
  Blob<Dtype> bias_multiplier_;
  Blob<Dtype> conj_weight_;
  Blob<Dtype> conj_bottom_;
  bool transpose_;  ///< if true, assume transposed weights
};

}  // namespace caffe

#endif  // CAFFE_COMPLEX_INNER_PRODUCT_LAYER_HPP_
