#ifndef CAFFE_COMPLEX_MAGNITUDE_LAYER_HPP_
#define CAFFE_COMPLEX_MAGNITUDE_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/complex_layer.hpp"

namespace caffe {

/**
 * @brief Magnitude of complex-valued input @f$ y = (\conj{x}x)^{-1/2} @f$.
 */
template <typename Dtype>
class ComplexMagnitudeLayer : public ComplexLayer<Dtype> {
 public:
  explicit ComplexMagnitudeLayer(const LayerParameter& param)
      : ComplexLayer<Dtype>(param) {}

  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "ComplexMagnitude"; }

 protected:
  /**
   * @param bottom input Blob vector (length 1)
   *   -# @f$ (N \times C \times H \times W \times 2) @f$
   *      the inputs @f$ x @f$, where @f
   * @param top output Blob vector (length 1)
   *   -# @f$ (N \times C \times H \times W) @f$ x @f$ is complex-valued,
   *   hence the added dimension of size 2 on the input blob (stored as [real, imaginary].
   *      the computed outputs @f$
   *        y = (\conj{x}x)^{-1/2}
   *      @f$, where @f$ y @f$ is real-valued.
   */
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  // TODO Complex
//  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
//      const vector<Blob<Dtype>*>& top);

  /**
   * @brief Computes the error gradient w.r.t. the magnitude inputs.
   *
   * @param top output Blob vector (length 1), providing the error gradient with
   *      respect to the outputs
   *   -# @f$ (N \times C \times H \times W) @f$
   *      containing error gradients @f$ \frac{\partial E}{\partial y} @f$
   *      with respect to computed outputs @f$ y @f$. Note: these are real-valued
   *      because this layer outputs only real values.
   * @param propagate_down see Layer::Backward.
   * @param bottom input Blob vector (length 1)
   *   -# @f$ (N \times C \times H \times W \times 2) @f$
   *      the inputs @f$ x @f$; Backward fills their diff with
   *      gradients @f$
   *        \frac{\partial E}{\partial x} = \frac{\partial E}{\partial y}\frac{x}{(\conj{x}x)^{-1/2}}
   *      @f$ if propagate_down[0], by default.
   */
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  // TODO Complex
//  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
//      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
};

}  // namespace caffe

#endif  // CAFFE_COMPLEX_MAGNITUDE_LAYER_HPP_
