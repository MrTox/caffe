#include <vector>

#include "caffe/layers/complex_deconv_layer.hpp"

namespace caffe {

template <typename Dtype>
void ComplexDeconvolutionLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const std::complex<Dtype>* weight = this->RealToComplexBlobData_gpu(0);
  for (int i = 0; i < bottom.size(); ++i) {
    const std::complex<Dtype>* bottom_data = this->RealToComplexBottomData_gpu(bottom, i);
    std::complex<Dtype>* top_data = this->RealToComplexTopData_mutable_gpu(top, i);
    for (int n = 0; n < this->num_; ++n) {
      bool conj_weight = false;
      this->backward_gpu_gemm(bottom_data + n * this->bottom_dim_, weight,
          top_data + n * this->top_dim_, conj_weight);
      if (this->bias_term_) {
        const std::complex<Dtype>* bias = this->RealToComplexBlobData_gpu(1);
        this->forward_gpu_bias(top_data + n * this->top_dim_, bias);
      }
    }
  }
}

template <typename Dtype>
void ComplexDeconvolutionLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  bool any_propagate_down = true;
  for (int i = 0; i < top.size(); ++i) {
    any_propagate_down &= propagate_down[i];
  }

  std::complex<Dtype>* weight_diff = NULL;
  if (this->param_propagate_down_[0]) {
    weight_diff = this->RealToComplexBlobDiff_mutable_gpu(0);
  }

  // MKL and Xcode Accelerate don't support a conjugate-no-transform operation (AtlasConj)
  // so we will manually make a conjugate copy.

  std::complex<Dtype>* conj_weight_data = NULL;
  std::complex<Dtype>* conj_bottom_data = NULL;
  if(any_propagate_down) {
    conj_weight_data = this->RealToComplex_mutable_gpu(this->conj_weight_.mutable_gpu_data());
    const std::complex<Dtype>* weight_data = this->RealToComplexBlobData_gpu(0);
    caffe_gpu_conj<std::complex<Dtype> >(conj_weight_.count()/2, weight_data, conj_weight_data);
  }
  if (this->param_propagate_down_[0]) {
    conj_bottom_data = this->RealToComplex_mutable_gpu(this->conj_bottom_.mutable_gpu_data());
  }

  for (int i = 0; i < top.size(); ++i) {
    const std::complex<Dtype>* top_diff;
    if (this->param_propagate_down_[0] || propagate_down[i] || (this->bias_term_ && this->param_propagate_down_[1]) ) {
      top_diff = this->RealToComplexTopDiff_gpu(top, i);
    }
    else {
      top_diff = NULL;
    }

    const std::complex<Dtype>* bottom_data;
    if (this->param_propagate_down_[0]) {
      bottom_data = this->RealToComplexBottomData_gpu(bottom, i);
    }
    else {
      bottom_data = NULL;
    }

    std::complex<Dtype>* bottom_diff = NULL;
    if(propagate_down[i]) {
      bottom_diff = this->RealToComplexBottomDiff_mutable_gpu(bottom, i);
    }

    // Bias gradient, if necessary.
    if (this->bias_term_ && this->param_propagate_down_[1]) {
      std::complex<Dtype>* bias_diff = this->RealToComplexBlobDiff_mutable_gpu(1);
      for (int n = 0; n < this->num_; ++n) {
        this->backward_gpu_bias(bias_diff, top_diff + n * this->top_dim_);
      }
      this->SyncComplexBlobDiff_gpu(1);
    }
    if (this->param_propagate_down_[0] || propagate_down[i]) {
      for (int n = 0; n < this->num_; ++n) {
        // Gradient w.r.t. weight. Note that we will accumulate diffs.
        if (this->param_propagate_down_[0]) {
          // MKL and Xcode Accelerate don't support a conjugate-no-transform operation (AtlasConj)
          // so we will manuall make a conjugate copy.
          caffe_gpu_conj<std::complex<Dtype> >(this->bottom_dim_, bottom_data + n*this->bottom_dim_, conj_bottom_data);

          bool conj_top_diff = false;
          this->weight_gpu_gemm(top_diff + n * this->top_dim_,
              conj_bottom_data, weight_diff, conj_top_diff);
        }
        // Gradient w.r.t. bottom data, if necessary, reusing the column buffer
        // we might have just computed above.
        if (propagate_down[i]) {
          this->forward_gpu_gemm(top_diff + n * this->top_dim_, conj_weight_data,
              bottom_diff + n * this->bottom_dim_,
              this->param_propagate_down_[0]);
        }
      }

      if (this->param_propagate_down_[0]) {
        this->SyncComplexBlobDiff_gpu(0);
      }

      if(propagate_down[i]) {
        this->SyncComplexBottomDiff_gpu(bottom, 0);
      }
    }
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(ComplexDeconvolutionLayer);

}  // namespace caffe
