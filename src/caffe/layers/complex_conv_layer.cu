#include <vector>

#include "caffe/layers/complex_conv_layer.hpp"

namespace caffe {

template <typename Dtype>
void ComplexConvolutionLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const std::complex<Dtype>* weight = this->RealToComplexBlobData_gpu(0);
  for (int i = 0; i < bottom.size(); ++i) {
    const std::complex<Dtype>* bottom_data = this->RealToComplexBottomData_gpu(bottom, i);
    std::complex<Dtype>* top_data = this->RealToComplexTopData_mutable_gpu(top, i);
    for (int n = 0; n < this->num_; ++n) {
      this->forward_gpu_gemm(bottom_data + n * this->bottom_dim_, weight,
          top_data + n * this->top_dim_);

      if (this->bias_term_) {
        const std::complex<Dtype>* bias = this->RealToComplexBlobData_gpu(1);
        this->forward_gpu_bias(top_data + n * this->top_dim_, bias);
      }
      this->SyncComplexTopData_gpu(top, i);
    }
  }
}

template <typename Dtype>
void ComplexConvolutionLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  bool any_propagate_down = true;
  for (int i = 0; i < top.size(); ++i) {
    any_propagate_down &= propagate_down[i];
  }

  const std::complex<Dtype>* weight;
  if (any_propagate_down) {
    weight = this->RealToComplexBlobData_gpu(0);
  }
  else {
    weight = NULL;
  }

  std::complex<Dtype>* weight_diff = NULL;
  if (this->param_propagate_down_[0]) {
    weight_diff = this->RealToComplexBlobDiff_mutable_gpu(0);
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
        // gradient w.r.t. weight. Note that we will accumulate diffs.
        if (this->param_propagate_down_[0]) {
          bool conj_bottom = true;
          this->weight_gpu_gemm(bottom_data + n * this->bottom_dim_,
              top_diff + n * this->top_dim_, weight_diff, conj_bottom);
        }
        // gradient w.r.t. bottom data, if necessary.
        if (propagate_down[i]) {
          bool conj_weight = true;
          this->backward_gpu_gemm(top_diff + n * this->top_dim_, weight,
              bottom_diff + n * this->bottom_dim_, conj_weight);
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

INSTANTIATE_LAYER_GPU_FUNCS(ComplexConvolutionLayer);

}  // namespace caffe
