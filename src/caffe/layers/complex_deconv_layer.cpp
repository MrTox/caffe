#include <vector>

#include "caffe/layers/complex_deconv_layer.hpp"

namespace caffe {

template <typename Dtype>
void ComplexDeconvolutionLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  BaseComplexConvolutionLayer<Dtype>::Reshape(bottom, top);

  conj_weight_.Reshape(this->blobs_[0]->shape());

  vector<int> conj_bottom_shape(1, this->bottom_dim_);
  // Add complex dimension
  conj_bottom_shape.push_back(2);
  conj_bottom_.Reshape(conj_bottom_shape);
}

template <typename Dtype>
void ComplexDeconvolutionLayer<Dtype>::compute_output_shape() {
  const int* kernel_shape_data = this->kernel_shape_.cpu_data();
  const int* stride_data = this->stride_.cpu_data();
  const int* pad_data = this->pad_.cpu_data();
  const int* dilation_data = this->dilation_.cpu_data();
  this->output_shape_.clear();
  for (int i = 0; i < this->num_spatial_axes_; ++i) {
    // i + 1 to skip channel axis
    const int input_dim = this->input_shape(i + 1);
    const int kernel_extent = dilation_data[i] * (kernel_shape_data[i] - 1) + 1;
    const int output_dim = stride_data[i] * (input_dim - 1)
        + kernel_extent - 2 * pad_data[i];
    this->output_shape_.push_back(output_dim);
  }
}

template <typename Dtype>
void ComplexDeconvolutionLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const std::complex<Dtype>* weight = this->RealToComplexBlobData_cpu(0);
  for (int i = 0; i < bottom.size(); ++i) {
    const std::complex<Dtype>* bottom_data = this->RealToComplexBottomData_cpu(bottom, i);
    std::complex<Dtype>* top_data = this->RealToComplexTopData_mutable_cpu(top, i);
    for (int n = 0; n < this->num_; ++n) {
      bool conj_weight = false;
      this->backward_cpu_gemm(bottom_data + n * this->bottom_dim_, weight,
          top_data + n * this->top_dim_, conj_weight);
      if (this->bias_term_) {
        const std::complex<Dtype>* bias = this->RealToComplexBlobData_cpu(1);
        this->forward_cpu_bias(top_data + n * this->top_dim_, bias);
      }
    }
  }
}

template <typename Dtype>
void ComplexDeconvolutionLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  bool any_propagate_down = true;
  for (int i = 0; i < top.size(); ++i) {
    any_propagate_down &= propagate_down[i];
  }

  const std::complex<Dtype>* weight;
  if (any_propagate_down) {
    weight = this->RealToComplexBlobData_cpu(0);
  }
  else {
    weight = NULL;
  }

  std::complex<Dtype>* weight_diff = NULL;
  if (this->param_propagate_down_[0]) {
    weight_diff = this->RealToComplexBlobDiff_mutable_cpu(0);
  }

  // MKL and Xcode Accelerate don't support a conjugate-no-transform operation (AtlasConj)
  // so we will manually make a conjugate copy.

  std::complex<Dtype>* conj_weight_data = NULL;
  std::complex<Dtype>* conj_bottom_data = NULL;
  if(any_propagate_down) {
    conj_weight_data = this->RealToComplex_mutable_cpu(this->conj_weight_.mutable_cpu_data());
    const std::complex<Dtype>* weight_data = this->RealToComplexBlobData_cpu(0);
    for(int i = 0; i < conj_weight_.count()/2; ++i) {
      conj_weight_data[i] = std::conj(weight_data[i]);
    }
  }
  if (this->param_propagate_down_[0]) {
    conj_bottom_data = this->RealToComplex_mutable_cpu(this->conj_bottom_.mutable_cpu_data());
  }

  for (int i = 0; i < top.size(); ++i) {
    const std::complex<Dtype>* top_diff;
    if (this->param_propagate_down_[0] || propagate_down[i] || (this->bias_term_ && this->param_propagate_down_[1]) ) {
      top_diff = this->RealToComplexTopDiff_cpu(top, i);
    }
    else {
      top_diff = NULL;
    }

    const std::complex<Dtype>* bottom_data;
    if (this->param_propagate_down_[0]) {
      bottom_data = this->RealToComplexBottomData_cpu(bottom, i);
    }
    else {
      bottom_data = NULL;
    }

    std::complex<Dtype>* bottom_diff = NULL;
    if(propagate_down[i]) {
      bottom_diff = this->RealToComplexBottomDiff_mutable_cpu(bottom, i);
    }

    // Bias gradient, if necessary.
    if (this->bias_term_ && this->param_propagate_down_[1]) {
      std::complex<Dtype>* bias_diff = this->RealToComplexBlobDiff_mutable_cpu(1);
      for (int n = 0; n < this->num_; ++n) {
        this->backward_cpu_bias(bias_diff, top_diff + n * this->top_dim_);
      }
      this->SyncComplexBlobDiff_cpu(1);
    }
    if (this->param_propagate_down_[0] || propagate_down[i]) {
      for (int n = 0; n < this->num_; ++n) {
        // Gradient w.r.t. weight. Note that we will accumulate diffs.
        if (this->param_propagate_down_[0]) {
          // MKL and Xcode Accelerate don't support a conjugate-no-transform operation (AtlasConj)
          // so we will manuall make a conjugate copy.
          for(int i = 0; i < this->bottom_dim_; ++i) {
            conj_bottom_data[i] = std::conj(bottom_data[i + n * this->bottom_dim_]);
          }

          bool conj_top_diff = false;
          this->weight_cpu_gemm(top_diff + n * this->top_dim_,
              conj_bottom_data, weight_diff, conj_top_diff);
        }
        // Gradient w.r.t. bottom data, if necessary, reusing the column buffer
        // we might have just computed above.
        if (propagate_down[i]) {
          this->forward_cpu_gemm(top_diff + n * this->top_dim_, conj_weight_data,
              bottom_diff + n * this->bottom_dim_,
              this->param_propagate_down_[0]);
        }
      }

      if (this->param_propagate_down_[0]) {
        this->SyncComplexBlobDiff_cpu(0);
      }

      if(propagate_down[i]) {
        this->SyncComplexBottomDiff_cpu(bottom, 0);
      }
    }
  }
}

#ifdef CPU_ONLY
STUB_GPU(ComplexDeconvolutionLayer);
#endif

INSTANTIATE_CLASS(ComplexDeconvolutionLayer);
REGISTER_LAYER_CLASS(ComplexDeconvolution);

}  // namespace caffe
