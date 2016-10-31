#include <algorithm>
#include <vector>

#include "caffe/layers/complex_batch_norm_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void ComplexBatchNormLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const std::complex<Dtype>* bottom_data = this->RealToComplexBottomData_gpu(bottom, 0);
  std::complex<Dtype>* top_data = this->RealToComplexTopData_mutable_gpu(top, 0);

  std::complex<Dtype>* mean_data = this->RealToComplex_mutable_gpu(mean_.mutable_gpu_data());
  std::complex<Dtype>* variance_data = this->RealToComplex_mutable_gpu(variance_.mutable_gpu_data());
  std::complex<Dtype>* num_by_chans_data = this->RealToComplex_mutable_gpu(num_by_chans_.mutable_gpu_data());
  const std::complex<Dtype>* spatial_sum_multiplier_data = this->RealToComplex_gpu(spatial_sum_multiplier_.gpu_data());
  const std::complex<Dtype>* batch_sum_multiplier_data = this->RealToComplex_gpu(batch_sum_multiplier_.gpu_data());
  std::complex<Dtype>* temp_data = this->RealToComplex_mutable_gpu(temp_.mutable_gpu_data());
  std::complex<Dtype>* x_norm_data = this->RealToComplex_mutable_gpu(x_norm_.mutable_gpu_data());

  int num = bottom[0]->shape(0);
  int spatial_dim = bottom[0]->count()/2/(bottom[0]->shape(0)*channels_);

  if (bottom[0] != top[0]) {
    caffe_copy(bottom[0]->count()/2, bottom_data, top_data);
  }

  if (use_global_stats_) {
    // use the stored mean/variance estimates.
    const Dtype scale_factor = this->blobs_[2]->cpu_data()[0] == 0 ?
        0 : 1 / this->blobs_[2]->cpu_data()[0];

    caffe_gpu_scale(mean_.count()/2, std::complex<Dtype>(scale_factor),
        this->RealToComplexBlobData_gpu(0), mean_data);
    this->SyncComplex_gpu(mean_data, mean_.mutable_gpu_data());

    caffe_gpu_scale(variance_.count()/2, std::complex<Dtype>(scale_factor),
        this->RealToComplexBlobData_gpu(1), variance_data);
    this->SyncComplex_gpu(variance_data, variance_.mutable_gpu_data());
  } else {
    // compute mean
    caffe_gpu_gemv<std::complex<Dtype> >(CblasNoTrans, channels_ * num, spatial_dim,
        std::complex<Dtype>(1. / (num * spatial_dim)), bottom_data,
        spatial_sum_multiplier_data, std::complex<Dtype>(0),
        num_by_chans_data);

    caffe_gpu_gemv<std::complex<Dtype> >(CblasTrans, num, channels_, std::complex<Dtype>(1),
        num_by_chans_data, batch_sum_multiplier_data, std::complex<Dtype>(0),
        mean_data);
    this->SyncComplex_gpu(mean_data, mean_.mutable_gpu_data());
  }

  // subtract mean
  caffe_gpu_gemm<std::complex<Dtype> >(CblasNoTrans, CblasNoTrans, num, channels_, 1, 1,
      batch_sum_multiplier_data, mean_data, std::complex<Dtype>(0.),
      num_by_chans_data);
  caffe_gpu_gemm<std::complex<Dtype> >(CblasNoTrans, CblasNoTrans, channels_ * num,
      spatial_dim, 1, std::complex<Dtype>(-1), num_by_chans_data,
      spatial_sum_multiplier_data, std::complex<Dtype>(1), top_data);

  if (!use_global_stats_) {
    // compute variance using var(X) = E((X-EX)^H*(X-EX))
    // (X-EX)^H*(X-EX)
    caffe_gpu_mul(top[0]->count()/2, top_data, top_data, temp_data, true);
    caffe_gpu_gemv<std::complex<Dtype> >(CblasNoTrans, channels_ * num, spatial_dim,
        std::complex<Dtype>(1. / (num * spatial_dim)), temp_data,
        spatial_sum_multiplier_data, std::complex<Dtype>(0),
        num_by_chans_data);
    caffe_gpu_gemv<std::complex<Dtype> >(CblasTrans, num, channels_, std::complex<Dtype>(1),
        num_by_chans_data, batch_sum_multiplier_data, std::complex<Dtype>(0),
        variance_data);  // E((X_EX)^H*(X-EX))

    // compute and save moving average
    this->blobs_[2]->mutable_cpu_data()[0] *= moving_average_fraction_;
    this->blobs_[2]->mutable_cpu_data()[0] += 1;
    caffe_gpu_axpby(mean_.count()/2, std::complex<Dtype>(1), mean_data,
        std::complex<Dtype>(moving_average_fraction_), this->RealToComplexBlobData_mutable_gpu(0));
    int m = bottom[0]->count()/2/channels_;
    std::complex<Dtype> bias_correction_factor = m > 1 ? std::complex<Dtype>((m)/(m-1)) : std::complex<Dtype>(1);
    caffe_gpu_axpby(variance_.count()/2, bias_correction_factor,
        variance_data, std::complex<Dtype>(moving_average_fraction_),
        this->RealToComplexBlobData_mutable_gpu(1));

    this->SyncComplexBlobData_gpu(0);
    this->SyncComplexBlobData_gpu(1);
  }


  // normalize variance
  caffe_gpu_add_scalar(variance_.count()/2, std::complex<Dtype>(eps_), variance_data);

  caffe_gpu_powx(variance_.count()/2, variance_data, Dtype(0.5),
             variance_data);

  this->SyncComplex_gpu(variance_data, variance_.mutable_gpu_data());

  // replicate variance to input size
  caffe_gpu_gemm<std::complex<Dtype> >(CblasNoTrans, CblasNoTrans, num, channels_, 1, std::complex<Dtype>(1),
      batch_sum_multiplier_data, variance_data, std::complex<Dtype>(0),
      num_by_chans_data);
  caffe_gpu_gemm<std::complex<Dtype> >(CblasNoTrans, CblasNoTrans, channels_ * num,
      spatial_dim, 1, std::complex<Dtype>(1), num_by_chans_data,
      spatial_sum_multiplier_data, std::complex<Dtype>(0), temp_data);

  caffe_gpu_div(top[0]->count()/2, top_data, temp_data, top_data);
  // TODO(cdoersch): The caching is only needed because later in-place layers
  //                 might clobber the data.  Can we skip this if they won't?
  caffe_copy(x_norm_.count()/2, top_data,
      x_norm_data);
  this->SyncComplex_gpu(x_norm_data, x_norm_.mutable_gpu_data());
  this->SyncComplex_gpu(temp_data, temp_.mutable_gpu_data());
}

template <typename Dtype>
void ComplexBatchNormLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  std::complex<Dtype>* mean_data = this->RealToComplex_mutable_gpu(mean_.mutable_gpu_data());
  std::complex<Dtype>* num_by_chans_data = this->RealToComplex_mutable_gpu(num_by_chans_.mutable_gpu_data());
  const std::complex<Dtype>* spatial_sum_multiplier_data = this->RealToComplex_gpu(spatial_sum_multiplier_.gpu_data());
  const std::complex<Dtype>* batch_sum_multiplier_data = this->RealToComplex_gpu(batch_sum_multiplier_.gpu_data());
  const std::complex<Dtype>* temp_data = this->RealToComplex_gpu(temp_.gpu_data());

  const std::complex<Dtype>* top_diff;
  if (bottom[0] != top[0]) {
    top_diff = this->RealToComplexTopDiff_gpu(top, 0);
  } else {
    std::complex<Dtype>* x_norm_diff = this->RealToComplex_mutable_gpu(x_norm_.mutable_gpu_diff());
    caffe_copy(x_norm_.count()/2, this->RealToComplexTopDiff_gpu(top, 0), x_norm_diff);
    top_diff = x_norm_diff;
  }
  std::complex<Dtype>* bottom_diff = this->RealToComplexBottomDiff_mutable_gpu(bottom, 0);
  if (use_global_stats_) {
    // Mean and variance are constant, so derivative is simple: dE(Y)/dX = dE/dY ./ sqrt(var(X) + eps)
    caffe_gpu_div(bottom[0]->count()/2, top_diff, temp_data, bottom_diff);
    return;
  }

  const std::complex<Dtype>* top_data = this->RealToComplex_gpu(x_norm_.gpu_data());
  int num = bottom[0]->shape()[0];
  int spatial_dim = bottom[0]->count()/2/(bottom[0]->shape(0)*channels_);

  // if Y = (X-mean(X))/(sqrt(var(X)+eps)), then
  //
  // dE(Y)/dX =
  //   (dE/dY - mean(dE/dY) - real(mean(dE/dY \cdot conj(Y))) \cdot Y)
  //     ./ sqrt(var(X) + eps)
  //
  // where \cdot and ./ are hadamard product and elementwise division,
  // respectively, dE/dY is the top diff, and mean/var/sum are all computed
  // along all dimensions except the channels dimension.  In the above
  // equation, the operations allow for expansion (i.e. broadcast) along all
  // dimensions except the channels dimension where required.

  // sum(dE/dY \cdot conj(Y))
  caffe_gpu_mul(bottom[0]->count()/2, top_data, top_diff, bottom_diff, true);

  caffe_gpu_gemv<std::complex<Dtype> >(CblasNoTrans, channels_ * num, spatial_dim, std::complex<Dtype>(1),
      bottom_diff, spatial_sum_multiplier_data, std::complex<Dtype>(0),
      num_by_chans_data);
  caffe_gpu_gemv<std::complex<Dtype> >(CblasTrans, num, channels_, std::complex<Dtype>(1),
      num_by_chans_data, batch_sum_multiplier_data, std::complex<Dtype>(0),
      mean_data);

  // TODO(gamma and beta)
  // 1) Copy mean_data to gamma_diff
  // 2) mean_data[i] *= conj(gamma[i])

  std::complex<Dtype>* mean_data_cpu = this->RealToComplex_mutable_cpu(mean_.mutable_cpu_data());
  for(int i = 0; i < mean_.count()/2; ++i) {
    mean_data_cpu[i] = std::real(mean_data_cpu[i]);
  }
  this->SyncComplex_cpu(mean_data_cpu, mean_.mutable_cpu_data());


  // reshape (broadcast) the above
  caffe_gpu_gemm<std::complex<Dtype> >(CblasNoTrans, CblasNoTrans, num, channels_, 1, std::complex<Dtype>(1),
      batch_sum_multiplier_data, mean_data, std::complex<Dtype>(0),
      num_by_chans_data);
  caffe_gpu_gemm<std::complex<Dtype> >(CblasNoTrans, CblasNoTrans, channels_ * num,
      spatial_dim, 1, std::complex<Dtype>(1), num_by_chans_data,
      spatial_sum_multiplier_data, std::complex<Dtype>(0), bottom_diff);

  // sum(dE/dY \cdot Y) \cdot Y
  caffe_gpu_mul(bottom[0]->count()/2, top_data, bottom_diff, bottom_diff);

  // sum(dE/dY)-sum(dE/dY \cdot Y) \cdot Y
  caffe_gpu_gemv<std::complex<Dtype> >(CblasNoTrans, channels_ * num, spatial_dim, std::complex<Dtype>(1),
      top_diff, spatial_sum_multiplier_data, std::complex<Dtype>(0),
      num_by_chans_data);
  caffe_gpu_gemv<std::complex<Dtype> >(CblasTrans, num, channels_, std::complex<Dtype>(1),
      num_by_chans_data, batch_sum_multiplier_data, std::complex<Dtype>(0),
      mean_data);

  // TODO(gamma and beta)
  // 1) Copy mean_data to beta_diff
  // 2) mean_data[i] *= conj(gamma[i])

  // reshape (broadcast) the above to make
  // sum(dE/dY)-sum(dE/dY \cdot Y) \cdot Y
  caffe_gpu_gemm<std::complex<Dtype> >(CblasNoTrans, CblasNoTrans, num, channels_, 1, std::complex<Dtype>(1),
      batch_sum_multiplier_data, mean_data, std::complex<Dtype>(0),
      num_by_chans_data);
  caffe_gpu_gemm<std::complex<Dtype> >(CblasNoTrans, CblasNoTrans, num * channels_,
      spatial_dim, 1, std::complex<Dtype>(1), num_by_chans_data,
      spatial_sum_multiplier_data, std::complex<Dtype>(1), bottom_diff);

  // TODO(gamma and beta)
  // Instead of top_diff below, use top_diff[i]*conj(gamma[i])

  // dE/dY - mean(dE/dY)-mean(dE/dY \cdot Y) \cdot Y
  caffe_gpu_axpby(bottom[0]->count()/2, std::complex<Dtype>(1), top_diff,
      std::complex<Dtype>(-1. / (num * spatial_dim)), bottom_diff);

  // note: temp_ still contains sqrt(var(X)+eps), computed during the forward
  // pass.
  caffe_gpu_div(bottom[0]->count()/2, bottom_diff, temp_data, bottom_diff);
}

INSTANTIATE_LAYER_GPU_FUNCS(ComplexBatchNormLayer);

}  // namespace caffe
