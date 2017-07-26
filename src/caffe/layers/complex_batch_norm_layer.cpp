#include <algorithm>
#include <vector>

#include "caffe/layers/complex_batch_norm_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void ComplexBatchNormLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  BatchNormParameter param = this->layer_param_.batch_norm_param();
  moving_average_fraction_ = param.moving_average_fraction();
  use_global_stats_ = this->phase_ == TEST;
  if (param.has_use_global_stats())
    use_global_stats_ = param.use_global_stats();
  if (bottom[0]->num_axes() == 1)
    channels_ = 1;
  else
    channels_ = bottom[0]->shape(1);
  eps_ = param.eps();
  if (this->blobs_.size() > 0) {
    LOG(INFO) << "Skipping parameter initialization";
  } else {
    this->blobs_.resize(3);
    vector<int> sizeC;
    sizeC.push_back(channels_);
    sizeC.push_back(2);
    this->blobs_[0].reset(new Blob<Dtype>(sizeC));
    this->blobs_[1].reset(new Blob<Dtype>(sizeC));
    vector<int> size1;
    size1.push_back(1);
    this->blobs_[2].reset(new Blob<Dtype>(size1));

    for (int i = 0; i < 2; ++i) {
      caffe_set(this->blobs_[i]->count()/2, std::complex<Dtype>(0),
                this->RealToComplexBlobData_mutable_cpu(i));
      this->SyncComplexBlobData_cpu(i);
    }
    this->blobs_[2]->mutable_cpu_data()[0] = 0;
  }
}

template <typename Dtype>
void ComplexBatchNormLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  CHECK_EQ(bottom[0]->shape(-1), 2);
  if (bottom[0]->num_axes() >= 2)
    CHECK_EQ(bottom[0]->shape(1), channels_);
  top[0]->ReshapeLike(*bottom[0]);

  vector<int> sz;
  sz.push_back(channels_);
  // Add complex dimension
  sz.push_back(2);
  mean_.Reshape(sz);
  variance_.Reshape(sz);
  temp_.ReshapeLike(*bottom[0]);
  x_norm_.ReshapeLike(*bottom[0]);
  sz[0]=bottom[0]->shape(0);
  batch_sum_multiplier_.Reshape(sz);

  int spatial_dim = bottom[0]->count()/2/(channels_*bottom[0]->shape(0));
  if (spatial_sum_multiplier_.num_axes() == 0 ||
      spatial_sum_multiplier_.shape(0) != spatial_dim) {
    sz[0] = spatial_dim;
    spatial_sum_multiplier_.Reshape(sz);
    std::complex<Dtype>* multiplier_data = this->RealToComplex_mutable_cpu(spatial_sum_multiplier_.mutable_cpu_data());
    caffe_set(spatial_sum_multiplier_.count()/2, std::complex<Dtype>(1), multiplier_data);
    this->SyncComplex_cpu(multiplier_data, spatial_sum_multiplier_.mutable_cpu_data());
  }

  int numbychans = channels_*bottom[0]->shape(0);
  if (num_by_chans_.num_axes() == 0 ||
      num_by_chans_.shape(0) != numbychans) {
    sz[0] = numbychans;
    num_by_chans_.Reshape(sz);
    std::complex<Dtype>* multiplier_data = this->RealToComplex_mutable_cpu(batch_sum_multiplier_.mutable_cpu_data());
    caffe_set(batch_sum_multiplier_.count()/2, std::complex<Dtype>(1),
        multiplier_data);
    this->SyncComplex_cpu(multiplier_data, batch_sum_multiplier_.mutable_cpu_data());
  }
}

template <typename Dtype>
void ComplexBatchNormLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const std::complex<Dtype>* bottom_data = this->RealToComplexBottomData_cpu(bottom, 0);
  std::complex<Dtype>* top_data = this->RealToComplexTopData_mutable_cpu(top, 0);

  std::complex<Dtype>* mean_data = this->RealToComplex_mutable_cpu(mean_.mutable_cpu_data());
  std::complex<Dtype>* variance_data = this->RealToComplex_mutable_cpu(variance_.mutable_cpu_data());
  std::complex<Dtype>* num_by_chans_data = this->RealToComplex_mutable_cpu(num_by_chans_.mutable_cpu_data());
  const std::complex<Dtype>* spatial_sum_multiplier_data = this->RealToComplex_cpu(spatial_sum_multiplier_.cpu_data());
  const std::complex<Dtype>* batch_sum_multiplier_data = this->RealToComplex_cpu(batch_sum_multiplier_.cpu_data());
  std::complex<Dtype>* temp_data = this->RealToComplex_mutable_cpu(temp_.mutable_cpu_data());
  std::complex<Dtype>* x_norm_data = this->RealToComplex_mutable_cpu(x_norm_.mutable_cpu_data());

  int num = bottom[0]->shape(0);
  int spatial_dim = bottom[0]->count()/2/(bottom[0]->shape(0)*channels_);

  if (bottom[0] != top[0]) {
    caffe_copy(bottom[0]->count()/2, bottom_data, top_data);
  }

  if (use_global_stats_) {
    // use the stored mean/variance estimates.
    const Dtype scale_factor = this->blobs_[2]->cpu_data()[0] == 0 ?
        0 : 1 / this->blobs_[2]->cpu_data()[0];

    caffe_cpu_scale(mean_.count()/2, std::complex<Dtype>(scale_factor),
        this->RealToComplexBlobData_cpu(0), mean_data);
    this->SyncComplex_cpu(mean_data, mean_.mutable_cpu_data());

    caffe_cpu_scale(variance_.count()/2, std::complex<Dtype>(scale_factor),
        this->RealToComplexBlobData_cpu(1), variance_data);
    this->SyncComplex_cpu(variance_data, variance_.mutable_cpu_data());
  } else {
    // compute mean
    caffe_cpu_gemv<std::complex<Dtype> >(CblasNoTrans, channels_ * num, spatial_dim,
        std::complex<Dtype>(1. / (num * spatial_dim)), bottom_data,
        spatial_sum_multiplier_data, std::complex<Dtype>(0),
        num_by_chans_data);
    caffe_cpu_gemv<std::complex<Dtype> >(CblasTrans, num, channels_, std::complex<Dtype>(1),
        num_by_chans_data, batch_sum_multiplier_data, std::complex<Dtype>(0),
        mean_data);
    this->SyncComplex_cpu(mean_data, mean_.mutable_cpu_data());
  }

  // subtract mean
  caffe_cpu_gemm<std::complex<Dtype> >(CblasNoTrans, CblasNoTrans, num, channels_, 1, 1,
      batch_sum_multiplier_data, mean_data, std::complex<Dtype>(0.),
      num_by_chans_data);
  caffe_cpu_gemm<std::complex<Dtype> >(CblasNoTrans, CblasNoTrans, channels_ * num,
      spatial_dim, 1, std::complex<Dtype>(-1), num_by_chans_data,
      spatial_sum_multiplier_data, std::complex<Dtype>(1), top_data);

  if (!use_global_stats_) {
    // compute variance using var(X) = E((X-EX)^H*(X-EX))
    // (X-EX)^H*(X-EX)
    for(int i = 0; i < top[0]->count()/2; ++i) {
      temp_data[i] = std::conj(top_data[i])*top_data[i];
    }
    caffe_cpu_gemv<std::complex<Dtype> >(CblasNoTrans, channels_ * num, spatial_dim,
        std::complex<Dtype>(1. / (num * spatial_dim)), temp_data,
        spatial_sum_multiplier_data, std::complex<Dtype>(0),
        num_by_chans_data);
    caffe_cpu_gemv<std::complex<Dtype> >(CblasTrans, num, channels_, std::complex<Dtype>(1),
        num_by_chans_data, batch_sum_multiplier_data, std::complex<Dtype>(0),
        variance_data);  // E((X_EX)^H*(X-EX))

    // compute and save moving average
    this->blobs_[2]->mutable_cpu_data()[0] *= moving_average_fraction_;
    this->blobs_[2]->mutable_cpu_data()[0] += 1;
    caffe_cpu_axpby(mean_.count()/2, std::complex<Dtype>(1), mean_data,
        std::complex<Dtype>(moving_average_fraction_), this->RealToComplexBlobData_mutable_cpu(0));
    int m = bottom[0]->count()/2/channels_;
    std::complex<Dtype> bias_correction_factor = m > 1 ? std::complex<Dtype>((m)/(m-1)) : std::complex<Dtype>(1);
    caffe_cpu_axpby(variance_.count()/2, bias_correction_factor,
        variance_data, std::complex<Dtype>(moving_average_fraction_),
        this->RealToComplexBlobData_mutable_cpu(1));

    this->SyncComplexBlobData_cpu(0);
    this->SyncComplexBlobData_cpu(1);
  }

  // normalize variance
  caffe_add_scalar(variance_.count()/2, std::complex<Dtype>(eps_), variance_data);
  caffe_powx(variance_.count()/2, variance_data, std::complex<Dtype>(0.5),
             variance_data);
  this->SyncComplex_cpu(variance_data, variance_.mutable_cpu_data());

  // replicate variance to input size
  caffe_cpu_gemm<std::complex<Dtype> >(CblasNoTrans, CblasNoTrans, num, channels_, 1, std::complex<Dtype>(1),
      batch_sum_multiplier_data, variance_data, std::complex<Dtype>(0),
      num_by_chans_data);
  caffe_cpu_gemm<std::complex<Dtype> >(CblasNoTrans, CblasNoTrans, channels_ * num,
      spatial_dim, 1, std::complex<Dtype>(1), num_by_chans_data,
      spatial_sum_multiplier_data, std::complex<Dtype>(0), temp_data);
  caffe_div(top[0]->count()/2, top_data, temp_data, top_data);
  // TODO(cdoersch): The caching is only needed because later in-place layers
  //                 might clobber the data.  Can we skip this if they won't?
  caffe_copy(x_norm_.count()/2, top_data,
      x_norm_data);
  this->SyncComplex_cpu(x_norm_data, x_norm_.mutable_cpu_data());
  this->SyncComplex_cpu(temp_data, temp_.mutable_cpu_data());
}

template <typename Dtype>
void ComplexBatchNormLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  std::complex<Dtype>* mean_data = this->RealToComplex_mutable_cpu(mean_.mutable_cpu_data());
  std::complex<Dtype>* num_by_chans_data = this->RealToComplex_mutable_cpu(num_by_chans_.mutable_cpu_data());
  const std::complex<Dtype>* spatial_sum_multiplier_data = this->RealToComplex_cpu(spatial_sum_multiplier_.cpu_data());
  const std::complex<Dtype>* batch_sum_multiplier_data = this->RealToComplex_cpu(batch_sum_multiplier_.cpu_data());
  const std::complex<Dtype>* temp_data = this->RealToComplex_cpu(temp_.cpu_data());

  const std::complex<Dtype>* top_diff;
  if (bottom[0] != top[0]) {
    top_diff = this->RealToComplexTopDiff_cpu(top, 0);
  } else {
    std::complex<Dtype>* x_norm_diff = this->RealToComplex_mutable_cpu(x_norm_.mutable_cpu_diff());
    caffe_copy(x_norm_.count()/2, this->RealToComplexTopDiff_cpu(top, 0), x_norm_diff);
    top_diff = x_norm_diff;
  }
  std::complex<Dtype>* bottom_diff = this->RealToComplexBottomDiff_mutable_cpu(bottom, 0);
  if (use_global_stats_) {
    // Mean and variance are constant, so derivative is simple: dE(Y)/dX = dE/dY ./ sqrt(var(X) + eps)
    caffe_div(bottom[0]->count()/2, top_diff, temp_data, bottom_diff);
    return;
  }

  const std::complex<Dtype>* top_data = this->RealToComplex_cpu(x_norm_.cpu_data());
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
  for(int i = 0; i < bottom[0]->count()/2; ++i) {
    bottom_diff[i] = top_diff[i]*std::conj(top_data[i]);
  }

  caffe_cpu_gemv<std::complex<Dtype> >(CblasNoTrans, channels_ * num, spatial_dim, std::complex<Dtype>(1),
      bottom_diff, spatial_sum_multiplier_data, std::complex<Dtype>(0),
      num_by_chans_data);
  caffe_cpu_gemv<std::complex<Dtype> >(CblasTrans, num, channels_, std::complex<Dtype>(1),
      num_by_chans_data, batch_sum_multiplier_data, std::complex<Dtype>(0),
      mean_data);

  // TODO(gamma and beta)
  // 1) Copy mean_data to gamma_diff
  // 2) mean_data[i] *= conj(gamma[i])

  for(int i = 0; i < mean_.count()/2; ++i) {
    mean_data[i] = std::real(mean_data[i]);
  }


  // reshape (broadcast) the above
  caffe_cpu_gemm<std::complex<Dtype> >(CblasNoTrans, CblasNoTrans, num, channels_, 1, std::complex<Dtype>(1),
      batch_sum_multiplier_data, mean_data, std::complex<Dtype>(0),
      num_by_chans_data);
  caffe_cpu_gemm<std::complex<Dtype> >(CblasNoTrans, CblasNoTrans, channels_ * num,
      spatial_dim, 1, std::complex<Dtype>(1), num_by_chans_data,
      spatial_sum_multiplier_data, std::complex<Dtype>(0), bottom_diff);

  // sum(dE/dY \cdot Y) \cdot Y
  caffe_mul(bottom[0]->count()/2, top_data, bottom_diff, bottom_diff);

  // sum(dE/dY)-sum(dE/dY \cdot Y) \cdot Y
  caffe_cpu_gemv<std::complex<Dtype> >(CblasNoTrans, channels_ * num, spatial_dim, std::complex<Dtype>(1),
      top_diff, spatial_sum_multiplier_data, std::complex<Dtype>(0),
      num_by_chans_data);
  caffe_cpu_gemv<std::complex<Dtype> >(CblasTrans, num, channels_, std::complex<Dtype>(1),
      num_by_chans_data, batch_sum_multiplier_data, std::complex<Dtype>(0),
      mean_data);

  // TODO(gamma and beta)
  // 1) Copy mean_data to beta_diff
  // 2) mean_data[i] *= conj(gamma[i])

  // reshape (broadcast) the above to make
  // sum(dE/dY)-sum(dE/dY \cdot Y) \cdot Y
  caffe_cpu_gemm<std::complex<Dtype> >(CblasNoTrans, CblasNoTrans, num, channels_, 1, std::complex<Dtype>(1),
      batch_sum_multiplier_data, mean_data, std::complex<Dtype>(0),
      num_by_chans_data);
  caffe_cpu_gemm<std::complex<Dtype> >(CblasNoTrans, CblasNoTrans, num * channels_,
      spatial_dim, 1, std::complex<Dtype>(1), num_by_chans_data,
      spatial_sum_multiplier_data, std::complex<Dtype>(1), bottom_diff);

  // TODO(gamma and beta)
  // Instead of top_diff below, use top_diff[i]*conj(gamma[i])

  // dE/dY - mean(dE/dY)-mean(dE/dY \cdot Y) \cdot Y
  caffe_cpu_axpby(bottom[0]->count()/2, std::complex<Dtype>(1), top_diff,
      std::complex<Dtype>(-1. / (num * spatial_dim)), bottom_diff);

  // note: temp_ still contains sqrt(var(X)+eps), computed during the forward
  // pass.
  caffe_div(bottom[0]->count()/2, bottom_diff, temp_data, bottom_diff);
}


#ifdef CPU_ONLY
STUB_GPU(ComplexBatchNormLayer);
#endif

INSTANTIATE_CLASS(ComplexBatchNormLayer);
REGISTER_LAYER_CLASS(ComplexBatchNorm);
}  // namespace caffe
