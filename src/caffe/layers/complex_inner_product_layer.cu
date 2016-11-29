#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layers/complex_inner_product_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void ComplexInnerProductLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const std::complex<Dtype>* bottom_data = this->RealToComplexBottomData_gpu(bottom, 0);
  std::complex<Dtype>* top_data = this->RealToComplexTopData_mutable_gpu(top, 0);
  const std::complex<Dtype>* weight = this->RealToComplexBlobData_cpu(0);
  if (M_ == 1) {
    caffe_gpu_gemv<std::complex<Dtype> >(CblasNoTrans, N_, K_, std::complex<Dtype>(1),
        weight, bottom_data, std::complex<Dtype>(0), top_data);
    if (bias_term_) {
      caffe_gpu_axpy<std::complex<Dtype> >(N_, this->RealToComplex_cpu(bias_multiplier_.cpu_data())[0],
          this->RealToComplexBlobData_gpu(1), top_data);
    }
  } else {
    caffe_gpu_gemm<std::complex<Dtype> >(CblasNoTrans,
                          transpose_ ? CblasNoTrans : CblasTrans,
                          M_, N_, K_, std::complex<Dtype>(1),
                          bottom_data, weight, std::complex<Dtype>(0), top_data);
    if (bias_term_) {
      caffe_gpu_gemm<std::complex<Dtype> >(CblasNoTrans, CblasNoTrans, M_, N_, 1, std::complex<Dtype>(1),
          this->RealToComplex_gpu(bias_multiplier_.gpu_data()),
          this->RealToComplexBlobData_gpu(1), std::complex<Dtype>(1), top_data);
    }
  }
  this->SyncComplexTopData_gpu(top, 0);
}

template <typename Dtype>
void ComplexInnerProductLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (this->param_propagate_down_[0]) {
    const std::complex<Dtype>* top_diff = this->RealToComplexTopDiff_gpu(top, 0);
    const std::complex<Dtype>* bottom_data = this->RealToComplexBottomData_cpu(bottom, 0);
    // Gradient with respect to weight
    if (transpose_) {
      caffe_gpu_gemm<std::complex<Dtype> >(CblasConjTrans, CblasNoTrans,
          K_, N_, M_,
          std::complex<Dtype>(1), bottom_data, top_diff,
          std::complex<Dtype>(1), this->RealToComplexBlobDiff_mutable_gpu(0));
    } else {
      // MKL and Xcode Accelerate don't support a conjugate-no-transform operation (AtlasConj)
      // so we will manuall make a conjugate copy.
      std::complex<Dtype>* conj_bottom_data = this->RealToComplex_mutable_gpu(this->conj_bottom_.mutable_gpu_data());
      caffe_gpu_conj<std::complex<Dtype> >(conj_bottom_.count()/2, bottom_data, conj_bottom_data);

//      caffe_cpu_gemm<std::complex<Dtype> >(CblasTrans, AtlasConj,
      caffe_gpu_gemm<std::complex<Dtype> >(CblasTrans, CblasNoTrans,
          N_, K_, M_,
          std::complex<Dtype>(1), top_diff, conj_bottom_data,
          std::complex<Dtype>(1), this->RealToComplexBlobDiff_mutable_gpu(0));
    }
    this->SyncComplexBlobDiff_gpu(0);
  }
  if (bias_term_ && this->param_propagate_down_[1]) {
    const std::complex<Dtype>* top_diff = this->RealToComplexTopDiff_gpu(top,0);
    // Gradient with respect to bias
    caffe_gpu_gemv<std::complex<Dtype> >(CblasTrans, M_, N_, std::complex<Dtype>(1), top_diff,
        this->RealToComplex_gpu(bias_multiplier_.gpu_data()), std::complex<Dtype>(1),
        this->RealToComplexBlobDiff_mutable_gpu(1));
    this->SyncComplexBlobDiff_gpu(1);
  }
  if (propagate_down[0]) {
    const std::complex<Dtype>* top_diff = this->RealToComplexTopDiff_gpu(top,0);
    // Gradient with respect to bottom data
    if (transpose_) {
      caffe_gpu_gemm<std::complex<Dtype> >(CblasNoTrans, CblasConjTrans,
          M_, K_, N_,
          std::complex<Dtype>(1), top_diff, this->RealToComplexBlobData_gpu(0),
          std::complex<Dtype>(0), this->RealToComplexBottomDiff_mutable_gpu(bottom, 0));
    } else {
      // MKL and Xcode Accelerate don't support a conjugate-no-transform operation (AtlasConj)
      // so we will manually make a conjugate copy.
      std::complex<Dtype>* conj_weight_data = this->RealToComplex_mutable_gpu(this->conj_weight_.mutable_gpu_data());
      const std::complex<Dtype>* weight_data = this->RealToComplexBlobData_gpu(0);
      caffe_gpu_conj<std::complex<Dtype> >(conj_weight_.count()/2, weight_data, conj_weight_data);

//      caffe_cpu_gemm<std::complex<Dtype> >(CblasNoTrans, AtlasConj,
      caffe_gpu_gemm<std::complex<Dtype> >(CblasNoTrans, CblasNoTrans,
          M_, K_, N_,
         std::complex<Dtype>(1), top_diff, conj_weight_data,
         std::complex<Dtype>(0), this->RealToComplexBottomDiff_mutable_gpu(bottom, 0));
    }
    this->SyncComplexBottomDiff_gpu(bottom, 0);
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(ComplexInnerProductLayer);

}  // namespace caffe
