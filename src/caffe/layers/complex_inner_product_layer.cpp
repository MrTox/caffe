#include <vector>
#include <complex>

#include "caffe/filler.hpp"
#include "caffe/layers/complex_inner_product_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void ComplexInnerProductLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  //
  CHECK_EQ(bottom[0]->shape(-1),2)
      << "Input blob should have last dimension of size 2 for the real/imaginary channel";

  const int num_output = this->layer_param_.inner_product_param().num_output();
  bias_term_ = this->layer_param_.inner_product_param().bias_term();
  transpose_ = this->layer_param_.inner_product_param().transpose();
  N_ = num_output;
  const int axis = bottom[0]->CanonicalAxisIndex(
      this->layer_param_.inner_product_param().axis());
  // Dimensions starting from "axis" are "flattened" into a single
  // length K_ vector. For example, if bottom[0]'s shape is (N, C, H, W),
  // and axis == 1, N inner products with dimension CHW are performed.
  K_ = bottom[0]->count(axis, bottom[0]->num_axes()-1);
  // Check if we need to set up the weights
  if (this->blobs_.size() > 0) {
    LOG(INFO) << "Skipping parameter initialization";
  } else {
    if (bias_term_) {
      this->blobs_.resize(2);
    } else {
      this->blobs_.resize(1);
    }
    // Initialize the weights
    vector<int> weight_shape(3);
    if (transpose_) {
      weight_shape[0] = K_;
      weight_shape[1] = N_;
    } else {
      weight_shape[0] = N_;
      weight_shape[1] = K_;
    }
    // Real/imaginary dimension
    weight_shape[2] = 2;
    this->blobs_[0].reset(new Blob<Dtype>(weight_shape));
    // fill the weights
    // Complex: fill real/imaginary components independently for now
    shared_ptr<Filler<Dtype> > weight_filler(GetFiller<Dtype>(
        this->layer_param_.inner_product_param().weight_filler()));
    weight_filler->Fill(this->blobs_[0].get());
    // If necessary, intiialize and fill the bias term
    if (bias_term_) {
      vector<int> bias_shape(3);
      bias_shape[0] = 1;
      bias_shape[1] = N_;
      bias_shape[2] = 2;
      this->blobs_[1].reset(new Blob<Dtype>(bias_shape));
      // Complex: fill real/imaginary components independently for now
      shared_ptr<Filler<Dtype> > bias_filler(GetFiller<Dtype>(
          this->layer_param_.inner_product_param().bias_filler()));
      bias_filler->Fill(this->blobs_[1].get());
    }
  }  // parameter initialization
  this->param_propagate_down_.resize(this->blobs_.size(), true);
}

template <typename Dtype>
void ComplexInnerProductLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  // Figure out the dimensions
  const int axis = bottom[0]->CanonicalAxisIndex(
      this->layer_param_.inner_product_param().axis());
  const int new_K = bottom[0]->count(axis, bottom[0]->num_axes()-1);
  CHECK_EQ(K_, new_K)
      << "Input size incompatible with inner product parameters.";
  // The first "axis" dimensions are independent inner products; the total
  // number of these is M_, the product over these dimensions.
  M_ = bottom[0]->count(0, axis);
  // The top shape will be the bottom shape with the flattened axes dropped,
  // and replaced by a single axis with dimension num_output (N_).
  vector<int> top_shape = bottom[0]->shape();
  top_shape.resize(axis + 2);
  top_shape[axis] = N_;
  // Real/imaginary dimension
  top_shape[axis+1] = 2;
  top[0]->Reshape(top_shape);
  // Set up the bias multiplier
  if (bias_term_) {
    vector<int> bias_shape(3);
    bias_shape[0] = 1;
    bias_shape[1] = M_;
    bias_shape[2] = 2;
    bias_multiplier_.Reshape(bias_shape);

    // Setting bias_data to complex(1,0)
    std::complex<Dtype>* bias_data = this->RealToComplex_mutable_cpu(bias_multiplier_.mutable_cpu_data());
    caffe_set(M_, std::complex<Dtype>(1), bias_data);
    this->SyncComplex_cpu(bias_data, bias_multiplier_.mutable_cpu_data());
  }

  conj_weight_.Reshape(this->blobs_[0]->shape());
  conj_bottom_.Reshape(bottom[0]->shape());
}

template <typename Dtype>
void ComplexInnerProductLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const std::complex<Dtype>* bottom_data = this->RealToComplexBottomData_cpu(bottom, 0);
  std::complex<Dtype>* top_data = this->RealToComplexTopData_mutable_cpu(top, 0);
  const std::complex<Dtype>* weight = this->RealToComplexBlobData_cpu(0);
  caffe_cpu_gemm<std::complex<Dtype> >(CblasNoTrans, transpose_ ? CblasNoTrans : CblasTrans,
      M_, N_, K_, std::complex<Dtype>(1),
      bottom_data, weight, std::complex<Dtype>(0), top_data);
  this->SyncComplexTopData_cpu(top, 0);

  if (bias_term_) {
    // TODO need safer to manage bias_multiplier data
    caffe_cpu_gemm<std::complex<Dtype> >(CblasNoTrans, CblasNoTrans, M_, N_, 1, std::complex<Dtype>(1),
        this->RealToComplex_cpu(bias_multiplier_.cpu_data()),
        this->RealToComplexBlobData_cpu(1), Dtype(1), top_data);
    this->SyncComplexTopData_cpu(top, 0);
  }
}

template <typename Dtype>
void ComplexInnerProductLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (this->param_propagate_down_[0]) {
    const std::complex<Dtype>* top_diff = this->RealToComplexTopDiff_cpu(top, 0);
    const std::complex<Dtype>* bottom_data = this->RealToComplexBottomData_cpu(bottom, 0);
    // Gradient with respect to weight
    if (transpose_) {
      caffe_cpu_gemm<std::complex<Dtype> >(CblasConjTrans, CblasNoTrans,
          K_, N_, M_,
          std::complex<Dtype>(1), bottom_data, top_diff,
          std::complex<Dtype>(1), this->RealToComplexBlobDiff_mutable_cpu(0));
    } else {
      // MKL and Xcode Accelerate don't support a conjugate-no-transform operation (AtlasConj)
      // so we will manuall make a conjugate copy.
      std::complex<Dtype>* conj_bottom_data = this->RealToComplex_mutable_cpu(this->conj_bottom_.mutable_cpu_data());
      for(int i = 0; i < conj_bottom_.count()/2; ++i) {
        conj_bottom_data[i] = std::conj(bottom_data[i]);
      }

//      caffe_cpu_gemm<std::complex<Dtype> >(CblasTrans, AtlasConj,
      caffe_cpu_gemm<std::complex<Dtype> >(CblasTrans, CblasNoTrans,
          N_, K_, M_,
          std::complex<Dtype>(1), top_diff, conj_bottom_data,
          std::complex<Dtype>(1), this->RealToComplexBlobDiff_mutable_cpu(0));
    }
    this->SyncComplexBlobDiff_cpu(0);
  }
  if (bias_term_ && this->param_propagate_down_[1]) {
    const std::complex<Dtype>* top_diff = this->RealToComplexTopDiff_cpu(top, 0);
    // Gradient with respect to bias
    caffe_cpu_gemv<std::complex<Dtype> >(CblasTrans, M_, N_, std::complex<Dtype>(1), top_diff,
        this->RealToComplex_cpu(bias_multiplier_.cpu_data()), std::complex<Dtype>(1),
        this->RealToComplexBlobDiff_mutable_cpu(1));
    this->SyncComplexBlobDiff_cpu(1);
  }
  if (propagate_down[0]) {
    const std::complex<Dtype>* top_diff = this->RealToComplexTopDiff_cpu(top, 0);
    // Gradient with respect to bottom data
    if (transpose_) {
      caffe_cpu_gemm<std::complex<Dtype> >(CblasNoTrans, CblasConjTrans,
          M_, K_, N_,
          std::complex<Dtype>(1), top_diff, this->RealToComplexBlobData_cpu(0),
          std::complex<Dtype>(0), this->RealToComplexBottomDiff_mutable_cpu(bottom, 0));
    } else {
      // MKL and Xcode Accelerate don't support a conjugate-no-transform operation (AtlasConj)
      // so we will manually make a conjugate copy.
      std::complex<Dtype>* conj_weight_data = this->RealToComplex_mutable_cpu(this->conj_weight_.mutable_cpu_data());
      const std::complex<Dtype>* weight_data = this->RealToComplexBlobData_cpu(0);
      for(int i = 0; i < conj_weight_.count()/2; ++i) {
        conj_weight_data[i] = std::conj(weight_data[i]);
      }

//      caffe_cpu_gemm<std::complex<Dtype> >(CblasNoTrans, AtlasConj,
      caffe_cpu_gemm<std::complex<Dtype> >(CblasNoTrans, CblasNoTrans,
          M_, K_, N_,
          std::complex<Dtype>(1), top_diff, conj_weight_data,
          std::complex<Dtype>(0), this->RealToComplexBottomDiff_mutable_cpu(bottom, 0));
    }
    this->SyncComplexBottomDiff_cpu(bottom, 0);
  }
}


// TODO Complex Uncomment when we implement Forward_gpu and Backward_gpu
//#ifdef CPU_ONLY
//STUB_GPU(ComplexInnerProductLayer);
//#endif

INSTANTIATE_CLASS(ComplexInnerProductLayer);
REGISTER_LAYER_CLASS(ComplexInnerProduct);

}  // namespace caffe
