#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layers/complex_inner_product_layer.hpp"

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

namespace caffe {

#ifndef CPU_ONLY
extern cudaDeviceProp CAFFE_TEST_CUDA_PROP;
#endif

template <typename TypeParam>
class ComplexInnerProductLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;
 protected:
  ComplexInnerProductLayerTest()
      : blob_bottom_(new Blob<Dtype>()),
        blob_bottom_nobatch_(new Blob<Dtype>()),
        blob_top_(new Blob<Dtype>()) {
    std::vector<int> bottom_size(5);
    bottom_size[0] = 2;
    bottom_size[1] = 3;
    bottom_size[2] = 4;
    bottom_size[3] = 5;
    bottom_size[4] = 2;
//    bottom_size[0] = 1;
//    bottom_size[1] = 1;
//    bottom_size[2] = 1;
//    bottom_size[3] = 1;
//    bottom_size[4] = 2;
    blob_bottom_->Reshape(bottom_size);
    std::vector<int> bottom_nobatch_size(5);
    bottom_nobatch_size[0] = 1;
    bottom_nobatch_size[1] = 2;
    bottom_nobatch_size[2] = 3;
    bottom_nobatch_size[3] = 4;
    bottom_nobatch_size[4] = 2;
    blob_bottom_nobatch_->Reshape(bottom_nobatch_size);

    // fill the values
    FillerParameter filler_param;
    UniformFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_);
    blob_top_vec_.push_back(blob_top_);
  }
  virtual ~ComplexInnerProductLayerTest() {
    delete blob_bottom_;
    delete blob_bottom_nobatch_;
    delete blob_top_;
  }

  Blob<Dtype>* const blob_bottom_;
  Blob<Dtype>* const blob_bottom_nobatch_;
  Blob<Dtype>* const blob_top_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(ComplexInnerProductLayerTest, TestDtypesAndDevices);

TYPED_TEST(ComplexInnerProductLayerTest, TestSetUp) {
  typedef typename TypeParam::Dtype Dtype;
  this->blob_bottom_vec_.push_back(this->blob_bottom_);
  LayerParameter layer_param;
  InnerProductParameter* inner_product_param =
      layer_param.mutable_inner_product_param();
  inner_product_param->set_num_output(10);
  shared_ptr<ComplexInnerProductLayer<Dtype> > layer(
      new ComplexInnerProductLayer<Dtype>(layer_param));
  layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(3, this->blob_top_->num_axes());
  EXPECT_EQ(2, this->blob_top_->shape(0));
  EXPECT_EQ(10, this->blob_top_->shape(1));
  EXPECT_EQ(2, this->blob_top_->shape(2));
}

/** @brief TestSetUp while toggling tranpose flag
 */
TYPED_TEST(ComplexInnerProductLayerTest, TestSetUpTranposeFalse) {
  typedef typename TypeParam::Dtype Dtype;
  this->blob_bottom_vec_.push_back(this->blob_bottom_);
  LayerParameter layer_param;
  InnerProductParameter* inner_product_param =
      layer_param.mutable_inner_product_param();
  inner_product_param->set_num_output(10);
  inner_product_param->set_transpose(false);
  shared_ptr<ComplexInnerProductLayer<Dtype> > layer(
      new ComplexInnerProductLayer<Dtype>(layer_param));
  layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(3, this->blob_top_->num_axes());
  EXPECT_EQ(2, this->blob_top_->shape(0));
  EXPECT_EQ(10, this->blob_top_->shape(1));
  EXPECT_EQ(2, this->blob_top_->shape(2));
  EXPECT_EQ(3, layer->blobs()[0]->num_axes());
  EXPECT_EQ(10, layer->blobs()[0]->shape(0));
  EXPECT_EQ(60, layer->blobs()[0]->shape(1));
  EXPECT_EQ(2, layer->blobs()[0]->shape(2));
}

/** @brief TestSetUp while toggling tranpose flag
 */
TYPED_TEST(ComplexInnerProductLayerTest, TestSetUpTranposeTrue) {
  typedef typename TypeParam::Dtype Dtype;
  this->blob_bottom_vec_.push_back(this->blob_bottom_);
  LayerParameter layer_param;
  InnerProductParameter* inner_product_param =
      layer_param.mutable_inner_product_param();
  inner_product_param->set_num_output(10);
  inner_product_param->set_transpose(true);
  shared_ptr<ComplexInnerProductLayer<Dtype> > layer(
      new ComplexInnerProductLayer<Dtype>(layer_param));
  layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(3, this->blob_top_->num_axes());
  EXPECT_EQ(2, this->blob_top_->shape(0));
  EXPECT_EQ(10, this->blob_top_->shape(1));
  EXPECT_EQ(2, this->blob_top_->shape(2));
  EXPECT_EQ(3, layer->blobs()[0]->num_axes());
  EXPECT_EQ(60, layer->blobs()[0]->shape(0));
  EXPECT_EQ(10, layer->blobs()[0]->shape(1));
  EXPECT_EQ(2, layer->blobs()[0]->shape(2));
}

TYPED_TEST(ComplexInnerProductLayerTest, TestForwardSimple) {
  typedef typename TypeParam::Dtype Dtype;

  Blob<Dtype>* bottom = new Blob<Dtype>();
  std::vector<int> bottom_size(5);
  bottom_size[0] = 1;
  bottom_size[1] = 1;
  bottom_size[2] = 1;
  bottom_size[3] = 1;
  bottom_size[4] = 2;
  bottom->Reshape(bottom_size);

  this->blob_bottom_vec_.push_back(bottom);

  LayerParameter layer_param;
  InnerProductParameter* inner_product_param =
      layer_param.mutable_inner_product_param();
  inner_product_param->set_num_output(1);
  shared_ptr<ComplexInnerProductLayer<Dtype> > layer(
      new ComplexInnerProductLayer<Dtype>(layer_param));
  layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);

  EXPECT_EQ(3, this->blob_top_->num_axes());
  EXPECT_EQ(1, this->blob_top_->shape(0));
  EXPECT_EQ(1, this->blob_top_->shape(1));
  EXPECT_EQ(2, this->blob_top_->shape(2));

  std::complex<Dtype> bot_complex(2*static_cast <float> (rand()) / static_cast <float> (RAND_MAX)-1, 2*static_cast <float> (rand()) / static_cast <float> (RAND_MAX)-1);
  std::complex<Dtype> weight_complex(2*static_cast <float> (rand()) / static_cast <float> (RAND_MAX)-1, 2*static_cast <float> (rand()) / static_cast <float> (RAND_MAX)-1);
  std::complex<Dtype> bias_complex(2*static_cast <float> (rand()) / static_cast <float> (RAND_MAX)-1, 2*static_cast <float> (rand()) / static_cast <float> (RAND_MAX)-1);
  std::complex<Dtype> expected_top_complex = bot_complex*weight_complex + bias_complex;

  Dtype* bot_data = bottom->mutable_cpu_data();
  Dtype* weight_data = layer->blobs()[0]->mutable_cpu_data();
  Dtype* bias_data = layer->blobs()[1]->mutable_cpu_data();

  bot_data[0] = std::real(bot_complex);
  bot_data[1] = std::imag(bot_complex);
  weight_data[0] = std::real(weight_complex);
  weight_data[1] = std::imag(weight_complex);
  bias_data[0] = std::real(bias_complex);
  bias_data[1] = std::imag(bias_complex);

  layer->Forward(this->blob_bottom_vec_, this->blob_top_vec_);

  const Dtype* top_data = this->blob_top_->cpu_data();

  EXPECT_FLOAT_EQ(std::real(expected_top_complex), top_data[0]);
  EXPECT_FLOAT_EQ(std::imag(expected_top_complex), top_data[1]);

  delete bottom;
}

TYPED_TEST(ComplexInnerProductLayerTest, TestForwardSimple2) {
  typedef typename TypeParam::Dtype Dtype;

  int C_in = 100;

  Blob<Dtype>* bottom = new Blob<Dtype>();
  std::vector<int> bottom_size(5);
  bottom_size[0] = 1;
  bottom_size[1] = C_in;
  bottom_size[2] = 1;
  bottom_size[3] = 1;
  bottom_size[4] = 2;
  bottom->Reshape(bottom_size);

  this->blob_bottom_vec_.push_back(bottom);

  LayerParameter layer_param;
  InnerProductParameter* inner_product_param =
      layer_param.mutable_inner_product_param();
  inner_product_param->set_num_output(1);
  shared_ptr<ComplexInnerProductLayer<Dtype> > layer(
      new ComplexInnerProductLayer<Dtype>(layer_param));
  layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);

  EXPECT_EQ(3, this->blob_top_->num_axes());
  EXPECT_EQ(1, this->blob_top_->shape(0));
  EXPECT_EQ(1, this->blob_top_->shape(1));
  EXPECT_EQ(2, this->blob_top_->shape(2));

  Dtype* bot_data = bottom->mutable_cpu_data();
  Dtype* weight_data = layer->blobs()[0]->mutable_cpu_data();
  Dtype* bias_data = layer->blobs()[1]->mutable_cpu_data();

  std::complex<Dtype>* bot_complex = new std::complex<Dtype>[C_in];
  std::complex<Dtype>* weight_complex = new std::complex<Dtype>[C_in];

  std::complex<Dtype> bias_complex(2*static_cast <float> (rand()) / static_cast <float> (RAND_MAX)-1, 2*static_cast <float> (rand()) / static_cast <float> (RAND_MAX)-1);
  bias_data[0] = std::real(bias_complex);
  bias_data[1] = std::imag(bias_complex);

  std::complex<Dtype> expected_top_complex(0,0);
  for(int i=0; i < C_in; i++) {
    bot_complex[i] = std::complex<Dtype>(2*static_cast <float> (rand()) / static_cast <float> (RAND_MAX)-1, 2*static_cast <float> (rand()) / static_cast <float> (RAND_MAX)-1);
    weight_complex[i] = std::complex<Dtype>(2*static_cast <float> (rand()) / static_cast <float> (RAND_MAX)-1, 2*static_cast <float> (rand()) / static_cast <float> (RAND_MAX)-1);

    bot_data[2*i] = std::real(bot_complex[i]);
    bot_data[2*i+1] = std::imag(bot_complex[i]);
    weight_data[2*i] = std::real(weight_complex[i]);
    weight_data[2*i+1] = std::imag(weight_complex[i]);

    expected_top_complex += bot_complex[i]*weight_complex[i];
  }
  expected_top_complex += bias_complex;

  layer->Forward(this->blob_bottom_vec_, this->blob_top_vec_);

  const Dtype* top_data = this->blob_top_->cpu_data();

  EXPECT_NEAR(std::real(expected_top_complex), top_data[0], 1e-5*std::abs(expected_top_complex));
  EXPECT_NEAR(std::imag(expected_top_complex), top_data[1], 1e-5*std::abs(expected_top_complex));

  delete bottom;
  delete[] bot_complex;
  delete[] weight_complex;
}

/**
 * @brief Init. an IP layer without transpose + random weights,
 * run Forward, save the result.
 * Init. another IP layer with transpose.
 * manually copy and transpose the weights from the first IP layer,
 * then run Forward on the same input and check that the result is the same
 */
TYPED_TEST(ComplexInnerProductLayerTest, TestForwardTranspose) {
  typedef typename TypeParam::Dtype Dtype;
  this->blob_bottom_vec_.push_back(this->blob_bottom_);
  bool IS_VALID_CUDA = false;
#ifndef CPU_ONLY
  IS_VALID_CUDA = CAFFE_TEST_CUDA_PROP.major >= 2;
#endif
  if (Caffe::mode() == Caffe::CPU ||
      sizeof(Dtype) == 4 || IS_VALID_CUDA) {
    LayerParameter layer_param;
    InnerProductParameter* inner_product_param =
        layer_param.mutable_inner_product_param();
    inner_product_param->set_num_output(10);
    inner_product_param->mutable_weight_filler()->set_type("uniform");
    inner_product_param->mutable_bias_filler()->set_type("uniform");
    inner_product_param->mutable_bias_filler()->set_min(1);
    inner_product_param->mutable_bias_filler()->set_max(2);
    inner_product_param->set_transpose(false);
    shared_ptr<ComplexInnerProductLayer<Dtype> > layer(
        new ComplexInnerProductLayer<Dtype>(layer_param));
    layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
    layer->Forward(this->blob_bottom_vec_, this->blob_top_vec_);
    const int count = this->blob_top_->count();
    Blob<Dtype>* const top = new Blob<Dtype>();
    top->ReshapeLike(*this->blob_top_);
    caffe_copy(count, this->blob_top_->cpu_data(), top->mutable_cpu_data());
    this->blob_top_vec_.clear();
    this->blob_top_vec_.push_back(new Blob<Dtype>());
    inner_product_param->set_transpose(true);
    shared_ptr<ComplexInnerProductLayer<Dtype> > ip_t(
        new ComplexInnerProductLayer<Dtype>(layer_param));
    ip_t->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
    const int count_w = layer->blobs()[0]->count();
    EXPECT_EQ(count_w, ip_t->blobs()[0]->count());
    // manually copy and transpose the weights from 1st IP layer into 2nd
    const Dtype* w = layer->blobs()[0]->cpu_data();
    Dtype* w_t = ip_t->blobs()[0]->mutable_cpu_data();
    const int width = layer->blobs()[0]->shape(1);
    const int width_t = ip_t->blobs()[0]->shape(1);
    for (int i = 0; i < count_w/2; ++i) {
      int r = i / width;
      int c = i % width;
      // copy while transposing
      w_t[2*(c*width_t+r)] = w[2*(r*width+c)];
      w_t[2*(c*width_t+r)+1] = w[2*(r*width+c)+1];
    }
    // copy bias from 1st IP layer to 2nd IP layer
    ASSERT_EQ(layer->blobs()[1]->count(), ip_t->blobs()[1]->count());
    caffe_copy(layer->blobs()[1]->count(), layer->blobs()[1]->cpu_data(),
        ip_t->blobs()[1]->mutable_cpu_data());
    ip_t->Forward(this->blob_bottom_vec_, this->blob_top_vec_);
    EXPECT_EQ(count, this->blob_top_->count())
        << "Invalid count for top blob for IP with transpose.";
    Blob<Dtype>* const top_t = new Blob<Dtype>();\
    top_t->ReshapeLike(*this->blob_top_vec_[0]);
    caffe_copy(count,
      this->blob_top_vec_[0]->cpu_data(),
      top_t->mutable_cpu_data());
    const Dtype* data = top->cpu_data();
    const Dtype* data_t = top_t->cpu_data();

    Dtype maxVal = 0;
    for (int i = 0; i < count; ++i) {
      if(std::abs(data[i]) > maxVal) {
        maxVal = std::abs(data[i]);
      }
      if(std::abs(data_t[i]) > maxVal) {
        maxVal = std::abs(data_t[i]);
      }
    }

    for (int i = 0; i < count; ++i) {
//      EXPECT_FLOAT_EQ(data[i], data_t[i]);
      EXPECT_NEAR(data[i], data_t[i], 1e-4*maxVal);
    }
  } else {
    LOG(ERROR) << "Skipping test due to old architecture.";
  }
}

//TYPED_TEST(ComplexInnerProductLayerTest, TestGradient) {
//  typedef typename TypeParam::Dtype Dtype;
//  this->blob_bottom_vec_.push_back(this->blob_bottom_);
//  bool IS_VALID_CUDA = false;
//#ifndef CPU_ONLY
//  IS_VALID_CUDA = CAFFE_TEST_CUDA_PROP.major >= 2;
//#endif
//  if (Caffe::mode() == Caffe::CPU ||
//      sizeof(Dtype) == 4 || IS_VALID_CUDA) {
//    LayerParameter layer_param;
//    InnerProductParameter* inner_product_param =
//        layer_param.mutable_inner_product_param();
//    inner_product_param->set_num_output(10);
//    inner_product_param->mutable_weight_filler()->set_type("gaussian");
//    inner_product_param->mutable_bias_filler()->set_type("gaussian");
//    inner_product_param->mutable_bias_filler()->set_min(1);
//    inner_product_param->mutable_bias_filler()->set_max(2);
//    ComplexInnerProductLayer<Dtype> layer(layer_param);
//    GradientChecker<Dtype> checker(1e-2, 1e-3);
//    checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
//        this->blob_top_vec_);
//  } else {
//    LOG(ERROR) << "Skipping test due to old architecture.";
//  }
//  ZZZ
//}
//
//TYPED_TEST(ComplexInnerProductLayerTest, TestGradientTranspose) {
//  typedef typename TypeParam::Dtype Dtype;
//  this->blob_bottom_vec_.push_back(this->blob_bottom_);
//  bool IS_VALID_CUDA = false;
//#ifndef CPU_ONLY
//  IS_VALID_CUDA = CAFFE_TEST_CUDA_PROP.major >= 2;
//#endif
//  if (Caffe::mode() == Caffe::CPU ||
//      sizeof(Dtype) == 4 || IS_VALID_CUDA) {
//    LayerParameter layer_param;
//    InnerProductParameter* inner_product_param =
//        layer_param.mutable_inner_product_param();
//    inner_product_param->set_num_output(11);
//    inner_product_param->mutable_weight_filler()->set_type("gaussian");
//    inner_product_param->mutable_bias_filler()->set_type("gaussian");
//    inner_product_param->mutable_bias_filler()->set_min(1);
//    inner_product_param->mutable_bias_filler()->set_max(2);
//    inner_product_param->set_transpose(true);
//    ComplexInnerProductLayer<Dtype> layer(layer_param);
//    GradientChecker<Dtype> checker(1e-2, 1e-3);
//    checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
//        this->blob_top_vec_);
//  } else {
//    LOG(ERROR) << "Skipping test due to old architecture.";
//  }
//}
//
//TYPED_TEST(ComplexInnerProductLayerTest, TestBackwardTranspose) {
//  typedef typename TypeParam::Dtype Dtype;
//  this->blob_bottom_vec_.push_back(this->blob_bottom_);
//  bool IS_VALID_CUDA = false;
//#ifndef CPU_ONLY
//  IS_VALID_CUDA = CAFFE_TEST_CUDA_PROP.major >= 2;
//#endif
//  if (Caffe::mode() == Caffe::CPU ||
//      sizeof(Dtype) == 4 || IS_VALID_CUDA) {
//    LayerParameter layer_param;
//    InnerProductParameter* inner_product_param =
//        layer_param.mutable_inner_product_param();
//    inner_product_param->set_num_output(10);
//    inner_product_param->mutable_weight_filler()->set_type("uniform");
//    inner_product_param->mutable_bias_filler()->set_type("uniform");
//    inner_product_param->mutable_bias_filler()->set_min(1);
//    inner_product_param->mutable_bias_filler()->set_max(2);
//    inner_product_param->set_transpose(false);
//    shared_ptr<ComplexInnerProductLayer<Dtype> > layer(
//        new ComplexInnerProductLayer<Dtype>(layer_param));
//    layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
//    layer->Forward(this->blob_bottom_vec_, this->blob_top_vec_);
//    // copy top blob
//    Blob<Dtype>* const top = new Blob<Dtype>();
//    top->CopyFrom(*this->blob_top_, false, true);
//    // fake top diff
//    Blob<Dtype>* const diff = new Blob<Dtype>();
//    diff->ReshapeLike(*this->blob_top_);
//    {
//      FillerParameter filler_param;
//      UniformFiller<Dtype> filler(filler_param);
//      filler.Fill(diff);
//    }
//    caffe_copy(this->blob_top_vec_[0]->count(),
//      diff->cpu_data(),
//      this->blob_top_vec_[0]->mutable_cpu_diff());
//    vector<bool> propagate_down(1, true);
//    layer->Backward(this->blob_top_vec_,
//        propagate_down,
//        this->blob_bottom_vec_);
//    // copy first ip's weights and their diffs
//    Blob<Dtype>* const w = new Blob<Dtype>();
//    w->CopyFrom(*layer->blobs()[0], false, true);
//    w->CopyFrom(*layer->blobs()[0], true, true);
//    // copy bottom diffs
//    Blob<Dtype>* const bottom_diff = new Blob<Dtype>();
//    bottom_diff->CopyFrom(*this->blob_bottom_vec_[0], true, true);
//    // repeat original top with tranposed ip
//    this->blob_top_vec_.clear();
//    this->blob_top_vec_.push_back(new Blob<Dtype>());
//    inner_product_param->set_transpose(true);
//    shared_ptr<ComplexInnerProductLayer<Dtype> > ip_t(
//        new ComplexInnerProductLayer<Dtype>(layer_param));
//    ip_t->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
//    // manually copy and transpose the weights from 1st IP layer into 2nd
//    {
//      const Dtype* w_src = w->cpu_data();
//      Dtype* w_t = ip_t->blobs()[0]->mutable_cpu_data();
//      const int width = layer->blobs()[0]->shape(1);
//      const int width_t = ip_t->blobs()[0]->shape(1);
//      for (int i = 0; i < layer->blobs()[0]->count(); ++i) {
//        int r = i / width;
//        int c = i % width;
//        w_t[c*width_t+r] = w_src[r*width+c];  // copy while transposing
//      }
//      // copy bias from 1st IP layer to 2nd IP layer
//      ASSERT_EQ(layer->blobs()[1]->count(), ip_t->blobs()[1]->count());
//      caffe_copy(layer->blobs()[1]->count(), layer->blobs()[1]->cpu_data(),
//          ip_t->blobs()[1]->mutable_cpu_data());
//    }
//    ip_t->Forward(this->blob_bottom_vec_, this->blob_top_vec_);
//    caffe_copy(this->blob_top_vec_[0]->count(),
//      diff->cpu_data(),
//      this->blob_top_vec_[0]->mutable_cpu_diff());
//    ip_t->Backward(this->blob_top_vec_, propagate_down, this->blob_bottom_vec_);
//    const Dtype* data = w->cpu_diff();
//    const Dtype* data_t = ip_t->blobs()[0]->cpu_diff();
//    const int WIDTH = layer->blobs()[0]->shape(1);
//    const int WIDTH_T = ip_t->blobs()[0]->shape(1);
//    for (int i = 0; i < layer->blobs()[0]->count(); ++i) {
//      int r = i / WIDTH;
//      int c = i % WIDTH;
//      EXPECT_NE(Dtype(0.), data[r*WIDTH+c]);
//      EXPECT_FLOAT_EQ(data[r*WIDTH+c], data_t[c*WIDTH_T+r]);
//    }
//    data = bottom_diff->cpu_diff();
//    data_t = this->blob_bottom_vec_[0]->cpu_diff();
//    for (int i = 0; i < this->blob_bottom_vec_[0]->count(); ++i) {
//      EXPECT_NE(Dtype(0.), data[i]);
//      EXPECT_FLOAT_EQ(data[i], data_t[i]);
//    }
//  } else {
//    LOG(ERROR) << "Skipping test due to old architecture.";
//  }
//}

TYPED_TEST(ComplexInnerProductLayerTest, TestForwardManual) {
  typedef typename TypeParam::Dtype Dtype;
  this->blob_bottom_vec_.push_back(this->blob_bottom_);
  bool IS_VALID_CUDA = false;
#ifndef CPU_ONLY
  IS_VALID_CUDA = CAFFE_TEST_CUDA_PROP.major >= 2;
#endif
  if (Caffe::mode() == Caffe::CPU ||
      sizeof(Dtype) == 4 || IS_VALID_CUDA) {
    LayerParameter layer_param;
    InnerProductParameter* inner_product_param =
        layer_param.mutable_inner_product_param();
    int B = this->blob_bottom_->shape(0);
    int C_in = this->blob_bottom_->shape(1);
    int H = this->blob_bottom_->shape(2);
    int W = this->blob_bottom_->shape(3);
    int K = C_in*H*W;
    int C_out = 10;
//    int C_out = 1;
    inner_product_param->set_num_output(C_out);
    inner_product_param->mutable_weight_filler()->set_type("uniform");
    inner_product_param->mutable_bias_filler()->set_type("uniform");
    inner_product_param->mutable_bias_filler()->set_min(1);
    inner_product_param->mutable_bias_filler()->set_max(2);
    inner_product_param->set_transpose(false);
    shared_ptr<ComplexInnerProductLayer<Dtype> > layer(
        new ComplexInnerProductLayer<Dtype>(layer_param));
    layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
    layer->Forward(this->blob_bottom_vec_, this->blob_top_vec_);
    const int count = this->blob_top_->count();
    Blob<Dtype>* const top_expected = new Blob<Dtype>();
    top_expected->ReshapeLike(*this->blob_top_);
    std::complex<Dtype>* top_expected_data = ComplexLayer<Dtype>::RealToComplex_mutable_cpu(top_expected->mutable_cpu_data());

    const std::complex<Dtype>* bottom_data = ComplexLayer<Dtype>::RealToComplex_cpu(this->blob_bottom_->cpu_data());
    const std::complex<Dtype>* weight_data = ComplexLayer<Dtype>::RealToComplex_cpu(layer->blobs()[0]->cpu_data());
    const std::complex<Dtype>* bias_data = ComplexLayer<Dtype>::RealToComplex_cpu(layer->blobs()[1]->cpu_data());

    size_t top_expected_index = 0;
    for(int b = 0; b < B; ++b) {
      size_t weight_index = 0;
      size_t bias_index = 0;
      for(int c_out = 0; c_out < C_out; ++c_out, ++bias_index, ++top_expected_index) {
        size_t bottom_index = b*K;
        std::complex<Dtype> value = bias_data[bias_index];
//        printf("value = %lf + i%lf\n", std::real(value), std::imag(value));
        for(int k = 0; k < K; ++k, ++bottom_index, ++weight_index) {
//          printf("bottom[%ld] = %lf + i%lf, weight[%ld] = %lf + i %lf\n",
//        		  bottom_index, std::real(bottom_data[bottom_index]), std::imag(bottom_data[bottom_index]),
//				  weight_index, std::real(weight_data[weight_index]), std::imag(weight_data[weight_index]));
          value += bottom_data[bottom_index]*weight_data[weight_index];
//          printf("value = %lf + i%lf\n", std::real(value), std::imag(value));
        }
        top_expected_data[top_expected_index] = value;
      }
    }

    const Dtype* top_actual_2ch_data = this->blob_top_->cpu_data();
    const Dtype* top_expected_2ch_data = top_expected->cpu_data();

    Dtype maxVal = 0;
    for (int i = 0; i < count/2; ++i) {
      if(std::abs(top_actual_2ch_data[2*i]) > maxVal) {
        maxVal = std::abs(top_actual_2ch_data[2*i]);
      }
      if(std::abs(top_actual_2ch_data[2*i+1]) > maxVal) {
        maxVal = std::abs(top_actual_2ch_data[2*i+1]);
      }
    }

    for (int i = 0; i < count/2; ++i) {
//      EXPECT_FLOAT_EQ(std::real(top_expected_data[i]), top_expected_2ch_data[2*i]);
//      EXPECT_FLOAT_EQ(std::imag(top_expected_data[i]), top_expected_2ch_data[2*i+1]);
//
//      EXPECT_FLOAT_EQ(top_actual_2ch_data[2*i], top_expected_2ch_data[2*i]);
//      EXPECT_FLOAT_EQ(top_actual_2ch_data[2*i+1], top_expected_2ch_data[2*i+1]);

      EXPECT_NEAR(std::real(top_expected_data[i]), top_expected_2ch_data[2*i], 1e-5*maxVal);
      EXPECT_NEAR(std::imag(top_expected_data[i]), top_expected_2ch_data[2*i+1], 1e-5*maxVal);

      EXPECT_NEAR(top_actual_2ch_data[2*i], top_expected_2ch_data[2*i], 1e-5*maxVal);
      EXPECT_NEAR(top_actual_2ch_data[2*i+1], top_expected_2ch_data[2*i+1], 1e-5*maxVal);

    }
  } else {
    LOG(ERROR) << "Skipping test due to old architecture.";
  }
}

TYPED_TEST(ComplexInnerProductLayerTest, TestBackwardManual) {
  typedef typename TypeParam::Dtype Dtype;
  this->blob_bottom_vec_.push_back(this->blob_bottom_);
  bool IS_VALID_CUDA = false;
#ifndef CPU_ONLY
  IS_VALID_CUDA = CAFFE_TEST_CUDA_PROP.major >= 2;
#endif
  if (Caffe::mode() == Caffe::CPU ||
      sizeof(Dtype) == 4 || IS_VALID_CUDA) {
    LayerParameter layer_param;
    InnerProductParameter* inner_product_param =
        layer_param.mutable_inner_product_param();
    int B = this->blob_bottom_->shape(0);
    int C_in = this->blob_bottom_->shape(1);
    int H = this->blob_bottom_->shape(2);
    int W = this->blob_bottom_->shape(3);
    int K = C_in*H*W;
    int C_out = 10;
//    int C_out = 1;
    inner_product_param->set_num_output(C_out);
    inner_product_param->mutable_weight_filler()->set_type("uniform");
    inner_product_param->mutable_bias_filler()->set_type("uniform");
    inner_product_param->mutable_bias_filler()->set_min(1);
    inner_product_param->mutable_bias_filler()->set_max(2);
    inner_product_param->set_transpose(false);
    shared_ptr<ComplexInnerProductLayer<Dtype> > layer(
        new ComplexInnerProductLayer<Dtype>(layer_param));
    layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
    layer->Forward(this->blob_bottom_vec_, this->blob_top_vec_);

    // fake top diff
    Blob<Dtype>* const diff = new Blob<Dtype>();
    diff->ReshapeLike(*this->blob_top_);
    {
      FillerParameter filler_param;
      UniformFiller<Dtype> filler(filler_param);
      filler.Fill(diff);
    }
    caffe_copy(this->blob_top_vec_[0]->count(),
        diff->cpu_data(),
        this->blob_top_vec_[0]->mutable_cpu_diff());

    vector<bool> propagate_down(1, true);

    layer->Backward(this->blob_top_vec_, propagate_down, this->blob_bottom_vec_);

    const std::complex<Dtype>* bottom_data = ComplexLayer<Dtype>::RealToComplex_cpu(this->blob_bottom_->cpu_data());
    const std::complex<Dtype>* weight_data = ComplexLayer<Dtype>::RealToComplex_cpu(layer->blobs()[0]->cpu_data());
    const std::complex<Dtype>* top_diff = ComplexLayer<Dtype>::RealToComplex_cpu(this->blob_top_->cpu_diff());
    const std::complex<Dtype>* bottom_diff_actual = ComplexLayer<Dtype>::RealToComplex_cpu(this->blob_bottom_->cpu_diff());
    const std::complex<Dtype>* weight_diff_actual = ComplexLayer<Dtype>::RealToComplex_cpu(layer->blobs()[0]->cpu_diff());
    const std::complex<Dtype>* bias_diff_actual = ComplexLayer<Dtype>::RealToComplex_cpu(layer->blobs()[1]->cpu_diff());

    // Check bottom diff

    std::complex<Dtype>* bottom_diff_expected = new std::complex<Dtype>[B*K];
    size_t bottom_diff_expected_index = 0;
    for(int b = 0; b < B; ++b) {
      for(int k = 0; k < K; ++k, ++bottom_diff_expected_index) {
        size_t top_diff_index = b*C_out;
        size_t weight_index = k;
        bottom_diff_expected[bottom_diff_expected_index] = std::complex<Dtype>(0);
        for(int c_out = 0; c_out < C_out; ++c_out, ++top_diff_index, weight_index+=K) {
          bottom_diff_expected[bottom_diff_expected_index] += top_diff[top_diff_index]*std::conj(weight_data[weight_index]);
        }
      }
    }

    Dtype maxVal = 0;
    for(int i = 0; i < B*K; ++i) {
      if(std::abs(bottom_diff_actual[i]) > maxVal) {
        maxVal = std::abs(bottom_diff_actual[i]);
      }
    }

    for (int i = 0; i < B*K; ++i) {
      EXPECT_NEAR(std::real(bottom_diff_expected[i]), std::real(bottom_diff_actual[i]), 1e-5*maxVal);
      EXPECT_NEAR(std::imag(bottom_diff_expected[i]), std::imag(bottom_diff_actual[i]), 1e-5*maxVal);
    }

    // Check weight diff

//    printf("C_out=%d, K=%d\n", C_out, K);
    std::complex<Dtype>* weight_diff_expected = new std::complex<Dtype>[C_out*K];
    size_t weight_diff_expected_index = 0;
    for(int c_out = 0; c_out < C_out; ++c_out) {
      for(int k = 0; k < K; ++k, ++weight_diff_expected_index) {
        size_t top_diff_index = c_out;
        size_t bottom_index = k;
        weight_diff_expected[weight_diff_expected_index] = std::complex<Dtype>(0);
//        printf("weight_diff[%ld] = %lf + i%lf\n", weight_diff_expected_index,
//        		std::real(weight_diff_expected[weight_diff_expected_index]),
//				std::imag(weight_diff_expected[weight_diff_expected_index]));
        for(int b = 0; b < B; ++b, top_diff_index+=C_out, bottom_index+=K) {
//          printf("top_diff[%ld] * conj(bottom[%ld]) = (%lf + 1i*%lf) * conj(%lf + 1i*%lf)\n",
//				  top_diff_index, bottom_index,
//				  std::real(top_diff[top_diff_index]), std::imag(top_diff[top_diff_index]),
//        		  std::real(bottom_data[bottom_index]), std::imag(bottom_data[bottom_index]));
          weight_diff_expected[weight_diff_expected_index] += top_diff[top_diff_index]*std::conj(bottom_data[bottom_index]);
//          printf("weight_diff[%ld] = %lf + i%lf\n", weight_diff_expected_index,
//          		std::real(weight_diff_expected[weight_diff_expected_index]),
//  				std::imag(weight_diff_expected[weight_diff_expected_index]));
        }
      }
    }

    maxVal = 0;
    for(int i = 0; i < C_out*K; ++i) {
      if(std::abs(weight_diff_actual[i]) > maxVal) {
        maxVal = std::abs(weight_diff_actual[i]);
      }
    }

    for (int i = 0; i < C_out*K; ++i) {
      EXPECT_NEAR(std::real(weight_diff_expected[i]), std::real(weight_diff_actual[i]), 1e-5*maxVal);
      EXPECT_NEAR(std::imag(weight_diff_expected[i]), std::imag(weight_diff_actual[i]), 1e-5*maxVal);
    }

    // Check bias diff

    std::complex<Dtype>* bias_diff_expected = new std::complex<Dtype>[B*K];
    size_t bias_diff_expected_index = 0;
    for(int c_out = 0; c_out < C_out; ++c_out, ++bias_diff_expected_index) {
      size_t top_diff_index = c_out;
      bias_diff_expected[bias_diff_expected_index] = std::complex<Dtype>(0);
//	  printf("bias_diff[%ld] = %lf + i%lf\n", bias_diff_expected_index,
//			std::real(bias_diff_expected[bias_diff_expected_index]),
//			std::imag(bias_diff_expected[bias_diff_expected_index]));
      for(int b = 0; b < B; ++b, top_diff_index+=C_out) {
//     	printf("top_diff[%ld] = %lf + i%lf\n", top_diff_index,
//     			std::real(top_diff[top_diff_index]),
//				std::imag(top_diff[top_diff_index]));
        bias_diff_expected[bias_diff_expected_index] += top_diff[top_diff_index];
//        printf("bias_diff[%ld] = %lf + i%lf\n", bias_diff_expected_index,
//        		std::real(bias_diff_expected[bias_diff_expected_index]),
//				std::imag(bias_diff_expected[bias_diff_expected_index]));
      }
    }

    maxVal = 0;
    for(int i = 0; i < C_out; ++i) {
      if(std::abs(bias_diff_actual[i]) > maxVal) {
        maxVal = std::abs(bias_diff_actual[i]);
      }
    }

    for (int i = 0; i < C_out; ++i) {
      EXPECT_NEAR(std::real(bias_diff_expected[i]), std::real(bias_diff_actual[i]), 1e-5*maxVal);
      EXPECT_NEAR(std::imag(bias_diff_expected[i]), std::imag(bias_diff_actual[i]), 1e-5*maxVal);
    }

    delete[] bottom_diff_expected;
    delete[] weight_diff_expected;
    delete[] bias_diff_expected;
  } else {
    LOG(ERROR) << "Skipping test due to old architecture.";
  }
}

// TODO TestBackwardManualTranspose


}  // namespace caffe
