#include <vector>

#include "caffe/layers/complex_dropout_layer.hpp"
#include "caffe/util/math_functions.hpp"


namespace caffe {
    
template <typename Dtype>
void ComplexDropoutLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                                     const vector<Blob<Dtype>*>& top) {
    ComplexLayer<Dtype>::LayerSetUp(bottom, top);
    threshold_ = this->layer_param_.dropout_param().dropout_ratio();
    DCHECK(threshold_ > 0.);
    DCHECK(threshold_ < 1.);
    scale_ = 1. / (1. - threshold_);
    uint_thres_ = static_cast<unsigned int>(UINT_MAX * threshold_);
}


template <typename Dtype>
void ComplexDropoutLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  CHECK_EQ(2, bottom[0]->shape(-1)); //make sure data is complex
                
//complex dropout reshape
    // Set up the cache for random number generation
    // ReshapeLike does not work because rand_vec_ is of Dtype uint
    rand_vec_.Reshape(bottom[0]->shape()); //this should stay the same because we're just matching shapes!
}
    
//drop one or drop both, but not just one (real or complex)
template <typename Dtype>
void ComplexDropoutLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
    //goals:
        //1. grab bottom, top, count, shape
        //2. for each datum, zero both complex and real or don't (in training, else identity)
        //3. pass this data to top
    
    //grab bottom and top
  const std::complex<Dtype>* bottom_data = this->RealToComplexBottomData_cpu(bottom, 0);
  std::complex<Dtype>* top_data = this->RealToComplexTopData_mutable_cpu(top,0);
    //grab the mask
  unsigned int* mask =  rand_vec_.mutable_cpu_data();
    //(mask must be complex so it masks both real and imaginary portions to 0)    
  const int count = top[0]->count()/2; //don't count twice due to two channel complex representation
    //dropout logic
    if (this->phase_ == TRAIN) {
        // Create random numbers
        caffe_rng_bernoulli(count, 1. - threshold_, mask);
        //iterate over reals
        for (int i = 0; i < count; ++i) {
            //grab real and complex pairs...
            top_data[i] = bottom_data[i] * std::complex<Dtype>(mask[i]) * std::complex<Dtype>(scale_);
                //bottom complex; mask complex; scale complex (must all be complex)
        }
        //if in test, perform identity
    } else {
        caffe_copy(bottom[0]->count(), bottom_data, top_data);
    }

    //include to sync float array with complex float array (in case we made a copy)
  this->SyncComplexTopData_cpu(top, 0);
}

    
//backprop dropout (keep 1s, drop 0s)
template <typename Dtype>
void ComplexDropoutLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
    
    //goals:
        //1. grab bottom_data, top_diff, count, shape
        //2. for each datum, zero both complex and real or don't (in training, else identity)
    
  unsigned int* mask =  rand_vec_.mutable_cpu_data();


  if (propagate_down[0]) { //if still propagating back
    const std::complex<Dtype>* top_diff = this->RealToComplexTopDiff_cpu(top,0);
    std::complex<Dtype>* bottom_diff = this->RealToComplexBottomDiff_mutable_cpu(bottom, 0);
      
      if (this->phase_ == TRAIN) {
          const int count = top[0]->count()/2;
          
          for (int i = 0; i < count; ++i) {
              bottom_diff[i] = top_diff[i] * std::complex<Dtype>(mask[i]) * std::complex<Dtype>(scale_);
          }
          
      } else {
          caffe_copy(top[0]->count(), top_diff, bottom_diff);
      }
      
      //include to sync float array with complex float array (in case we made a copy)
      this->SyncComplexBlobDiff_cpu(0);

  }
}


#ifdef CPU_ONLY
STUB_GPU(ComplexDropoutLayer);
#endif

INSTANTIATE_CLASS(ComplexDropoutLayer);
REGISTER_LAYER_CLASS(ComplexDropout);

}  // namespace caffe
