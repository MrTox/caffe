#include <vector>
#include "caffe/layers/complex_separable_sigmoid_layer.hpp"

namespace caffe {

    
template <typename Dtype>
void ComplexSeparableSigmoidLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  CHECK_EQ(2, bottom[0]->shape(-1));

  vector<int> shape = bottom[0]->shape();
  top[0]->Reshape(shape);
}

template <typename Dtype>
void ComplexSeparableSigmoidLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const std::complex<Dtype>* bottom_data = this->RealToComplexBottomData_cpu(bottom, 0);
  std::complex<Dtype>* top_data = this->RealToComplexTopData_mutable_cpu(top,0);

    
      const int count = top[0]->count()/2;
      for (int i = 0; i < count; ++i) {
          std::complex<Dtype> z = bottom_data[i];
          
          std::complex<Dtype> one = std::complex<Dtype>(1.0);
          
          std::complex<Dtype> im = std::imag(z);
          std::complex<Dtype> re = std::real(z);
          std::complex<Dtype> rez = one / (one + exp(-re));
          std::complex<Dtype> imz = one / (one + exp(-im));
          std::complex<Dtype> f = rez + std::complex<Dtype>(0,1) * imz;
          top_data[i] = f;
      }
    

  this->SyncComplexTopData_cpu(top, 0);
}
    

template <typename Dtype>
void ComplexSeparableSigmoidLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[0]) {
    const std::complex<Dtype>* top_diff = this->RealToComplexTopDiff_cpu(top,0);
    const std::complex<Dtype>* bottom_data = this->RealToComplexBottomData_cpu(bottom, 0);
    std::complex<Dtype>* bottom_diff = this->RealToComplexBottomDiff_mutable_cpu(bottom, 0);
      
      
    const int count = top[0]->count()/2;
    for (int i = 0; i < count; ++i) {
      std::complex<Dtype> z = bottom_data[i];
        
        std::complex<Dtype> one = std::complex<Dtype>(1.0);


        std::complex<Dtype> im = std::imag(z);
        std::complex<Dtype> re = std::real(z);
        std::complex<Dtype> rez = exp(-re) / pow((one + exp(-re)),2);
        std::complex<Dtype> imz = exp(-im) / pow((one + exp(-im)),2);
        std::complex<Dtype> dfdz = std::complex<Dtype>(0.5) * (rez + imz);
        
        std::complex<Dtype> dfdcz = std::complex<Dtype>(0.5) * (rez - imz);

        
        //LOSS GRADIENT FROM COMPONENTS
      bottom_diff[i] = std::conj(top_diff[i])*dfdcz + top_diff[i]*std::conj(dfdz);
    }

    this->SyncComplexBottomDiff_cpu(bottom, 0);
  }
}


#ifdef CPU_ONLY
STUB_GPU(ComplexSeparableSigmoidLayer);
#endif

INSTANTIATE_CLASS(ComplexSeparableSigmoidLayer);
REGISTER_LAYER_CLASS(ComplexSeparableSigmoid);

}  // namespace caffe
