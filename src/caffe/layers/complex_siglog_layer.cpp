#include <vector>
#include "caffe/layers/complex_siglog_layer.hpp"

namespace caffe {

    
template <typename Dtype>
void ComplexSiglogLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  CHECK_EQ(2, bottom[0]->shape(-1));

  vector<int> shape = bottom[0]->shape();
  top[0]->Reshape(shape);
}
    
/**
     Siglog forward pass:
     output = 
        \frac{sz}{c+ (1/r) |sz|^d}
**/

template <typename Dtype>
void ComplexSiglogLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const std::complex<Dtype>* bottom_data = this->RealToComplexBottomData_cpu(bottom, 0);
  std::complex<Dtype>* top_data = this->RealToComplexTopData_mutable_cpu(top,0);

//define the constants as fixed for now at 1:
  const Dtype r = 1.0;
  const Dtype c = 1.0;
  const Dtype s = 1.0;
  const Dtype d = 1.0;

//top_data[i] = \frac{sz}{c+ (1/r) |sz|^d}
  const int count = top[0]->count()/2;
  for (int i = 0; i < count; ++i) {
      std::complex<Dtype> z = bottom_data[i];
      std::complex<Dtype> zs_to_d = pow(std::arg(z * std::complex<Dtype>(s)), d);
      top_data[i] = (z*std::complex<Dtype>(s)) / (c + zs_to_d * std::complex<Dtype>(1.0/r));
  }

  this->SyncComplexTopData_cpu(top, 0);
}

/**
     Siglog backward pass:
     output = 
        dl_/dz_[\frac{s|s|^d z^2 d |z|^{d-1}}{2|z|r (c + (1/r) |sz|^d)^2}]
            + dl/dz_[\frac{(c+(1/r)|sz|^d)s - s(d\2r |s|^d |z|^d)}{c+ 1/r(|sz|^d)^2}]
**/
template <typename Dtype>
void ComplexSiglogLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[0]) {
    const std::complex<Dtype>* top_diff = this->RealToComplexTopDiff_cpu(top,0);
    const std::complex<Dtype>* bottom_data = this->RealToComplexBottomData_cpu(bottom, 0);
    std::complex<Dtype>* bottom_diff = this->RealToComplexBottomDiff_mutable_cpu(bottom, 0);
      
      //define the constants as fixed for now at 1:
      const Dtype r = 1.0;
      const Dtype c = 1.0;
      const Dtype s = 1.0;
      const Dtype d = 1.0;
      
    const int count = top[0]->count()/2;
    for (int i = 0; i < count; ++i) {
      std::complex<Dtype> z = bottom_data[i];

        /**   dfdz:
         = ((c + (1/r) * pow(std::arg(std::complex<Dtype>(s) * z), d)*s - (s) * ((d/(2*r)) * pow(std::arg(std::complex<Dtype>(s)), d) * pow(std::arg(z), d)))) / (pow(c + (1/r) * pow(std::arg(std::complex<Dtype>(s)*z), d) ,2));
        **/

        std::complex<Dtype> numerator_dfdz = (c + (1/r) * pow(std::arg(z * std::complex<Dtype>(s)), d)*s- (s) * ((d/(2*r)) * pow(std::arg(std::complex<Dtype>(s)), d) * pow(std::arg(z), d)));
        std::complex<Dtype> denominator_dfdz = pow(c + (1/r) * pow(std::arg(z * std::complex<Dtype>(s)), d) ,2);
        std::complex<Dtype> dfdz = numerator_dfdz / denominator_dfdz;
        
        
        /**   dfdcz:
         = (s * pow(std::arg(std::complex<Dtype>(s)), d) * pow(z,2) * d * pow(std::arg(z), d-1))
             /(2 * std::arg(z) * r * pow((c + (1/r) * pow(std::arg(std::complex<Dtype>(s) * z), d)), 2));
         **/
             
        std::complex<Dtype> s_to_d = pow(std::arg(std::complex<Dtype>(s)), d);
        std::complex<Dtype> zs_to_d = pow(std::arg(z * std::complex<Dtype>(s)), d);
        
        std::complex<Dtype> numerator_dfdzc= std::complex<Dtype>(s)* std::complex<Dtype>(d) * std::complex<Dtype>(pow(z,2)) * s_to_d * std::complex<Dtype>(pow(std::arg(z), d-1));
        
        std::complex<Dtype> denominator_dfdzc= pow((c + zs_to_d * std::complex<Dtype>((1.0/r))),2) * std::complex<Dtype>(2.0 * r) *  std::arg(z);
        
        std::complex<Dtype> dfdcz = numerator_dfdzc/denominator_dfdzc;


        
        //LOSS GRADIENT FROM COMPONENTS
      bottom_diff[i] = std::conj(top_diff[i])*dfdcz + top_diff[i]*std::conj(dfdz);
    }

    this->SyncComplexBlobDiff_cpu(0);
  }
}


#ifdef CPU_ONLY
STUB_GPU(ComplexSiglogLayer);
#endif

INSTANTIATE_CLASS(ComplexSiglogLayer);
REGISTER_LAYER_CLASS(ComplexSiglog);

}  // namespace caffe
