#include <climits>
#include <vector>

#include "caffe/complex_blob.hpp"
#include "caffe/common.hpp"
#include "caffe/syncedmem.hpp"
#include "caffe/util/math_functions.hpp"

using std::complex;

namespace caffe {

template <typename Dtype>
void ComplexBlob<Dtype>::Reshape(const vector<int>& shape) {
  CHECK_LE(shape.size(), kMaxBlobAxes);
  count_ = 1;
  shape_.resize(shape.size());
  if (!shape_data_ || shape_data_->size() < shape.size() * sizeof(int)) {
    shape_data_.reset(new SyncedMemory(shape.size() * sizeof(int)));
  }
  int* shape_data = static_cast<int*>(shape_data_->mutable_cpu_data());
  for (int i = 0; i < shape.size(); ++i) {
    CHECK_GE(shape[i], 0);
    CHECK_LE(shape[i], INT_MAX / count_) << "blob size exceeds INT_MAX";
    count_ *= shape[i];
    shape_[i] = shape[i];
    shape_data[i] = shape[i];
  }
  if (count_ > capacity_) {
    capacity_ = count_;
    data_.reset(new SyncedMemory(capacity_ * sizeof(complex<Dtype>)));
    diff_.reset(new SyncedMemory(capacity_ * sizeof(complex<Dtype>)));
    conj_diff_.reset(new SyncedMemory(capacity_ * sizeof(complex<Dtype>)));
  }
}

template <typename Dtype>
const complex<Dtype>* ComplexBlob<Dtype>::cpu_conj_diff() const {
  CHECK(conj_diff_);
  return (const complex<Dtype>*)conj_diff_->cpu_data();
}

template <typename Dtype>
const complex<Dtype>* ComplexBlob<Dtype>::gpu_conj_diff() const {
  CHECK(conj_diff_);
  return (const complex<Dtype>*)conj_diff_->gpu_data();
}

template <typename Dtype>
complex<Dtype>* ComplexBlob<Dtype>::mutable_cpu_conj_diff() {
  CHECK(conj_diff_);
  return static_cast<complex<Dtype>*>(conj_diff_->mutable_cpu_data());
}

template <typename Dtype>
complex<Dtype>* ComplexBlob<Dtype>::mutable_gpu_conj_diff() {
  CHECK(conj_diff_);
  return static_cast<complex<Dtype>*>(conj_diff_->mutable_gpu_data());
}

template <typename Dtype>
void ComplexBlob<Dtype>::ShareData(const ComplexBlob& other) {
  CHECK_EQ(count_, other.count());
  data_ = other.data();
}

template <typename Dtype>
void ComplexBlob<Dtype>::ShareDiff(const ComplexBlob& other) {
  CHECK_EQ(count_, other.count());
  diff_ = other.diff();
}

template <typename Dtype>
void ComplexBlob<Dtype>::ShareConjDiff(const ComplexBlob& other) {
  CHECK_EQ(count_, other.count());
  conj_diff_ = other.conj_diff();
}

// The "update" method is used for parameter blobs in a Net, which are stored
// as ComplexBlob<float> or ComplexBlob<double> -- hence we do not define it for
// ComplexBlob<int> or ComplexBlob<unsigned int>.
template <> void ComplexBlob<unsigned int>::Update() { NOT_IMPLEMENTED; }
template <> void ComplexBlob<int>::Update() { NOT_IMPLEMENTED; }

template <typename Dtype>
void ComplexBlob<Dtype>::Update() {
  // We will perform update based on where the data is located.
  switch (data_->head()) {
  case SyncedMemory::HEAD_AT_CPU:
    // perform computation on CPU
    //TODO conj_diff
    caffe_axpy<complex<Dtype> >(count_, Dtype(-1),
        static_cast<const complex<Dtype>*>(diff_->cpu_data()),
        static_cast<complex<Dtype>*>(data_->mutable_cpu_data()));
    break;
  case SyncedMemory::HEAD_AT_GPU:
  case SyncedMemory::SYNCED:
#ifndef CPU_ONLY
    // perform computation on GPU
    // TODO conj_diff
    caffe_gpu_axpy<complex<Dtype> >(count_, Dtype(-1),
        static_cast<const complex<Dtype>*>(diff_->gpu_data()),
        static_cast<complex<Dtype>*>(data_->mutable_gpu_data()));
#else
    NO_GPU;
#endif
    break;
  default:
    LOG(FATAL) << "Syncedmem not initialized.";
  }
}

template <> unsigned int ComplexBlob<unsigned int>::asum_conj_diff() const {
  NOT_IMPLEMENTED;
  return 0;
}

template <> int ComplexBlob<int>::asum_conj_diff() const {
  NOT_IMPLEMENTED;
  return 0;
}

template <typename Dtype>
Dtype ComplexBlob<Dtype>::asum_conj_diff() const {
  // TODO conj_diff
  if (!conj_diff_) { return 0; }
  switch (conj_diff_->head()) {
  case SyncedMemory::HEAD_AT_CPU:
    return caffe_cpu_asum(count_, cpu_conj_diff());
  case SyncedMemory::HEAD_AT_GPU:
  case SyncedMemory::SYNCED:
#ifndef CPU_ONLY
  {
    Dtype asum;
    caffe_gpu_asum(count_, gpu_conj_diff(), &asum);
    return asum;
  }
#else
    NO_GPU;
#endif
  case SyncedMemory::UNINITIALIZED:
    return 0;
  default:
    LOG(FATAL) << "Unknown SyncedMemory head state: " << conj_diff_->head();
  }
  return 0;
}

template <> unsigned int ComplexBlob<unsigned int>::sumsq_conj_diff() const {
  NOT_IMPLEMENTED;
  return 0;
}

template <> int ComplexBlob<int>::sumsq_conj_diff() const {
  NOT_IMPLEMENTED;
  return 0;
}

template <typename Dtype>
Dtype ComplexBlob<Dtype>::sumsq_conj_diff() const {
  // TODO diff
  Dtype sumsq;
  const complex<Dtype>* conj_diff;
  if (!conj_diff_) { return 0; }
  switch (conj_diff_->head()) {
  case SyncedMemory::HEAD_AT_CPU:
    conj_diff = cpu_conj_diff();
    sumsq = caffe_cpu_dot(count_, conj_diff, conj_diff);
    break;
  case SyncedMemory::HEAD_AT_GPU:
  case SyncedMemory::SYNCED:
#ifndef CPU_ONLY
    conj_diff = gpu_conj_diff();
    caffe_gpu_dot(count_, conj_diff, conj_diff, &sumsq);
    break;
#else
    NO_GPU;
#endif
  case SyncedMemory::UNINITIALIZED:
    return 0;
  default:
    LOG(FATAL) << "Unknown SyncedMemory head state: " << conj_diff_->head();
  }
  return sumsq;
}

template <> void ComplexBlob<unsigned int>::scale_conj_diff(complex<unsigned int> scale_factor) {
  NOT_IMPLEMENTED;
}

template <> void ComplexBlob<int>::scale_conj_diff(complex<int> scale_factor) {
  NOT_IMPLEMENTED;
}

template <typename Dtype>
void ComplexBlob<Dtype>::scale_conj_diff(complex<Dtype> scale_factor) {
  // TODO conj_diff
  complex<Dtype>* conj_diff;
  if (!conj_diff_) { return; }
  switch (conj_diff_->head()) {
  case SyncedMemory::HEAD_AT_CPU:
    conj_diff = mutable_cpu_conj_diff();
    caffe_scal(count_, scale_factor, conj_diff);
    return;
  case SyncedMemory::HEAD_AT_GPU:
  case SyncedMemory::SYNCED:
#ifndef CPU_ONLY
    conj_diff = mutable_gpu_conj_diff();
    caffe_gpu_scal(count_, scale_factor, conj_diff);
    return;
#else
    NO_GPU;
#endif
  case SyncedMemory::UNINITIALIZED:
    return;
  default:
    LOG(FATAL) << "Unknown SyncedMemory head state: " << conj_diff_->head();
  }
}

template <typename Dtype>
void ComplexBlob<Dtype>::CopyFrom(const ComplexBlob& source, bool copy_diff, bool reshape) {
  if (source.count() != count_ || source.shape() != shape_) {
    if (reshape) {
      ReshapeLike(source);
    } else {
      LOG(FATAL) << "Trying to copy blobs of different sizes.";
    }
  }
  switch (Caffe::mode()) {
  case Caffe::GPU:
    if (copy_diff) {
      // TODO Also copy conj_diff?
      caffe_copy(count_, source.gpu_diff(),
          static_cast<Dtype*>(diff_->mutable_gpu_data()));
      caffe_copy(count_, source.gpu_conj_diff(),
          static_cast<Dtype*>(conj_diff_->mutable_gpu_data()));
    } else {
      caffe_copy(count_, source.gpu_data(),
          static_cast<Dtype*>(data_->mutable_gpu_data()));
    }
    break;
  case Caffe::CPU:
    if (copy_diff) {
      caffe_copy(count_, source.cpu_diff(),
          static_cast<Dtype*>(diff_->mutable_cpu_data()));
      caffe_copy(count_, source.cpu_conj_diff(),
          static_cast<Dtype*>(conj_diff_->mutable_cpu_data()));
    } else {
      caffe_copy(count_, source.cpu_data(),
          static_cast<Dtype*>(data_->mutable_cpu_data()));
    }
    break;
  default:
    LOG(FATAL) << "Unknown caffe mode.";
  }
}

template <typename Dtype>
void ComplexBlob<Dtype>::FromProto(const BlobProto& proto, bool reshape) {
  if (reshape) {
    vector<int> shape;
    if (proto.has_num() || proto.has_channels() ||
        proto.has_height() || proto.has_width()) {
      // Using deprecated 4D Blob dimensions --
      // shape is (num, channels, height, width).
      shape.resize(4);
      shape[0] = proto.num();
      shape[1] = proto.channels();
      shape[2] = proto.height();
      shape[3] = proto.width();
    } else {
      shape.resize(proto.shape().dim_size());
      for (int i = 0; i < proto.shape().dim_size(); ++i) {
        shape[i] = proto.shape().dim(i);
      }
    }
    Reshape(shape);
  } else {
    CHECK(ShapeEquals(proto)) << "shape mismatch (reshape not set)";
  }
  // copy data
  complex<Dtype>* data_vec = mutable_cpu_data();
  if (proto.double_data_size() > 0) {
    CHECK_EQ(count_, proto.double_data_size());
    for (int i = 0; i < count_; ++i) {
      data_vec[i] = proto.double_data(i);
    }
  } else {
    CHECK_EQ(count_, proto.data_size());
    for (int i = 0; i < count_; ++i) {
      data_vec[i] = proto.data(i);
    }
  }
  if (proto.double_diff_size() > 0) {
    CHECK_EQ(count_, proto.double_diff_size());
    complex<Dtype>* diff_vec = mutable_cpu_diff();
    for (int i = 0; i < count_; ++i) {
      diff_vec[i] = proto.double_diff(i);
    }
  } else if (proto.diff_size() > 0) {
    CHECK_EQ(count_, proto.diff_size());
    complex<Dtype>* diff_vec = mutable_cpu_diff();
    for (int i = 0; i < count_; ++i) {
      diff_vec[i] = proto.diff(i);
    }
  }
  if (proto.double_conj_diff_size() > 0) {
    CHECK_EQ(count_, proto.double_conj_diff_size());
    Dtype* conj_diff_vec = mutable_cpu_conj_diff();
    for (int i = 0; i < count_; ++i) {
      conj_diff_vec[i] = proto.double_conj_diff(i);
    }
  } else if (proto.conj_diff_size() > 0) {
    CHECK_EQ(count_, proto.conj_diff_size());
    Dtype* diff_vec = mutable_cpu_conj_diff();
    for (int i = 0; i < count_; ++i) {
      conj_diff_vec[i] = proto.conj_diff(i);
    }
  }
}

template <>
void Blob<double>::ToProto(BlobProto* proto, bool write_diff) const {
  proto->clear_shape();
  for (int i = 0; i < shape_.size(); ++i) {
    proto->mutable_shape()->add_dim(shape_[i]);
  }
  proto->clear_double_data();
  proto->clear_double_diff();
  proto->clear_double_conj_diff();
    const complex<double>* data_vec = cpu_data();
  for (int i = 0; i < count_; ++i) {
    proto->add_double_data(data_vec[i]);
  }
  if (write_diff) {
    const complex<double>* diff_vec = cpu_diff();
    for (int i = 0; i < count_; ++i) {
      proto->add_double_diff(diff_vec[i]);
    }
    const complex<double>* conj_diff_vec = cpu_conj_diff();
    for (int i = 0; i < count_; ++i) {
      proto->add_double_conj_diff(conj_diff_vec[i]);
    }
  }
}

template <>
void Blob<float>::ToProto(BlobProto* proto, bool write_diff) const {
  proto->clear_shape();
  for (int i = 0; i < shape_.size(); ++i) {
    proto->mutable_shape()->add_dim(shape_[i]);
  }
  proto->clear_data();
  proto->clear_diff();
  proto->clear_conj_diff();
  const complex<float>* data_vec = cpu_data();
  for (int i = 0; i < count_; ++i) {
    proto->add_data(data_vec[i]);
  }
  if (write_diff) {
    const complex<float>* diff_vec = cpu_diff();
    for (int i = 0; i < count_; ++i) {
      proto->add_diff(diff_vec[i]);
    }
    const complex<float>* conj_diff_vec = cpu_conj_diff();
    for (int i = 0; i < count_; ++i) {
      proto->add_conj_diff(conj_diff_vec[i]);
    }
  }
}

INSTANTIATE_CLASS(ComplexBlob);
template class ComplexBlob<int>;
template class ComplexBlob<unsigned int>;

}  // namespace caffe

