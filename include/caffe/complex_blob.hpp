#ifndef CAFFE_COMPLEX_BLOB_HPP_
#define CAFFE_COMPLEX_BLOB_HPP_

#include <algorithm>
#include <string>
#include <vector>
#include <complex>

#include "caffe/common.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/syncedmem.hpp"

const int kMaxBlobAxes = 32;

using std::complex;

namespace caffe {

/**
 * @brief An extension to Blob to handle complex-valued data.
 *
 * TODO(dox): more thorough description.
 */
template <typename Dtype>
class ComplexBlob : public Blob<Dtype>{
 public:
  ComplexBlob()
       : data_(), diff_(), conj_diff_(), count_(0), capacity_(0) {}

  /// @brief Deprecated; use <code>Blob(const vector<int>& shape)</code>.
  explicit ComplexBlob(const int num, const int channels, const int height,
      const int width);
  explicit ComplexBlob(const vector<int>& shape);

  /// @brief Deprecated; use <code>Reshape(const vector<int>& shape)</code>.
  void Reshape(const int num, const int channels, const int height,
      const int width);
  /**
   * @brief Change the dimensions of the blob, allocating new memory if
   *        necessary.
   *
   * This function can be called both to create an initial allocation
   * of memory, and to adjust the dimensions of a top blob during Layer::Reshape
   * or Layer::Forward. When changing the size of blob, memory will only be
   * reallocated if sufficient memory does not already exist, and excess memory
   * will never be freed.
   *
   * Note that reshaping an input blob and immediately calling Net::Backward is
   * an error; either Net::Forward or Net::Reshape need to be called to
   * propagate the new input shape to higher layers.
   */
  void Reshape(const vector<int>& shape);
  void Reshape(const BlobShape& shape);
  void ReshapeLike(const Blob& other);

  /**
   * @brief Copy from a source Blob.
   *
   * @param source the Blob to copy from
   * @param copy_diff if false, copy the data; if true, copy the diff
   * @param reshape if false, require this Blob to be pre-shaped to the shape
   *        of other (and die otherwise); if true, Reshape this Blob to other's
   *        shape if necessary
   */
  void CopyFrom(const Blob<Dtype>& source, bool copy_diff = false,
      bool reshape = false);

  inline complex<Dtype> data_at(const int n, const int c, const int h,
      const int w) const {
    return cpu_data()[offset(n, c, h, w)];
  }

  inline complex<Dtype> diff_at(const int n, const int c, const int h,
      const int w) const {
    return cpu_diff()[offset(n, c, h, w)];
  }

  inline complex<Dtype> conj_diff_at(const int n, const int c, const int h,
      const int w) const {
    return cpu_conj_diff()[offset(n, c, h, w)];
  }

  inline complex<Dtype> data_at(const vector<int>& index) const {
    return cpu_data()[offset(index)];
  }

  inline complex<Dtype> diff_at(const vector<int>& index) const {
    return cpu_diff()[offset(index)];
  }

  inline complex<Dtype> conj_diff_at(const vector<int>& index) const {
    return cpu_conj_diff()[offset(index)];
  }

  inline const shared_ptr<SyncedMemory>& data() const {
    CHECK(data_);
    return data_;
  }

  inline const shared_ptr<SyncedMemory>& diff() const {
    CHECK(diff_);
    return diff_;
  }

  inline const shared_ptr<SyncedMemory>& conj_diff() const {
    CHECK(conj_diff_);
    return conj_diff_;
  }

  const complex<Dtype>* cpu_conj_diff() const;
  const complex<Dtype>* gpu_conj_diff() const;
  complex<Dtype>* mutable_cpu_conj_diff();
  complex<Dtype>* mutable_gpu_conj_diff();
  void Update();
  void FromProto(const BlobProto& proto, bool reshape = true);
  void ToProto(BlobProto* proto, bool write_diff = false) const;

  /// @brief Compute the sum of absolute values (L1 norm) of the conj_diff.
  Dtype asum_conj_diff() const;
  /// @brief Compute the sum of squares (L2 norm squared) of the conj_diff.
  Dtype sumsq_conj_diff() const;

  /// @brief Scale the blob conj_diff by a constant factor.
  void scale_conj_diff(complex<Dtype> scale_factor);

  /**
   * @brief Set the data_ shared_ptr to point to the SyncedMemory holding the
   *        data_ of Blob other -- useful in Layer%s which simply perform a copy
   *        in their Forward pass.
   *
   * This deallocates the SyncedMemory holding this Blob's data_, as
   * shared_ptr calls its destructor when reset with the "=" operator.
   */
  void ShareData(const ComplexBlob& other);
  /**
   * @brief Set the diff_ shared_ptr to point to the SyncedMemory holding the
   *        diff_ of Blob other -- useful in Layer%s which simply perform a copy
   *        in their Forward pass.
   *
   * This deallocates the SyncedMemory holding this Blob's diff_, as
   * shared_ptr calls its destructor when reset with the "=" operator.
   */
  void ShareDiff(const ComplexBlob& other);

  /**
   * @brief Set the conj_diff_ shared_ptr to point to the SyncedMemory holding the
   *        conj_diff_ of Blob other -- useful in Layer%s which simply perform a copy
   *        in their Forward pass.
   *
   * This deallocates the SyncedMemory holding this Blob's conj_diff_, as
   * shared_ptr calls its destructor when reset with the "=" operator.
   */
  void ShareConjDiff(const ComplexBlob& other);

 protected:
  shared_ptr<SyncedMemory> conj_diff_;

  DISABLE_COPY_AND_ASSIGN(ComplexBlob);
};  // class ComplexBlob

}  // namespace caffe

#endif  // CAFFE_COMPLEX_BLOB_HPP_
