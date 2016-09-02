#include <math_functions.h>  // CUDA's, not caffe's, for fabs, signbit
#include <thrust/device_vector.h>
#include <thrust/functional.h>  // thrust::plus
#include <thrust/reduce.h>

#include <cmath>

#include "caffe/common.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <>
void caffe_gpu_gemm<float>(const CBLAS_TRANSPOSE TransA,
    const CBLAS_TRANSPOSE TransB, const int M, const int N, const int K,
    const float alpha, const float* A, const float* B, const float beta,
    float* C) {
  // Note that cublas follows fortran order.
  int lda = (TransA == CblasNoTrans) ? K : M;
  int ldb = (TransB == CblasNoTrans) ? N : K;
  cublasOperation_t cuTransA =
      (TransA == CblasNoTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;
  cublasOperation_t cuTransB =
      (TransB == CblasNoTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;
  CUBLAS_CHECK(cublasSgemm(Caffe::cublas_handle(), cuTransB, cuTransA,
      N, M, K, &alpha, B, ldb, A, lda, &beta, C, N));
}

template <>
void caffe_gpu_gemm<double>(const CBLAS_TRANSPOSE TransA,
    const CBLAS_TRANSPOSE TransB, const int M, const int N, const int K,
    const double alpha, const double* A, const double* B, const double beta,
    double* C) {
  // Note that cublas follows fortran order.
  int lda = (TransA == CblasNoTrans) ? K : M;
  int ldb = (TransB == CblasNoTrans) ? N : K;
  cublasOperation_t cuTransA =
      (TransA == CblasNoTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;
  cublasOperation_t cuTransB =
      (TransB == CblasNoTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;
  CUBLAS_CHECK(cublasDgemm(Caffe::cublas_handle(), cuTransB, cuTransA,
      N, M, K, &alpha, B, ldb, A, lda, &beta, C, N));
}

template <>
void caffe_gpu_gemm<std::complex<float> >(const CBLAS_TRANSPOSE TransA,
    const CBLAS_TRANSPOSE TransB, const int M, const int N, const int K,
    const std::complex<float> alpha, const std::complex<float>* A, const std::complex<float>* B,
    const std::complex<float> beta, std::complex<float>* C) {
  // Note that cublas follows fortran order.
  int lda = (TransA == CblasNoTrans) ? K : M;
  int ldb = (TransB == CblasNoTrans) ? N : K;
  cublasOperation_t cuTransA;
  if(TransA == CblasNoTrans) {
    cuTransA = CUBLAS_OP_N;
  }
  else if(TransA == CblasTrans){
    cuTransA = CUBLAS_OP_T;
  }
  else {
    cuTransA = CUBLAS_OP_C;
  }
  cublasOperation_t cuTransB;
  if(TransB == CblasNoTrans) {
    cuTransB = CUBLAS_OP_N;
  }
  else if(TransB == CblasTrans){
    cuTransB = CUBLAS_OP_T;
  }
  else {
    cuTransB = CUBLAS_OP_C;
  }
  CUBLAS_CHECK(cublasCgemm(Caffe::cublas_handle(), cuTransB, cuTransA,
      N, M, K, (const cuComplex*)&alpha, (const cuComplex*)B, ldb, (const cuComplex*)A, lda,
      (const cuComplex*)&beta, (cuComplex*)C, N));
}

template <>
void caffe_gpu_gemm<std::complex<double> >(const CBLAS_TRANSPOSE TransA,
    const CBLAS_TRANSPOSE TransB, const int M, const int N, const int K,
    const std::complex<double> alpha, const std::complex<double>* A, const std::complex<double>* B,
    const std::complex<double> beta, std::complex<double>* C) {
  // Note that cublas follows fortran order.
  int lda = (TransA == CblasNoTrans) ? K : M;
  int ldb = (TransB == CblasNoTrans) ? N : K;
  cublasOperation_t cuTransA;
  if(TransA == CblasNoTrans) {
    cuTransA = CUBLAS_OP_N;
  }
  else if(TransA == CblasTrans){
    cuTransA = CUBLAS_OP_T;
  }
  else {
    cuTransA = CUBLAS_OP_C;
  }
  cublasOperation_t cuTransB;
  if(TransB == CblasNoTrans) {
    cuTransB = CUBLAS_OP_N;
  }
  else if(TransB == CblasTrans){
    cuTransB = CUBLAS_OP_T;
  }
  else {
    cuTransB = CUBLAS_OP_C;
  }
  CUBLAS_CHECK(cublasZgemm(Caffe::cublas_handle(), cuTransB, cuTransA,
      N, M, K, (const cuDoubleComplex*)&alpha, (const cuDoubleComplex*)B, ldb, (const cuDoubleComplex*)A, lda,
      (const cuDoubleComplex*)&beta, (cuDoubleComplex*)C, N));
}

template <>
void caffe_gpu_gemv<float>(const CBLAS_TRANSPOSE TransA, const int M,
    const int N, const float alpha, const float* A, const float* x,
    const float beta, float* y) {
  cublasOperation_t cuTransA =
      (TransA == CblasNoTrans) ? CUBLAS_OP_T : CUBLAS_OP_N;
  CUBLAS_CHECK(cublasSgemv(Caffe::cublas_handle(), cuTransA, N, M, &alpha,
      A, N, x, 1, &beta, y, 1));
}

template <>
void caffe_gpu_gemv<double>(const CBLAS_TRANSPOSE TransA, const int M,
    const int N, const double alpha, const double* A, const double* x,
    const double beta, double* y) {
  cublasOperation_t cuTransA =
      (TransA == CblasNoTrans) ? CUBLAS_OP_T : CUBLAS_OP_N;
  CUBLAS_CHECK(cublasDgemv(Caffe::cublas_handle(), cuTransA, N, M, &alpha,
      A, N, x, 1, &beta, y, 1));
}

template <>
void caffe_gpu_gemv<std::complex<float> >(const CBLAS_TRANSPOSE TransA, const int M,
    const int N, const std::complex<float> alpha, const std::complex<float>* A, const std::complex<float>* x,
    const std::complex<float> beta, std::complex<float>* y) {
  cublasOperation_t cuTransA;
  if(TransA == CblasNoTrans) {
    cuTransA = CUBLAS_OP_N;
  }
  else if(TransA == CblasTrans){
    cuTransA = CUBLAS_OP_T;
  }
  else {
    cuTransA = CUBLAS_OP_C;
  }
  CUBLAS_CHECK(cublasCgemv(Caffe::cublas_handle(), cuTransA, N, M, (const cuComplex*)&alpha,
      (const cuComplex*)A, N, (const cuComplex*)x, 1, (const cuComplex*)&beta, (cuComplex*)y, 1));
}

template <>
void caffe_gpu_gemv<std::complex<double> >(const CBLAS_TRANSPOSE TransA, const int M,
    const int N, const std::complex<double> alpha, const std::complex<double>* A, const std::complex<double>* x,
    const std::complex<double> beta, std::complex<double>* y) {
  cublasOperation_t cuTransA;
  if(TransA == CblasNoTrans) {
    cuTransA = CUBLAS_OP_N;
  }
  else if(TransA == CblasTrans){
    cuTransA = CUBLAS_OP_T;
  }
  else {
    cuTransA = CUBLAS_OP_C;
  }
  CUBLAS_CHECK(cublasZgemv(Caffe::cublas_handle(), cuTransA, N, M, (const cuDoubleComplex*)&alpha,
      (const cuDoubleComplex*)A, N, (const cuDoubleComplex*)x, 1, (const cuDoubleComplex*)&beta, (cuDoubleComplex*)y, 1));
}

template <>
void caffe_gpu_axpy<float>(const int N, const float alpha, const float* X,
    float* Y) {
  CUBLAS_CHECK(cublasSaxpy(Caffe::cublas_handle(), N, &alpha, X, 1, Y, 1));
}

template <>
void caffe_gpu_axpy<double>(const int N, const double alpha, const double* X,
    double* Y) {
  CUBLAS_CHECK(cublasDaxpy(Caffe::cublas_handle(), N, &alpha, X, 1, Y, 1));
}

void caffe_gpu_memcpy(const size_t N, const void* X, void* Y) {
  if (X != Y) {
    CUDA_CHECK(cudaMemcpy(Y, X, N, cudaMemcpyDefault));  // NOLINT(caffe/alt_fn)
  }
}

template <>
void caffe_gpu_axpy<std::complex<float> >(const int N, const std::complex<float> alpha, const std::complex<float>* X,
    std::complex<float>* Y) {
  CUBLAS_CHECK(cublasCaxpy(Caffe::cublas_handle(), N, (const cuComplex*)&alpha, (const cuComplex*)X, 1, (cuComplex*)Y, 1));
}

template <>
void caffe_gpu_axpy<std::complex<double> >(const int N, const std::complex<double> alpha, const std::complex<double>* X,
    std::complex<double>* Y) {
  CUBLAS_CHECK(cublasZaxpy(Caffe::cublas_handle(), N, (const cuDoubleComplex*)&alpha, (const cuDoubleComplex*)X, 1, (cuDoubleComplex*)Y, 1));
}

template <>
void caffe_gpu_scal<float>(const int N, const float alpha, float *X) {
  CUBLAS_CHECK(cublasSscal(Caffe::cublas_handle(), N, &alpha, X, 1));
}

template <>
void caffe_gpu_scal<double>(const int N, const double alpha, double *X) {
  CUBLAS_CHECK(cublasDscal(Caffe::cublas_handle(), N, &alpha, X, 1));
}

template <>
void caffe_gpu_scal<std::complex<float> >(const int N, const std::complex<float> alpha, std::complex<float> *X) {
  CUBLAS_CHECK(cublasCscal(Caffe::cublas_handle(), N, (const cuComplex*)&alpha, (cuComplex*)X, 1));
}

template <>
void caffe_gpu_scal<std::complex<double> >(const int N, const std::complex<double> alpha, std::complex<double> *X) {
  CUBLAS_CHECK(cublasZscal(Caffe::cublas_handle(), N, (const cuDoubleComplex*)&alpha, (cuDoubleComplex*)X, 1));
}

template <>
void caffe_gpu_axpby<float>(const int N, const float alpha, const float* X,
    const float beta, float* Y) {
  caffe_gpu_scal<float>(N, beta, Y);
  caffe_gpu_axpy<float>(N, alpha, X, Y);
}

template <>
void caffe_gpu_axpby<double>(const int N, const double alpha, const double* X,
    const double beta, double* Y) {
  caffe_gpu_scal<double>(N, beta, Y);
  caffe_gpu_axpy<double>(N, alpha, X, Y);
}

template <>
void caffe_gpu_axpby<std::complex<float> >(const int N, const std::complex<float> alpha, const std::complex<float>* X,
    const std::complex<float> beta, std::complex<float>* Y) {
  caffe_gpu_scal<std::complex<float> >(N, beta, Y);
  caffe_gpu_axpy<std::complex<float> >(N, alpha, X, Y);
}

template <>
void caffe_gpu_axpby<std::complex<double> >(const int N, const std::complex<double> alpha, const std::complex<double>* X,
    const std::complex<double> beta, std::complex<double>* Y) {
  caffe_gpu_scal<std::complex<double> >(N, beta, Y);
  caffe_gpu_axpy<std::complex<double> >(N, alpha, X, Y);
}

template <>
void caffe_gpu_dot<float>(const int n, const float* x, const float* y,
    float* out, const bool conj_x) {
  CUBLAS_CHECK(cublasSdot(Caffe::cublas_handle(), n, x, 1, y, 1, out));
}

template <>
void caffe_gpu_dot<double>(const int n, const double* x, const double* y,
    double * out, const bool conj_x) {
  CUBLAS_CHECK(cublasDdot(Caffe::cublas_handle(), n, x, 1, y, 1, out));
}

template <>
void caffe_gpu_dot<std::complex<float> >(const int n, const std::complex<float>* x, const std::complex<float>* y,
    std::complex<float>* out, const bool conj_x) {
  if(conj_x) {
    CUBLAS_CHECK(cublasCdotc(Caffe::cublas_handle(), n, (const cuComplex*)x, 1, (const cuComplex*)y, 1, (cuComplex*)out));
  } else {
    CUBLAS_CHECK(cublasCdotu(Caffe::cublas_handle(), n, (const cuComplex*)x, 1, (const cuComplex*)y, 1, (cuComplex*)out));
  }
}

template <>
void caffe_gpu_dot<std::complex<double> >(const int n, const std::complex<double>* x, const std::complex<double>* y,
    std::complex<double>* out, const bool conj_x) {
  if(conj_x) {
    CUBLAS_CHECK(cublasZdotc(Caffe::cublas_handle(), n, (const cuDoubleComplex*)x, 1, (const cuDoubleComplex*)y, 1, (cuDoubleComplex*)out));
  } else {
    CUBLAS_CHECK(cublasZdotu(Caffe::cublas_handle(), n, (const cuDoubleComplex*)x, 1, (const cuDoubleComplex*)y, 1, (cuDoubleComplex*)out));
  }
}

template <>
void caffe_gpu_asum<float>(const int n, const float* x, float* y) {
  CUBLAS_CHECK(cublasSasum(Caffe::cublas_handle(), n, x, 1, y));
}

template <>
void caffe_gpu_asum<double>(const int n, const double* x, double* y) {
  CUBLAS_CHECK(cublasDasum(Caffe::cublas_handle(), n, x, 1, y));
}

template <>
void caffe_gpu_asum<std::complex<float> >(const int n, const std::complex<float>* x, std::complex<float>* y) {
  float result;
  CUBLAS_CHECK(cublasScasum(Caffe::cublas_handle(), n, (cuComplex*)x, 1, &result));
  *y = std::complex<float>(result);
}

template <>
void caffe_gpu_asum<std::complex<double> >(const int n, const std::complex<double>* x, std::complex<double>* y) {
  double result;
  CUBLAS_CHECK(cublasDzasum(Caffe::cublas_handle(), n, (cuDoubleComplex*)x, 1, &result));
  *y = std::complex<double>(result);
}

template <>
void caffe_gpu_scale<float>(const int n, const float alpha, const float *x,
                            float* y) {
  CUBLAS_CHECK(cublasScopy(Caffe::cublas_handle(), n, x, 1, y, 1));
  CUBLAS_CHECK(cublasSscal(Caffe::cublas_handle(), n, &alpha, y, 1));
}

template <>
void caffe_gpu_scale<double>(const int n, const double alpha, const double *x,
                             double* y) {
  CUBLAS_CHECK(cublasDcopy(Caffe::cublas_handle(), n, x, 1, y, 1));
  CUBLAS_CHECK(cublasDscal(Caffe::cublas_handle(), n, &alpha, y, 1));
}

template <>
void caffe_gpu_scale<std::complex<float> >(const int n, const std::complex<float> alpha, const std::complex<float> *x,
                            std::complex<float>* y) {
  CUBLAS_CHECK(cublasCcopy(Caffe::cublas_handle(), n, (const cuComplex*)x, 1, (cuComplex*)y, 1));
  CUBLAS_CHECK(cublasCscal(Caffe::cublas_handle(), n, (const cuComplex*)&alpha, (cuComplex*)y, 1));
}

template <>
void caffe_gpu_scale<std::complex<double> >(const int n, const std::complex<double> alpha, const std::complex<double> *x,
                            std::complex<double>* y) {
  CUBLAS_CHECK(cublasZcopy(Caffe::cublas_handle(), n, (const cuDoubleComplex*)x, 1, (cuDoubleComplex*)y, 1));
  CUBLAS_CHECK(cublasZscal(Caffe::cublas_handle(), n, (const cuDoubleComplex*)&alpha, (cuDoubleComplex*)y, 1));
}

template <typename Dtype>
__global__ void set_kernel(const int n, const Dtype alpha, Dtype* y) {
  CUDA_KERNEL_LOOP(index, n) {
    y[index] = alpha;
  }
}

template <typename Dtype>
void caffe_gpu_set(const int N, const Dtype alpha, Dtype* Y) {
  if (alpha == 0) {
    CUDA_CHECK(cudaMemset(Y, 0, sizeof(Dtype) * N));  // NOLINT(caffe/alt_fn)
    return;
  }
  // NOLINT_NEXT_LINE(whitespace/operators)
  set_kernel<Dtype><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, alpha, Y);
}

template void caffe_gpu_set<int>(const int N, const int alpha, int* Y);
template void caffe_gpu_set<float>(const int N, const float alpha, float* Y);
template void caffe_gpu_set<double>(const int N, const double alpha, double* Y);

template <>
void caffe_gpu_set(const int N, const std::complex<float> alpha, std::complex<float>* Y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  set_kernel<cuComplex><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, *(const cuComplex*)&alpha, (cuComplex*)Y);
}

template <>
void caffe_gpu_set(const int N, const std::complex<double> alpha, std::complex<double>* Y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  set_kernel<cuDoubleComplex><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, *(const cuDoubleComplex*)&alpha, (cuDoubleComplex*)Y);
}


template <typename Dtype>
__global__ void add_scalar_kernel(const int n, const Dtype alpha, Dtype* y) {
  CUDA_KERNEL_LOOP(index, n) {
    y[index] += alpha;
  }
}

template __global__ void add_scalar_kernel(const int n, const float alpha, float* y);
template __global__ void add_scalar_kernel(const int n, const double alpha, double* y);

template <>
__global__ void add_scalar_kernel(const int n, const cuComplex alpha, cuComplex* y) {
  CUDA_KERNEL_LOOP(index, n) {
    y[index] = cuCaddf(y[index],alpha);
  }
}

template <>
__global__ void add_scalar_kernel(const int n, const cuDoubleComplex alpha, cuDoubleComplex* y) {
  CUDA_KERNEL_LOOP(index, n) {
    y[index] = cuCadd(y[index],alpha);
  }
}

template <>
void caffe_gpu_add_scalar(const int N, const float alpha, float* Y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  add_scalar_kernel<float><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, alpha, Y);
}

template <>
void caffe_gpu_add_scalar(const int N, const double alpha, double* Y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  add_scalar_kernel<double><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, alpha, Y);
}

template <>
void caffe_gpu_add_scalar(const int N, const std::complex<float> alpha, std::complex<float>* Y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  add_scalar_kernel<cuComplex><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, *(const cuComplex*)&alpha, (cuComplex*)Y);
}

template <>
void caffe_gpu_add_scalar(const int N, const std::complex<double> alpha, std::complex<double>* Y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  add_scalar_kernel<cuDoubleComplex><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, *(const cuDoubleComplex*)&alpha, (cuDoubleComplex*)Y);
}

template <typename Dtype>
__global__ void add_kernel(const int n, const Dtype* a,
    const Dtype* b, Dtype* y) {
  CUDA_KERNEL_LOOP(index, n) {
    y[index] = a[index] + b[index];
  }
}

template __global__ void add_kernel(const int n, const float* a, const float* b, float* y);
template __global__ void add_kernel(const int n, const double* a, const double* b, double* y);

template <>
__global__ void add_kernel(const int n, const cuComplex* a, const cuComplex* b, cuComplex* y) {
  CUDA_KERNEL_LOOP(index, n) {
    y[index] = cuCaddf(a[index],b[index]);
  }
}

template <>
__global__ void add_kernel(const int n, const cuDoubleComplex* a, const cuDoubleComplex* b, cuDoubleComplex* y) {
  CUDA_KERNEL_LOOP(index, n) {
    y[index] = cuCadd(a[index],b[index]);
  }
}

template <>
void caffe_gpu_add<float>(const int N, const float* a, const float* b,
    float* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  add_kernel<float><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, a, b, y);
}

template <>
void caffe_gpu_add<double>(const int N, const double* a, const double* b,
    double* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  add_kernel<double><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, a, b, y);
}

template <>
void caffe_gpu_add<std::complex<float> >(const int N, const std::complex<float>* a, const std::complex<float>* b,
    std::complex<float>* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  add_kernel<cuComplex><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, (const cuComplex*)a, (const cuComplex*)b, (cuComplex*)y);
}

template <>
void caffe_gpu_add<std::complex<double> >(const int N, const std::complex<double>* a, const std::complex<double>* b,
    std::complex<double>* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  add_kernel<cuDoubleComplex><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, (const cuDoubleComplex*)a, (const cuDoubleComplex*)b, (cuDoubleComplex*)y);
}

template <typename Dtype>
__global__ void sub_kernel(const int n, const Dtype* a,
    const Dtype* b, Dtype* y) {
  CUDA_KERNEL_LOOP(index, n) {
    y[index] = a[index] - b[index];
  }
}

template __global__ void sub_kernel(const int n, const float* a, const float* b, float* y);
template __global__ void sub_kernel(const int n, const double* a, const double* b, double* y);

template <>
__global__ void sub_kernel(const int n, const cuComplex* a, const cuComplex* b, cuComplex* y) {
  CUDA_KERNEL_LOOP(index, n) {
    y[index] = cuCsubf(a[index],b[index]);
  }
}

template <>
__global__ void sub_kernel(const int n, const cuDoubleComplex* a, const cuDoubleComplex* b, cuDoubleComplex* y) {
  CUDA_KERNEL_LOOP(index, n) {
    y[index] = cuCsub(a[index],b[index]);
  }
}

template <>
void caffe_gpu_sub<float>(const int N, const float* a, const float* b,
    float* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  sub_kernel<float><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, a, b, y);
}

template <>
void caffe_gpu_sub<double>(const int N, const double* a, const double* b,
    double* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  sub_kernel<double><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, a, b, y);
}

template <>
void caffe_gpu_sub<std::complex<float> >(const int N, const std::complex<float>* a, const std::complex<float>* b,
    std::complex<float>* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  sub_kernel<cuComplex><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, (const cuComplex*)a, (const cuComplex*)b, (cuComplex*)y);
}

template <>
void caffe_gpu_sub<std::complex<double> >(const int N, const std::complex<double>* a, const std::complex<double>* b,
    std::complex<double>* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  sub_kernel<cuDoubleComplex><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, (const cuDoubleComplex*)a, (const cuDoubleComplex*)b, (cuDoubleComplex*)y);
}

template <typename Dtype>
__global__ void mul_kernel(const int n, const Dtype* a,
    const Dtype* b, Dtype* y) {
  CUDA_KERNEL_LOOP(index, n) {
    y[index] = a[index] * b[index];
  }
}

template __global__ void mul_kernel(const int n, const float* a, const float* b, float* y);
template __global__ void mul_kernel(const int n, const double* a, const double* b, double* y);

template <>
__global__ void mul_kernel(const int n, const cuComplex* a, const cuComplex* b, cuComplex* y) {
  CUDA_KERNEL_LOOP(index, n) {
    y[index] = cuCmulf(a[index],b[index]);
  }
}

template <>
__global__ void mul_kernel(const int n, const cuDoubleComplex* a, const cuDoubleComplex* b, cuDoubleComplex* y) {
  CUDA_KERNEL_LOOP(index, n) {
    y[index] = cuCmul(a[index],b[index]);
  }
}

template <>
void caffe_gpu_mul<float>(const int N, const float* a,
    const float* b, float* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  mul_kernel<float><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, a, b, y);
}

template <>
void caffe_gpu_mul<double>(const int N, const double* a,
    const double* b, double* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  mul_kernel<double><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, a, b, y);
}

template <>
void caffe_gpu_mul<std::complex<float> >(const int N, const std::complex<float>* a, const std::complex<float>* b,
    std::complex<float>* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  mul_kernel<cuComplex><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, (const cuComplex*)a, (const cuComplex*)b, (cuComplex*)y);
}

template <>
void caffe_gpu_mul<std::complex<double> >(const int N, const std::complex<double>* a, const std::complex<double>* b,
    std::complex<double>* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  mul_kernel<cuDoubleComplex><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, (const cuDoubleComplex*)a, (const cuDoubleComplex*)b, (cuDoubleComplex*)y);
}

template <typename Dtype>
__global__ void div_kernel(const int n, const Dtype* a,
    const Dtype* b, Dtype* y) {
  CUDA_KERNEL_LOOP(index, n) {
    y[index] = a[index] / b[index];
  }
}

template __global__ void div_kernel(const int n, const float* a, const float* b, float* y);
template __global__ void div_kernel(const int n, const double* a, const double* b, double* y);

template <>
__global__ void div_kernel(const int n, const cuComplex* a, const cuComplex* b, cuComplex* y) {
  CUDA_KERNEL_LOOP(index, n) {
    y[index] = cuCdivf(a[index],b[index]);
  }
}

template <>
__global__ void div_kernel(const int n, const cuDoubleComplex* a, const cuDoubleComplex* b, cuDoubleComplex* y) {
  CUDA_KERNEL_LOOP(index, n) {
    y[index] = cuCdiv(a[index],b[index]);
  }
}

template <>
void caffe_gpu_div<float>(const int N, const float* a,
    const float* b, float* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  div_kernel<float><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, a, b, y);
}

template <>
void caffe_gpu_div<double>(const int N, const double* a,
    const double* b, double* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  div_kernel<double><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, a, b, y);
}

template <>
void caffe_gpu_div<std::complex<float> >(const int N, const std::complex<float>* a, const std::complex<float>* b,
    std::complex<float>* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  div_kernel<cuComplex><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, (const cuComplex*)a, (const cuComplex*)b, (cuComplex*)y);
}

template <>
void caffe_gpu_div<std::complex<double> >(const int N, const std::complex<double>* a, const std::complex<double>* b,
    std::complex<double>* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  div_kernel<cuDoubleComplex><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, (const cuDoubleComplex*)a, (const cuDoubleComplex*)b, (cuDoubleComplex*)y);
}

template <typename Dtype>
__global__ void abs_kernel(const int n, const Dtype* a, Dtype* y) {
  CUDA_KERNEL_LOOP(index, n) {
    y[index] = abs(a[index]);
  }
}

template __global__ void abs_kernel(const int n, const float* a, float* y);
template __global__ void abs_kernel(const int n, const double* a, double* y);

template <>
__global__ void abs_kernel(const int n, const cuComplex* a, cuComplex* y) {
  CUDA_KERNEL_LOOP(index, n) {
    y[index] = make_cuFloatComplex(cuCabsf(a[index]),0);
  }
}

template <>
__global__ void abs_kernel(const int n, const cuDoubleComplex* a, cuDoubleComplex* y) {
  CUDA_KERNEL_LOOP(index, n) {
    y[index] = make_cuDoubleComplex(cuCabs(a[index]),0);
  }
}

template <>
void caffe_gpu_abs<float>(const int N, const float* a, float* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  abs_kernel<float><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, a, y);
}

template <>
void caffe_gpu_abs<double>(const int N, const double* a, double* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  abs_kernel<double><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, a, y);
}

template <>
void caffe_gpu_abs<std::complex<float> >(const int N, const std::complex<float>* a, std::complex<float>* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  abs_kernel<cuComplex><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, (const cuComplex*)a, (cuComplex*)y);
}

template <>
void caffe_gpu_abs<std::complex<double> >(const int N, const std::complex<double>* a, std::complex<double>* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  abs_kernel<cuDoubleComplex><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, (const cuDoubleComplex*)a, (cuDoubleComplex*)y);
}

__global__ void abs_kernel(const int n, const cuComplex* a, float* y) {
  CUDA_KERNEL_LOOP(index, n) {
    y[index] = cuCabsf(a[index]);
  }
}

__global__ void abs_kernel(const int n, const cuDoubleComplex* a, double* y) {
  CUDA_KERNEL_LOOP(index, n) {
    y[index] = cuCabs(a[index]);
  }
}

template <>
void caffe_gpu_abs<float>(const int N, const std::complex<float>* a, float* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  abs_kernel<<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, (const cuComplex*)a, y);
}

template <>
void caffe_gpu_abs<double>(const int N, const std::complex<double>* a, double* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  abs_kernel<<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, (const cuDoubleComplex*)a, y);
}


template <typename Dtype>
__global__ void exp_kernel(const int n, const Dtype* a, Dtype* y) {
  CUDA_KERNEL_LOOP(index, n) {
    y[index] = exp(a[index]);
  }
}

template __global__ void exp_kernel(const int n, const float* a, float* y);
template __global__ void exp_kernel(const int n, const double* a, double* y);

__device__ void caffe_gpu_complex_exp(const cuComplex a, cuComplex &b) {
    float exp_real = expf(a.x);
    float sin_imag;
    float cos_imag;
    sincosf(a.y, &sin_imag, &cos_imag);
    b.x = exp_real*cos_imag;
    b.y = exp_real*sin_imag;
}

__device__ void caffe_gpu_complex_exp(const cuDoubleComplex a, cuDoubleComplex &b) {
    double exp_real = exp(a.x);
    double sin_imag;
    double cos_imag;
    sincos(a.y, &sin_imag, &cos_imag);
    b.x = exp_real*cos_imag;
    b.y = exp_real*sin_imag;
}

__global__ void exp_kernel(const int n, const cuComplex* a, cuComplex* b) {
  CUDA_KERNEL_LOOP(index, n) {
    caffe_gpu_complex_exp(a[index],b[index]);
  }
}

__global__ void exp_kernel(const int n, const cuDoubleComplex* a, cuDoubleComplex* b) {
  CUDA_KERNEL_LOOP(index, n) {
	caffe_gpu_complex_exp(a[index],b[index]);
  }
}

template <>
void caffe_gpu_exp<float>(const int N, const float* a, float* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  exp_kernel<float><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, a, y);
}

template <>
void caffe_gpu_exp<double>(const int N, const double* a, double* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  exp_kernel<double><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, a, y);
}

template <>
void caffe_gpu_exp<std::complex<float> >(const int N, const std::complex<float>* a, std::complex<float>* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  exp_kernel<<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, (const cuComplex*)a, (cuComplex*)y);
}

template <>
void caffe_gpu_exp<std::complex<double> >(const int N, const std::complex<double>* a, std::complex<double>* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  exp_kernel<<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, (const cuDoubleComplex*)a, (cuDoubleComplex*)y);
}

template <typename Dtype>
__global__ void log_kernel(const int n, const Dtype* a, Dtype* y) {
  CUDA_KERNEL_LOOP(index, n) {
    y[index] = log(a[index]);
  }
}

template __global__ void log_kernel(const int n, const float* a, float* y);
template __global__ void log_kernel(const int n, const double* a, double* y);

template <>
__global__ void log_kernel(const int n, const cuComplex* a, cuComplex* y) {
  CUDA_KERNEL_LOOP(index, n) {
    y[index].x = cuCabsf(a[index]);
    y[index].y = atan2f(a[index].y, a[index].x);
  }
}

template <>
__global__ void log_kernel(const int n, const cuDoubleComplex* a, cuDoubleComplex* y) {
  CUDA_KERNEL_LOOP(index, n) {
    y[index].x = cuCabs(a[index]);
    y[index].y = atan2(a[index].y, a[index].x);
  }
}

template <>
void caffe_gpu_log<float>(const int N, const float* a, float* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  log_kernel<float><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, a, y);
}

template <>
void caffe_gpu_log<double>(const int N, const double* a, double* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  log_kernel<double><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, a, y);
}

template <>
void caffe_gpu_log<std::complex<float> >(const int N, const std::complex<float>* a, std::complex<float>* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  log_kernel<cuComplex><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, (const cuComplex*)a, (cuComplex*)y);
}

template <>
void caffe_gpu_log<std::complex<double> >(const int N, const std::complex<double>* a, std::complex<double>* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  log_kernel<cuDoubleComplex><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, (const cuDoubleComplex*)a, (cuDoubleComplex*)y);
}

template <typename Dtype>
__global__ void powx_kernel(const int n, const Dtype* a,
    const Dtype alpha, Dtype* y) {
  CUDA_KERNEL_LOOP(index, n) {
    y[index] = pow(a[index], alpha);
  }
}

template <>
void caffe_gpu_powx<float>(const int N, const float* a,
    const float alpha, float* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  powx_kernel<float><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, a, alpha, y);
}

template <>
void caffe_gpu_powx<double>(const int N, const double* a,
    const double alpha, double* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  powx_kernel<double><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, a, alpha, y);
}

__global__ void powx_kernel(const int n, const cuComplex* a,
    const float alpha, cuComplex* y) {
  CUDA_KERNEL_LOOP(index, n) {
	// y = a^alpha
	//   = exp(alpha*log(a))
	// alpha*log(a) = {alpha*abs(a), alpha*arg(z)}
	// y = exp(alpha*log(a))
	//   = {exp(alpha*abs(a)*cos(alpha*arg(z)), exp(alpha*abs(a)*sign(alpha*arg(z))}
	float exp_alpha_abs_a = expf(alpha * cuCabsf(a[index]));
	float theta = atan2f(a[index].y, a[index].x);
	float cos_alpha_theta;
	float sin_alpha_theta;
	sincosf(alpha*theta, &sin_alpha_theta, &cos_alpha_theta);
    y[index].x = exp_alpha_abs_a * cos_alpha_theta;
    y[index].y = exp_alpha_abs_a * sin_alpha_theta;
  }
}

__global__ void powx_kernel(const int n, const cuDoubleComplex* a,
    const double alpha, cuDoubleComplex* y) {
  CUDA_KERNEL_LOOP(index, n) {
	// y = a^alpha
	//   = exp(alpha*log(a))
	// alpha*log(a) = {alpha*abs(a), alpha*arg(z)}
	// y = exp(alpha*log(a))
	//   = {exp(alpha*abs(a)*cos(alpha*arg(z)), exp(alpha*abs(a)*sign(alpha*arg(z))}
	double exp_alpha_abs_a = expf(alpha * cuCabs(a[index]));
	double theta = atan2(a[index].y, a[index].x);
	double cos_alpha_theta;
	double sin_alpha_theta;
	sincos(alpha*theta, &sin_alpha_theta, &cos_alpha_theta);
    y[index].x = exp_alpha_abs_a * cos_alpha_theta;
    y[index].y = exp_alpha_abs_a * sin_alpha_theta;
  }
}

template <>
void caffe_gpu_powx<float>(const int N, const std::complex<float>* a,
    const float alpha, std::complex<float>* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  powx_kernel<<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, (const cuComplex*)a, alpha, (cuComplex*)y);
}

template <>
void caffe_gpu_powx<double>(const int N, const std::complex<double>* a,
    const double alpha, std::complex<double>* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  powx_kernel<<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, (const cuDoubleComplex*)a, alpha, (cuDoubleComplex*)y);
}

__global__ void conj_kernel(const int n, const cuComplex* a, cuComplex* y) {
  CUDA_KERNEL_LOOP(index, n) {
    y[index] = cuConjf(a[index]);
  }
}

__global__ void conj_kernel(const int n, const cuDoubleComplex* a, cuDoubleComplex* y) {
  CUDA_KERNEL_LOOP(index, n) {
    y[index] = cuConj(a[index]);
  }
}

template <>
void caffe_gpu_conj<std::complex<float> >(const int N, const std::complex<float>* a, std::complex<float>* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  conj_kernel<<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, (const cuComplex*)a, (cuComplex*)y);
}

template <>
void caffe_gpu_conj<std::complex<double> >(const int N, const std::complex<double>* a, std::complex<double>* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  conj_kernel<<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, (const cuDoubleComplex*)a, (cuDoubleComplex*)y);
}


DEFINE_AND_INSTANTIATE_GPU_UNARY_FUNC(sign, y[index] = (Dtype(0) < x[index])
                                      - (x[index] < Dtype(0)));
DEFINE_AND_INSTANTIATE_GPU_UNARY_FUNC(sgnbit, y[index] = signbit(x[index]));

void caffe_gpu_rng_uniform(const int n, unsigned int* r) {
  CURAND_CHECK(curandGenerate(Caffe::curand_generator(), r, n));
}

template <>
void caffe_gpu_rng_uniform<float>(const int n, const float a, const float b,
                                  float* r) {
  CURAND_CHECK(curandGenerateUniform(Caffe::curand_generator(), r, n));
  const float range = b - a;
  if (range != static_cast<float>(1)) {
    caffe_gpu_scal(n, range, r);
  }
  if (a != static_cast<float>(0)) {
    caffe_gpu_add_scalar(n, a, r);
  }
}

template <>
void caffe_gpu_rng_uniform<double>(const int n, const double a, const double b,
                                   double* r) {
  CURAND_CHECK(curandGenerateUniformDouble(Caffe::curand_generator(), r, n));
  const double range = b - a;
  if (range != static_cast<double>(1)) {
    caffe_gpu_scal(n, range, r);
  }
  if (a != static_cast<double>(0)) {
    caffe_gpu_add_scalar(n, a, r);
  }
}

template <>
void caffe_gpu_rng_uniform<std::complex<float> >(const int n, const std::complex<float> a,
		const std::complex<float> b, std::complex<float>* r) {
  CURAND_CHECK(curandGenerateUniform(Caffe::curand_generator(), reinterpret_cast<float*>(r), n*2));
  const std::complex<float> range = b - a;
  // Zero out imaginary
  for(int i = 0; i < n; i++) {
	  r[i] = std::complex<float>(std::real(r[i]),0);
  }
  if (std::real(range) != 1 && std::imag(range) != 0) {
    caffe_gpu_scal(n, range, r);
  }
  if (std::real(a) != 0 && std::imag(a) != 0) {
    caffe_gpu_add_scalar(n, a, r);
  }
}

template <>
void caffe_gpu_rng_uniform<std::complex<double> >(const int n, const std::complex<double> a,
		const std::complex<double> b, std::complex<double>* r) {
  CURAND_CHECK(curandGenerateUniformDouble(Caffe::curand_generator(), reinterpret_cast<double*>(r), n*2));
  const std::complex<double> range = b - a;
  // Zero out imaginary
  for(int i = 0; i < n; i++) {
	  r[i] = std::complex<double>(std::real(r[i]),0);
  }
  if (std::real(range) != 1 && std::imag(range) != 0) {
    caffe_gpu_scal(n, range, r);
  }
  if (std::real(a) != 0 && std::imag(a) != 0) {
    caffe_gpu_add_scalar(n, a, r);
  }
}

template <>
void caffe_gpu_rng_gaussian(const int n, const float mu, const float sigma,
                            float* r) {
  CURAND_CHECK(
      curandGenerateNormal(Caffe::curand_generator(), r, n, mu, sigma));
}

template <>
void caffe_gpu_rng_gaussian(const int n, const double mu, const double sigma,
                            double* r) {
  CURAND_CHECK(
      curandGenerateNormalDouble(Caffe::curand_generator(), r, n, mu, sigma));
}

template <>
void caffe_gpu_rng_gaussian(const int n, const std::complex<float> mu, const std::complex<float> sigma,
                            std::complex<float>* r) {
  CURAND_CHECK(
      curandGenerateNormal(Caffe::curand_generator(), reinterpret_cast<float*>(r), n*2,
    		  std::real(mu), std::real(sigma)));
}

template <>
void caffe_gpu_rng_gaussian(const int n, const std::complex<double> mu, const std::complex<double> sigma,
                            std::complex<double>* r) {
  CURAND_CHECK(
      curandGenerateNormalDouble(Caffe::curand_generator(), reinterpret_cast<double*>(r), n*2,
    		  std::real(mu), std::real(sigma)));
}


}  // namespace caffe
