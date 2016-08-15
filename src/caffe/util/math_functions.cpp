#include <boost/math/special_functions/next.hpp>
#include <boost/random.hpp>

#include <limits>
#include <complex>

#include "caffe/common.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"

namespace caffe {

template<>
void caffe_cpu_gemm<float>(const CBLAS_TRANSPOSE TransA,
    const CBLAS_TRANSPOSE TransB, const int M, const int N, const int K,
    const float alpha, const float* A, const float* B, const float beta,
    float* C) {
  int lda = (TransA == CblasNoTrans) ? K : M;
  int ldb = (TransB == CblasNoTrans) ? N : K;
  cblas_sgemm(CblasRowMajor, TransA, TransB, M, N, K, alpha, A, lda, B,
      ldb, beta, C, N);
}

template<>
void caffe_cpu_gemm<double>(const CBLAS_TRANSPOSE TransA,
    const CBLAS_TRANSPOSE TransB, const int M, const int N, const int K,
    const double alpha, const double* A, const double* B, const double beta,
    double* C) {
  int lda = (TransA == CblasNoTrans) ? K : M;
  int ldb = (TransB == CblasNoTrans) ? N : K;
  cblas_dgemm(CblasRowMajor, TransA, TransB, M, N, K, alpha, A, lda, B,
      ldb, beta, C, N);
}

template<>
void caffe_cpu_gemm<std::complex<float> >(const CBLAS_TRANSPOSE TransA,
    const CBLAS_TRANSPOSE TransB, const int M, const int N, const int K,
    const std::complex<float> alpha, const std::complex<float>* A,
    const std::complex<float>* B, const std::complex<float> beta,
    std::complex<float>* C) {
  int lda = (TransA == CblasNoTrans) ? K : M;
  int ldb = (TransB == CblasNoTrans) ? N : K;
  cblas_cgemm(CblasRowMajor, TransA, TransB, M, N, K,
      reinterpret_cast<const float*>(&alpha), reinterpret_cast<const float*>(A), lda,
	  reinterpret_cast<const float*>(B), ldb,
	  reinterpret_cast<const float*>(&beta), reinterpret_cast<float*>(C), N);
}

template<>
void caffe_cpu_gemm<std::complex<double> >(const CBLAS_TRANSPOSE TransA,
    const CBLAS_TRANSPOSE TransB, const int M, const int N, const int K,
    const std::complex<double> alpha, const std::complex<double>* A,
    const std::complex<double>* B, const std::complex<double> beta,
    std::complex<double>* C) {
  int lda = (TransA == CblasNoTrans) ? K : M;
  int ldb = (TransB == CblasNoTrans) ? N : K;
  cblas_zgemm(CblasRowMajor, TransA, TransB, M, N, K,
	  reinterpret_cast<const double*>(&alpha), reinterpret_cast<const double*>(A), lda,
	  reinterpret_cast<const double*>(B), ldb,
	  reinterpret_cast<const double*>(&beta), reinterpret_cast<double*>(C), N);
}

template <>
void caffe_cpu_gemv<float>(const CBLAS_TRANSPOSE TransA, const int M,
    const int N, const float alpha, const float* A, const float* x,
    const float beta, float* y) {
  cblas_sgemv(CblasRowMajor, TransA, M, N, alpha, A, N, x, 1, beta, y, 1);
}

template <>
void caffe_cpu_gemv<double>(const CBLAS_TRANSPOSE TransA, const int M,
    const int N, const double alpha, const double* A, const double* x,
    const double beta, double* y) {
  cblas_dgemv(CblasRowMajor, TransA, M, N, alpha, A, N, x, 1, beta, y, 1);
}

template <>
void caffe_cpu_gemv<std::complex<float> >(const CBLAS_TRANSPOSE TransA, const int M,
    const int N, const std::complex<float> alpha, const std::complex<float>* A,
    const std::complex<float>* x,
    const std::complex<float> beta, std::complex<float>* y) {
  cblas_cgemv(CblasRowMajor, TransA, M, N,
		  reinterpret_cast<const float*>(&alpha), reinterpret_cast<const float*>(A),
		  N, reinterpret_cast<const float*>(x), 1,
		  reinterpret_cast<const float*>(&beta), reinterpret_cast<float*>(y), 1);
}

template <>
void caffe_cpu_gemv<std::complex<double> >(const CBLAS_TRANSPOSE TransA, const int M,
    const int N, const std::complex<double> alpha, const std::complex<double>* A,
    const std::complex<double>* x,
    const std::complex<double> beta, std::complex<double>* y) {
  cblas_zgemv(CblasRowMajor, TransA, M, N,
		  reinterpret_cast<const double*>(&alpha), reinterpret_cast<const double*>(A),
		  N, reinterpret_cast<const double*>(x), 1,
		  reinterpret_cast<const double*>(&beta), reinterpret_cast<double*>(y), 1);
}

template <>
void caffe_axpy<float>(const int N, const float alpha, const float* X,
    float* Y) { cblas_saxpy(N, alpha, X, 1, Y, 1); }

template <>
void caffe_axpy<double>(const int N, const double alpha, const double* X,
    double* Y) { cblas_daxpy(N, alpha, X, 1, Y, 1); }

template <>
void caffe_axpy<std::complex<float> >(const int N, const std::complex<float> alpha,
    const std::complex<float>* X,
    std::complex<float>* Y) {
  cblas_caxpy(N, reinterpret_cast<const float*>(&alpha),
		  reinterpret_cast<const float*>(X), 1,
		  reinterpret_cast<float*>(Y), 1); }

template <>
void caffe_axpy<std::complex<double> >(const int N, const std::complex<double> alpha,
    const std::complex<double>* X,
    std::complex<double>* Y) {
  cblas_zaxpy(N, reinterpret_cast<const double*>(&alpha),
		  reinterpret_cast<const double*>(X), 1,
		  reinterpret_cast<double*>(Y), 1); }

template <typename Dtype>
void caffe_set(const int N, const Dtype alpha, Dtype* Y) {
  if (alpha == 0) {
    memset(Y, 0, sizeof(Dtype) * N);  // NOLINT(caffe/alt_fn)
    return;
  }
  for (int i = 0; i < N; ++i) {
    Y[i] = alpha;
  }
}

template void caffe_set<int>(const int N, const int alpha, int* Y);
template void caffe_set<float>(const int N, const float alpha, float* Y);
template void caffe_set<double>(const int N, const double alpha, double* Y);

template <>
void caffe_set<std::complex<float> >(const int N, const std::complex<float> alpha, std::complex<float>* Y) {
  for (int i = 0; i < N; ++i) {
    Y[i] = alpha;
  }
}

template <>
void caffe_set<std::complex<double> >(const int N, const std::complex<double> alpha, std::complex<double>* Y) {
  for (int i = 0; i < N; ++i) {
    Y[i] = alpha;
  }
}

template <>
void caffe_add_scalar(const int N, const float alpha, float* Y) {
  for (int i = 0; i < N; ++i) {
    Y[i] += alpha;
  }
}

template <>
void caffe_add_scalar(const int N, const double alpha, double* Y) {
  for (int i = 0; i < N; ++i) {
    Y[i] += alpha;
  }
}

template <>
void caffe_add_scalar(const int N, const std::complex<float> alpha, std::complex<float>* Y) {
  for (int i = 0; i < N; ++i) {
    Y[i] += alpha;
  }
}

template <>
void caffe_add_scalar(const int N, const std::complex<double> alpha, std::complex<double>* Y) {
  for (int i = 0; i < N; ++i) {
    Y[i] += alpha;
  }
}


template <typename Dtype>
void caffe_copy(const int N, const Dtype* X, Dtype* Y) {
  if (X != Y) {
    if (Caffe::mode() == Caffe::GPU) {
#ifndef CPU_ONLY
      // NOLINT_NEXT_LINE(caffe/alt_fn)
      CUDA_CHECK(cudaMemcpy(Y, X, sizeof(Dtype) * N, cudaMemcpyDefault));
#else
      NO_GPU;
#endif
    } else {
      memcpy(Y, X, sizeof(Dtype) * N);  // NOLINT(caffe/alt_fn)
    }
  }
}

template void caffe_copy<int>(const int N, const int* X, int* Y);
template void caffe_copy<unsigned int>(const int N, const unsigned int* X,
    unsigned int* Y);
template void caffe_copy<float>(const int N, const float* X, float* Y);
template void caffe_copy<double>(const int N, const double* X, double* Y);
template void caffe_copy<std::complex<float> >(const int N, const std::complex<float>* X,
    std::complex<float>* Y);
template void caffe_copy<std::complex<double> >(const int N, const std::complex<double>* X,
    std::complex<double>* Y);

template <>
void caffe_scal<float>(const int N, const float alpha, float *X) {
  cblas_sscal(N, alpha, X, 1);
}

template <>
void caffe_scal<double>(const int N, const double alpha, double *X) {
  cblas_dscal(N, alpha, X, 1);
}

template <>
void caffe_scal<std::complex<float> >(const int N, const std::complex<float> alpha, std::complex<float> *X) {
  cblas_cscal(N, reinterpret_cast<const float*>(&alpha), reinterpret_cast<float*>(X), 1);
}

template <>
void caffe_scal<std::complex<double> >(const int N, const std::complex<double> alpha, std::complex<double> *X) {
  cblas_zscal(N, reinterpret_cast<const double*>(&alpha), reinterpret_cast<double*>(X), 1);
}

template <>
void caffe_cpu_axpby<float>(const int N, const float alpha, const float* X,
                            const float beta, float* Y) {
  cblas_saxpby(N, alpha, X, 1, beta, Y, 1);
}

template <>
void caffe_cpu_axpby<double>(const int N, const double alpha, const double* X,
                             const double beta, double* Y) {
  cblas_daxpby(N, alpha, X, 1, beta, Y, 1);
}

template <>
void caffe_cpu_axpby<std::complex<float> >(const int N, const std::complex<float> alpha, const std::complex<float>* X,
                            const std::complex<float> beta, std::complex<float>* Y) {
  cblas_caxpby(N, alpha, X, 1, beta, Y, 1);
}

template <>
void caffe_cpu_axpby<std::complex<double> >(const int N, const std::complex<double> alpha, const std::complex<double>* X,
                            const std::complex<double> beta, std::complex<double>* Y) {
  cblas_zaxpby(N, alpha, X, 1, beta, Y, 1);
}

template <>
void caffe_add<float>(const int n, const float* a, const float* b,
    float* y) {
  vsAdd(n, a, b, y);
}

template <>
void caffe_add<double>(const int n, const double* a, const double* b,
    double* y) {
  vdAdd(n, a, b, y);
}

template <>
void caffe_add<std::complex<float> >(const int n, const std::complex<float>* a, const std::complex<float>* b,
    std::complex<float>* y) {
  vcAdd(n, a, b, y);
}

template <>
void caffe_add<std::complex<double> >(const int n, const std::complex<double>* a, const std::complex<double>* b,
    std::complex<double>* y) {
  vzAdd(n, a, b, y);
}

template <>
void caffe_sub<float>(const int n, const float* a, const float* b,
    float* y) {
  vsSub(n, a, b, y);
}

template <>
void caffe_sub<double>(const int n, const double* a, const double* b,
    double* y) {
  vdSub(n, a, b, y);
}

template <>
void caffe_sub<std::complex<float> >(const int n, const std::complex<float>* a, const std::complex<float>* b,
    std::complex<float>* y) {
  vcSub(n, a, b, y);
}

template <>
void caffe_sub<std::complex<double> >(const int n, const std::complex<double>* a, const std::complex<double>* b,
    std::complex<double>* y) {
  vzSub(n, a, b, y);
}

template <>
void caffe_mul<float>(const int n, const float* a, const float* b,
    float* y) {
  vsMul(n, a, b, y);
}

template <>
void caffe_mul<double>(const int n, const double* a, const double* b,
    double* y) {
  vdMul(n, a, b, y);
}

template <>
void caffe_mul<std::complex<float> >(const int n, const std::complex<float>* a, const std::complex<float>* b,
    std::complex<float>* y) {
  vcMul(n, a, b, y);
}

template <>
void caffe_mul<std::complex<double> >(const int n, const std::complex<double>* a, const std::complex<double>* b,
    std::complex<double>* y) {
  vzMul(n, a, b, y);
}

template <>
void caffe_div<float>(const int n, const float* a, const float* b,
    float* y) {
  vsDiv(n, a, b, y);
}

template <>
void caffe_div<double>(const int n, const double* a, const double* b,
    double* y) {
  vdDiv(n, a, b, y);
}

template <>
void caffe_div<std::complex<float> >(const int n, const std::complex<float>* a, const std::complex<float>* b,
    std::complex<float>* y) {
  vcDiv(n, a, b, y);
}

template <>
void caffe_div<std::complex<double> >(const int n, const std::complex<double>* a, const std::complex<double>* b,
    std::complex<double>* y) {
  vzDiv(n, a, b, y);
}

template <>
void caffe_powx<float>(const int n, const float* a, const float b,
    float* y) {
  vsPowx(n, a, b, y);
}

template <>
void caffe_powx<double>(const int n, const double* a, const double b,
    double* y) {
  vdPowx(n, a, b, y);
}

template <>
void caffe_powx<std::complex<float> >(const int n, const std::complex<float>* a, const std::complex<float> b,
    std::complex<float>* y) {
  vcPowx(n, a, b, y);
}

template <>
void caffe_powx<std::complex<double> >(const int n, const std::complex<double>* a, const std::complex<double> b,
    std::complex<double>* y) {
  vzPowx(n, a, b, y);
}

template <>
void caffe_sqr<float>(const int n, const float* a, float* y) {
  vsSqr(n, a, y);
}

template <>
void caffe_sqr<double>(const int n, const double* a, double* y) {
  vdSqr(n, a, y);
}

template <>
void caffe_sqr<std::complex<float> >(const int n, const std::complex<float>* a, std::complex<float>* y) {
  vcSqr(n, a, y);
}

template <>
void caffe_sqr<std::complex<double> >(const int n, const std::complex<double>* a, std::complex<double>* y) {
  vzSqr(n, a, y);
}

template <>
void caffe_exp<float>(const int n, const float* a, float* y) {
  vsExp(n, a, y);
}

template <>
void caffe_exp<double>(const int n, const double* a, double* y) {
  vdExp(n, a, y);
}

template <>
void caffe_exp<std::complex<float> >(const int n, const std::complex<float>* a, std::complex<float>* y) {
  vcExp(n, a, y);
}

template <>
void caffe_exp<std::complex<double> >(const int n, const std::complex<double>* a, std::complex<double>* y) {
  vzExp(n, a, y);
}

template <>
void caffe_log<float>(const int n, const float* a, float* y) {
  vsLn(n, a, y);
}

template <>
void caffe_log<double>(const int n, const double* a, double* y) {
  vdLn(n, a, y);
}

template <>
void caffe_log<std::complex<float> >(const int n, const std::complex<float>* a, std::complex<float>* y) {
  vcLn(n, a, y);
}

template <>
void caffe_log<std::complex<double> >(const int n, const std::complex<double>* a, std::complex<double>* y) {
  vzLn(n, a, y);
}

template <>
void caffe_abs<float>(const int n, const float* a, float* y) {
    vsAbs(n, a, y);
}

template <>
void caffe_abs<double>(const int n, const double* a, double* y) {
    vdAbs(n, a, y);
}

template <>
void caffe_abs<std::complex<float> >(const int n, const std::complex<float>* a, std::complex<float>* y) {
    vcAbs(n, a, y);
}

template <>
void caffe_abs<std::complex<double> >(const int n, const std::complex<double>* a, std::complex<double>* y) {
    vzAbs(n, a, y);
}

unsigned int caffe_rng_rand() {
  return (*caffe_rng())();
}

template <typename Dtype>
Dtype caffe_nextafter(const Dtype b) {
  return boost::math::nextafter<Dtype>(
      b, std::numeric_limits<Dtype>::max());
}

template
float caffe_nextafter(const float b);

template
double caffe_nextafter(const double b);

template <typename Dtype>
void caffe_rng_uniform(const int n, const Dtype a, const Dtype b, Dtype* r) {
  CHECK_GE(n, 0);
  CHECK(r);
  CHECK_LE(a, b);
  boost::uniform_real<Dtype> random_distribution(a, caffe_nextafter<Dtype>(b));
  boost::variate_generator<caffe::rng_t*, boost::uniform_real<Dtype> >
      variate_generator(caffe_rng(), random_distribution);
  for (int i = 0; i < n; ++i) {
    r[i] = variate_generator();
  }
}

template
void caffe_rng_uniform<float>(const int n, const float a, const float b,
                              float* r);

template
void caffe_rng_uniform<double>(const int n, const double a, const double b,
                               double* r);

template <>
void caffe_rng_uniform<std::complex<float> >(const int n, const std::complex<float> a, const std::complex<float> b, std::complex<float>* r) {
  CHECK_GE(n, 0);
  CHECK(r);
  CHECK_EQ(std::imag(a), 0);
  CHECK_EQ(std::imag(b), 0);
  CHECK_LE(std::real(a), std::real(b));
  boost::uniform_real<float> random_distribution(std::real(a), caffe_nextafter<float>(std::real(b)));
  boost::variate_generator<caffe::rng_t*, boost::uniform_real<float> >
      variate_generator(caffe_rng(), random_distribution);
  for (int i = 0; i < n; ++i) {
    r[i] = variate_generator();
  }
}

template <>
void caffe_rng_uniform<std::complex<double> >(const int n, const std::complex<double> a, const std::complex<double> b, std::complex<double>* r) {
  CHECK_GE(n, 0);
  CHECK(r);
  CHECK_EQ(std::imag(a), 0);
  CHECK_EQ(std::imag(b), 0);
  CHECK_LE(std::real(a), std::real(b));
  boost::uniform_real<double> random_distribution(std::real(a), caffe_nextafter<double>(std::real(b)));
  boost::variate_generator<caffe::rng_t*, boost::uniform_real<double> >
      variate_generator(caffe_rng(), random_distribution);
  for (int i = 0; i < n; ++i) {
    r[i] = variate_generator();
  }
}

template <typename Dtype>
void caffe_rng_gaussian(const int n, const Dtype mu,
                        const Dtype sigma, Dtype* r) {
  CHECK_GE(n, 0);
  CHECK(r);
  CHECK_GT(sigma, 0);
  boost::normal_distribution<Dtype> random_distribution(mu, sigma);
  boost::variate_generator<caffe::rng_t*, boost::normal_distribution<Dtype> >
      variate_generator(caffe_rng(), random_distribution);
  for (int i = 0; i < n; ++i) {
    r[i] = variate_generator();
  }
}

template
void caffe_rng_gaussian<float>(const int n, const float mu,
                               const float sigma, float* r);

template
void caffe_rng_gaussian<double>(const int n, const double mu,
                                const double sigma, double* r);

template <>
void caffe_rng_gaussian<std::complex<float> >(const int n, const std::complex<float> mu,
                               const std::complex<float> sigma, std::complex<float>* r) {
  CHECK_GE(n, 0);
  CHECK(r);
  CHECK_GT(std::real(sigma), 0);
  CHECK_EQ(std::imag(sigma), 0);

  boost::normal_distribution<float> random_distribution_real(std::real(mu), std::real(sigma));
  boost::variate_generator<caffe::rng_t*, boost::normal_distribution<float> >
      variate_generator_real(caffe_rng(), random_distribution_real);

  boost::normal_distribution<float> random_distribution_imag(std::imag(mu), std::real(sigma));
  boost::variate_generator<caffe::rng_t*, boost::normal_distribution<float> >
      variate_generator_imag(caffe_rng(), random_distribution_imag);

  for (int i = 0; i < n; ++i) {
    r[i] = std::complex<float>(variate_generator_real(), variate_generator_imag());
  }
}

template <>
void caffe_rng_gaussian<std::complex<double> >(const int n, const std::complex<double> mu,
                               const std::complex<double> sigma, std::complex<double>* r) {
  CHECK_GE(n, 0);
  CHECK(r);
  CHECK_GT(std::real(sigma), 0);
  CHECK_EQ(std::imag(sigma), 0);

  boost::normal_distribution<double> random_distribution_real(std::real(mu), std::real(sigma));
  boost::variate_generator<caffe::rng_t*, boost::normal_distribution<double> >
      variate_generator_real(caffe_rng(), random_distribution_real);

  boost::normal_distribution<double> random_distribution_imag(std::imag(mu), std::real(sigma));
  boost::variate_generator<caffe::rng_t*, boost::normal_distribution<double> >
      variate_generator_imag(caffe_rng(), random_distribution_imag);

  for (int i = 0; i < n; ++i) {
    r[i] = std::complex<double>(variate_generator_real(), variate_generator_imag());
  }
}


template <typename Dtype>
void caffe_rng_bernoulli(const int n, const Dtype p, int* r) {
  CHECK_GE(n, 0);
  CHECK(r);
  CHECK_GE(p, 0);
  CHECK_LE(p, 1);
  boost::bernoulli_distribution<Dtype> random_distribution(p);
  boost::variate_generator<caffe::rng_t*, boost::bernoulli_distribution<Dtype> >
      variate_generator(caffe_rng(), random_distribution);
  for (int i = 0; i < n; ++i) {
    r[i] = variate_generator();
  }
}

template
void caffe_rng_bernoulli<double>(const int n, const double p, int* r);

template
void caffe_rng_bernoulli<float>(const int n, const float p, int* r);

template <>
void caffe_rng_bernoulli<std::complex<float> >(const int n, const std::complex<float> p, int* r) {
  CHECK_EQ(std::imag(p), 0);

  caffe_rng_bernoulli<float>(n, std::real(p), r);
}

template <>
void caffe_rng_bernoulli<std::complex<double> >(const int n, const std::complex<double> p, int* r) {
  CHECK_EQ(std::imag(p), 0);

  caffe_rng_bernoulli<double>(n, std::real(p), r);
}


template <typename Dtype>
void caffe_rng_bernoulli(const int n, const Dtype p, unsigned int* r) {
  CHECK_GE(n, 0);
  CHECK(r);
  CHECK_GE(p, 0);
  CHECK_LE(p, 1);
  boost::bernoulli_distribution<Dtype> random_distribution(p);
  boost::variate_generator<caffe::rng_t*, boost::bernoulli_distribution<Dtype> >
      variate_generator(caffe_rng(), random_distribution);
  for (int i = 0; i < n; ++i) {
    r[i] = static_cast<unsigned int>(variate_generator());
  }
}

template
void caffe_rng_bernoulli<double>(const int n, const double p, unsigned int* r);

template
void caffe_rng_bernoulli<float>(const int n, const float p, unsigned int* r);

template <>
void caffe_rng_bernoulli<std::complex<float> >(const int n, const std::complex<float> p, unsigned int* r) {
  CHECK_EQ(std::imag(p), 0);

  caffe_rng_bernoulli<float>(n, std::real(p), r);
}

template <>
void caffe_rng_bernoulli<std::complex<double> >(const int n, const std::complex<double> p, unsigned int* r) {
  CHECK_EQ(std::imag(p), 0);

  caffe_rng_bernoulli<double>(n, std::real(p), r);
}

template <>
float caffe_cpu_strided_dot<float>(const int n, const float* x, const int incx,
    const float* y, const int incy, const bool conj_x) {
  return cblas_sdot(n, x, incx, y, incy);
}

template <>
double caffe_cpu_strided_dot<double>(const int n, const double* x,
    const int incx, const double* y, const int incy, const bool conj_x) {
  return cblas_ddot(n, x, incx, y, incy);
}

template <>
std::complex<float> caffe_cpu_strided_dot<std::complex<float> >(const int n, const std::complex<float>* x, const int incx,
    const std::complex<float>* y, const int incy, const bool conj_x) {
  std::complex<float> result;
  NOT_IMPLEMENTED; // TODO Check proper result cast
  if(conj_x) {
    cblas_cdotc_sub(n, reinterpret_cast<const float*>(x), incx, reinterpret_cast<const float*>(y), incy,
    		reinterpret_cast<float __complex__*>(&result));
  }
  else {
    cblas_cdotu_sub(n, reinterpret_cast<const float*>(x), incx, reinterpret_cast<const float*>(y), incy,
    		reinterpret_cast<float __complex__*>(&result));
  }
  return result;
}

template <>
std::complex<double> caffe_cpu_strided_dot<std::complex<double> >(const int n, const std::complex<double>* x, const int incx,
    const std::complex<double>* y, const int incy, const bool conj_x) {
  std::complex<double> result;
  NOT_IMPLEMENTED; // TODO Check proper result cast
  if(conj_x) {
    cblas_zdotc_sub(n, reinterpret_cast<const double*>(x), incx, reinterpret_cast<const double*>(y), incy,
			reinterpret_cast<double __complex__*>(&result));
  }
  else {
    cblas_zdotu_sub(n, reinterpret_cast<const double*>(x), incx, reinterpret_cast<const double*>(y), incy,
    		reinterpret_cast<double __complex__*>(&result));
  }
  return result;
}

template <typename Dtype>
Dtype caffe_cpu_dot(const int n, const Dtype* x, const Dtype* y, const bool conj_x) {
  return caffe_cpu_strided_dot(n, x, 1, y, 1, conj_x);
}

template
float caffe_cpu_dot<float>(const int n, const float* x, const float* y, const bool conj_x);

template
double caffe_cpu_dot<double>(const int n, const double* x, const double* y, const bool conj_x);

template
std::complex<float> caffe_cpu_dot<std::complex<float> >(const int n, const std::complex<float>* x, const std::complex<float>* y,
    const bool conj_x);

template
std::complex<double> caffe_cpu_dot<std::complex<double> >(const int n, const std::complex<double>* x, const std::complex<double>* y,
    const bool conj_x);

template <>
float caffe_cpu_asum<float>(const int n, const float* x) {
  return cblas_sasum(n, x, 1);
}

template <>
double caffe_cpu_asum<double>(const int n, const double* x) {
  return cblas_dasum(n, x, 1);
}

template <>
std::complex<float> caffe_cpu_asum<std::complex<float> >(const int n, const std::complex<float>* x) {
  std::complex<float> result = cblas_scasum(n, reinterpret_cast<const float*>(x), 1);
  return result;
}

template <>
std::complex<double> caffe_cpu_asum<std::complex<double> >(const int n, const std::complex<double>* x) {
  std::complex<double> result = cblas_dzasum(n, reinterpret_cast<const double*>(x), 1);
  return result;
}

template <>
void caffe_cpu_scale<float>(const int n, const float alpha, const float *x,
                            float* y) {
  cblas_scopy(n, x, 1, y, 1);
  cblas_sscal(n, alpha, y, 1);
}

template <>
void caffe_cpu_scale<double>(const int n, const double alpha, const double *x,
                             double* y) {
  cblas_dcopy(n, x, 1, y, 1);
  cblas_dscal(n, alpha, y, 1);
}

template <>
void caffe_cpu_scale<std::complex<float> >(const int n, const std::complex<float> alpha, const std::complex<float> *x,
                                           std::complex<float>* y) {
  cblas_ccopy(n, reinterpret_cast<const float*>(x), 1, reinterpret_cast<float*>(y), 1);
  cblas_cscal(n, reinterpret_cast<const float*>(&alpha), reinterpret_cast<float*>(y), 1);
}

template <>
void caffe_cpu_scale<std::complex<double> >(const int n, const std::complex<double> alpha, const std::complex<double> *x,
                                           std::complex<double>* y) {
  cblas_zcopy(n, reinterpret_cast<const double*>(x), 1, reinterpret_cast<double*>(y), 1);
  cblas_zscal(n, reinterpret_cast<const double*>(&alpha), reinterpret_cast<double*>(y), 1);
}

}  // namespace caffe
