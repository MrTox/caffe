#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layers/complex_pooling_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype, typename ComplexDtype>
__global__ void ComplexMaxPoolForward(const int nthreads,
    const ComplexDtype* const bottom_data, const Dtype* const bottom_abs_data, const int num, const int channels,
    const int height, const int width, const int pooled_height,
    const int pooled_width, const int kernel_h, const int kernel_w,
    const int stride_h, const int stride_w, const int pad_h, const int pad_w,
    ComplexDtype* const top_data, int* mask, ComplexDtype* top_mask) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    const int pw = index % pooled_width;
    const int ph = (index / pooled_width) % pooled_height;
    const int c = (index / pooled_width / pooled_height) % channels;
    const int n = index / pooled_width / pooled_height / channels;
    int hstart = ph * stride_h - pad_h;
    int wstart = pw * stride_w - pad_w;
    const int hend = min(hstart + kernel_h, height);
    const int wend = min(wstart + kernel_w, width);
    hstart = max(hstart, 0);
    wstart = max(wstart, 0);
    int maxidx = -1;
    Dtype maxval = -1;
    const ComplexDtype* const bottom_slice =
        bottom_data + (n * channels + c) * height * width;
    const Dtype* const bottom_abs_slice =
        bottom_abs_data + (n * channels + c) * height * width;
    for (int h = hstart; h < hend; ++h) {
      for (int w = wstart; w < wend; ++w) {
        if (bottom_abs_slice[h * width + w] > maxval) {
          maxidx = h * width + w;
          maxval = bottom_abs_slice[h * width + w];
        }
      }
    }
    top_data[index] = bottom_slice[maxidx];
    if (mask) {
      mask[index] = maxidx;
    } else {
      top_mask[index].x = maxidx;
      top_mask[index].y = 0;
    }
  }
}

__global__ void ComplexAvePoolForward(const int nthreads,
    const cuComplex* const bottom_data, const int num, const int channels,
    const int height, const int width, const int pooled_height,
    const int pooled_width, const int kernel_h, const int kernel_w,
    const int stride_h, const int stride_w, const int pad_h, const int pad_w,
    cuComplex* const top_data) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    const int pw = index % pooled_width;
    const int ph = (index / pooled_width) % pooled_height;
    const int c = (index / pooled_width / pooled_height) % channels;
    const int n = index / pooled_width / pooled_height / channels;
    int hstart = ph * stride_h - pad_h;
    int wstart = pw * stride_w - pad_w;
    int hend = min(hstart + kernel_h, height + pad_h);
    int wend = min(wstart + kernel_w, width + pad_w);
    cuComplex pool_size = make_cuComplex((hend - hstart) * (wend - wstart), 0);
    hstart = max(hstart, 0);
    wstart = max(wstart, 0);
    hend = min(hend, height);
    wend = min(wend, width);
    cuComplex aveval = make_cuComplex(0,0);
    const cuComplex* const bottom_slice =
        bottom_data + (n * channels + c) * height * width;
    for (int h = hstart; h < hend; ++h) {
      for (int w = wstart; w < wend; ++w) {
        aveval = cuCaddf(aveval, bottom_slice[h * width + w]);
      }
    }
    top_data[index] = cuCdivf(aveval, pool_size);
  }
}

__global__ void ComplexAvePoolForward(const int nthreads,
    const cuDoubleComplex* const bottom_data, const int num, const int channels,
    const int height, const int width, const int pooled_height,
    const int pooled_width, const int kernel_h, const int kernel_w,
    const int stride_h, const int stride_w, const int pad_h, const int pad_w,
    cuDoubleComplex* const top_data) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    const int pw = index % pooled_width;
    const int ph = (index / pooled_width) % pooled_height;
    const int c = (index / pooled_width / pooled_height) % channels;
    const int n = index / pooled_width / pooled_height / channels;
    int hstart = ph * stride_h - pad_h;
    int wstart = pw * stride_w - pad_w;
    int hend = min(hstart + kernel_h, height + pad_h);
    int wend = min(wstart + kernel_w, width + pad_w);
    cuDoubleComplex pool_size = make_cuDoubleComplex((hend - hstart) * (wend - wstart), 0);
    hstart = max(hstart, 0);
    wstart = max(wstart, 0);
    hend = min(hend, height);
    wend = min(wend, width);
    cuDoubleComplex aveval = make_cuDoubleComplex(0,0);
    const cuDoubleComplex* const bottom_slice =
        bottom_data + (n * channels + c) * height * width;
    for (int h = hstart; h < hend; ++h) {
      for (int w = wstart; w < wend; ++w) {
        aveval = cuCadd(aveval, bottom_slice[h * width + w]);
      }
    }
    top_data[index] = cuCdiv(aveval, pool_size);
  }
}

__global__ void ComplexStoPoolForwardTrain(const int nthreads,
    const cuComplex* const bottom_data,
    const int num, const int channels, const int height,
    const int width, const int pooled_height, const int pooled_width,
    const int kernel_h, const int kernel_w, const int stride_h,
    const int stride_w, cuComplex* const rand_idx, cuComplex* const top_data) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    const int pw = index % pooled_width;
    const int ph = (index / pooled_width) % pooled_height;
    const int c = (index / pooled_width / pooled_height) % channels;
    const int n = index / pooled_width / pooled_height / channels;
    const int hstart = ph * stride_h;
    const int hend = min(hstart + kernel_h, height);
    const int wstart = pw * stride_w;
    const int wend = min(wstart + kernel_w, width);
    cuComplex cumsum = make_cuComplex(0,0);
    const cuComplex* const bottom_slice =
        bottom_data + (n * channels + c) * height * width;
    // First pass: get sum
    for (int h = hstart; h < hend; ++h) {
      for (int w = wstart; w < wend; ++w) {
        cumsum = cuCaddf(cumsum, bottom_slice[h * width + w]);
      }
    }
    const float thres = cuCabsf(cuCmulf(rand_idx[index], cumsum));
    // Second pass: get value, and set index.
    cumsum.x = 0;
    cumsum.y = 0;
    for (int h = hstart; h < hend; ++h) {
      for (int w = wstart; w < wend; ++w) {
        cumsum = cuCaddf(cumsum, bottom_slice[h * width + w]);
        if (cuCabsf(cumsum) >= thres) {
          rand_idx[index].x = ((n * channels + c) * height + h) * width + w;
          top_data[index] = bottom_slice[h * width + w];
          return;
        }
      }
    }
  }
}

__global__ void ComplexStoPoolForwardTrain(const int nthreads,
    const cuDoubleComplex* const bottom_data,
    const int num, const int channels, const int height,
    const int width, const int pooled_height, const int pooled_width,
    const int kernel_h, const int kernel_w, const int stride_h,
    const int stride_w, cuDoubleComplex* const rand_idx, cuDoubleComplex* const top_data) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    const int pw = index % pooled_width;
    const int ph = (index / pooled_width) % pooled_height;
    const int c = (index / pooled_width / pooled_height) % channels;
    const int n = index / pooled_width / pooled_height / channels;
    const int hstart = ph * stride_h;
    const int hend = min(hstart + kernel_h, height);
    const int wstart = pw * stride_w;
    const int wend = min(wstart + kernel_w, width);
    cuDoubleComplex cumsum = make_cuDoubleComplex(0,0);
    const cuDoubleComplex* const bottom_slice =
        bottom_data + (n * channels + c) * height * width;
    // First pass: get sum
    for (int h = hstart; h < hend; ++h) {
      for (int w = wstart; w < wend; ++w) {
        cumsum = cuCadd(cumsum, bottom_slice[h * width + w]);
      }
    }
    const double thres = cuCabs(cuCmul(rand_idx[index], cumsum));
    // Second pass: get value, and set index.
    cumsum.x = 0;
    cumsum.y = 0;
    for (int h = hstart; h < hend; ++h) {
      for (int w = wstart; w < wend; ++w) {
        cumsum = cuCadd(cumsum, bottom_slice[h * width + w]);
        if (cuCabs(cumsum) >= thres) {
          rand_idx[index].x = ((n * channels + c) * height + h) * width + w;
          top_data[index] = bottom_slice[h * width + w];
          return;
        }
      }
    }
  }
}


__global__ void ComplexStoPoolForwardTest(const int nthreads,
    const cuComplex* const bottom_data,
    const int num, const int channels, const int height,
    const int width, const int pooled_height, const int pooled_width,
    const int kernel_h, const int kernel_w, const int stride_h,
    const int stride_w, cuComplex* const top_data) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    const int pw = index % pooled_width;
    const int ph = (index / pooled_width) % pooled_height;
    const int c = (index / pooled_width / pooled_height) % channels;
    const int n = index / pooled_width / pooled_height / channels;
    const int hstart = ph * stride_h;
    const int hend = min(hstart + kernel_h, height);
    const int wstart = pw * stride_w;
    const int wend = min(wstart + kernel_w, width);
    // We set cumsum to be 0 to avoid divide-by-zero problems
    cuComplex cumsum = make_cuComplex(FLT_MIN,0);
    cuComplex cumvalues = make_cuComplex(0,0);
    const cuComplex* const bottom_slice =
        bottom_data + (n * channels + c) * height * width;
    // First pass: get sum
    for (int h = hstart; h < hend; ++h) {
      for (int w = wstart; w < wend; ++w) {
        cumsum = cuCaddf(cumsum, bottom_slice[h * width + w]);
        cumvalues = cuCaddf(cumvalues, cuCmulf(bottom_slice[h * width + w], bottom_slice[h * width + w]));
      }
    }
    top_data[index] = cuCdivf(cumvalues, cumsum);
  }
}

__global__ void ComplexStoPoolForwardTest(const int nthreads,
    const cuDoubleComplex* const bottom_data,
    const int num, const int channels, const int height,
    const int width, const int pooled_height, const int pooled_width,
    const int kernel_h, const int kernel_w, const int stride_h,
    const int stride_w, cuDoubleComplex* const top_data) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    const int pw = index % pooled_width;
    const int ph = (index / pooled_width) % pooled_height;
    const int c = (index / pooled_width / pooled_height) % channels;
    const int n = index / pooled_width / pooled_height / channels;
    const int hstart = ph * stride_h;
    const int hend = min(hstart + kernel_h, height);
    const int wstart = pw * stride_w;
    const int wend = min(wstart + kernel_w, width);
    // We set cumsum to be 0 to avoid divide-by-zero problems
    cuDoubleComplex cumsum = make_cuDoubleComplex(FLT_MIN,0);
    cuDoubleComplex cumvalues = make_cuDoubleComplex(0,0);
    const cuDoubleComplex* const bottom_slice =
        bottom_data + (n * channels + c) * height * width;
    // First pass: get sum
    for (int h = hstart; h < hend; ++h) {
      for (int w = wstart; w < wend; ++w) {
        cumsum = cuCadd(cumsum, bottom_slice[h * width + w]);
        cumvalues = cuCadd(cumvalues, cuCmul(bottom_slice[h * width + w], bottom_slice[h * width + w]));
      }
    }
    top_data[index] = cuCdiv(cumvalues, cumsum);
  }
}

template <>
void ComplexPoolingLayer<float>::Forward_gpu(const vector<Blob<float>*>& bottom,
      const vector<Blob<float>*>& top) {
  const std::complex<float>* bottom_data = this->RealToComplexBottomData_gpu(bottom, 0);
  std::complex<float>* top_data = this->RealToComplexTopData_mutable_gpu(top, 0);
  int count = top[0]->count()/2;
  // We'll output the mask to top[1] if it's of size >1.
  const bool use_top_mask = top.size() > 1;
  int* mask = NULL;
  std::complex<float>* top_mask = NULL;
  switch (this->layer_param_.pooling_param().pool()) {
  case PoolingParameter_PoolMethod_MAX:
  {
    if (use_top_mask) {
      top_mask = this->RealToComplexTopData_mutable_gpu(top, 1);
    } else {
      mask = max_idx_.mutable_gpu_data();
    }

    float* bottom_abs_data = bottom_abs_.mutable_gpu_data();
    caffe_gpu_abs(count, bottom_data, bottom_abs_data);

    // NOLINT_NEXT_LINE(whitespace/operators)
    ComplexMaxPoolForward<float, cuComplex><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
        count, (const cuComplex*)bottom_data, bottom_abs_data, bottom[0]->shape(0), channels_,
        height_, width_, pooled_height_, pooled_width_, kernel_h_,
        kernel_w_, stride_h_, stride_w_, pad_h_, pad_w_, (cuComplex*)top_data,
        mask, (cuComplex*)top_mask);
    this->SyncComplexTopData_gpu(top, 0);
    if (use_top_mask) {
      this->SyncComplexTopData_gpu(top, 1);
    }
    break;
  }
  case PoolingParameter_PoolMethod_AVE:
    // NOLINT_NEXT_LINE(whitespace/operators)
    ComplexAvePoolForward<<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
        count, (const cuComplex*)bottom_data, bottom[0]->num(), channels_,
        height_, width_, pooled_height_, pooled_width_, kernel_h_,
        kernel_w_, stride_h_, stride_w_, pad_h_, pad_w_, (cuComplex*)top_data);
    this->SyncComplexTopData_gpu(top, 0);
    break;
  case PoolingParameter_PoolMethod_STOCHASTIC:
    if (this->phase_ == TRAIN) {
      // We need to create the random index as well.
      std::complex<float>* rand_idx_data = this->RealToComplex_mutable_gpu(rand_idx_.mutable_gpu_data());
      caffe_gpu_rng_uniform(count, std::complex<float>(0), std::complex<float>(1),
                            rand_idx_data);
      // NOLINT_NEXT_LINE(whitespace/operators)
      ComplexStoPoolForwardTrain<<<CAFFE_GET_BLOCKS(count),
                                   CAFFE_CUDA_NUM_THREADS>>>(
          count, (const cuComplex*)bottom_data, bottom[0]->num(), channels_,
          height_, width_, pooled_height_, pooled_width_, kernel_h_,
          kernel_w_, stride_h_, stride_w_,
          (cuComplex*)rand_idx_data, (cuComplex*)top_data);
      this->SyncComplex_gpu(rand_idx_data, rand_idx_.mutable_gpu_data());
    } else {
      // NOLINT_NEXT_LINE(whitespace/operators)
      ComplexStoPoolForwardTest<<<CAFFE_GET_BLOCKS(count),
                                  CAFFE_CUDA_NUM_THREADS>>>(
          count, (const cuComplex*)bottom_data, bottom[0]->num(), channels_,
          height_, width_, pooled_height_, pooled_width_, kernel_h_,
          kernel_w_, stride_h_, stride_w_, (cuComplex*)top_data);
    }
    this->SyncComplexTopData_gpu(top, 0);
    break;
  default:
    LOG(FATAL) << "Unknown pooling method.";
  }
  CUDA_POST_KERNEL_CHECK;
}

template <>
void ComplexPoolingLayer<double>::Forward_gpu(const vector<Blob<double>*>& bottom,
      const vector<Blob<double>*>& top) {
  const std::complex<double>* bottom_data = this->RealToComplexBottomData_gpu(bottom, 0);
  std::complex<double>* top_data = this->RealToComplexTopData_mutable_gpu(top, 0);
  int count = top[0]->count()/2;
  // We'll output the mask to top[1] if it's of size >1.
  const bool use_top_mask = top.size() > 1;
  int* mask = NULL;
  std::complex<double>* top_mask = NULL;
  switch (this->layer_param_.pooling_param().pool()) {
  case PoolingParameter_PoolMethod_MAX:
  {
    if (use_top_mask) {
      top_mask = this->RealToComplexTopData_mutable_gpu(top, 1);
    } else {
      mask = max_idx_.mutable_gpu_data();
    }

    double* bottom_abs_data = bottom_abs_.mutable_gpu_data();
    caffe_gpu_abs(count, bottom_data, bottom_abs_data);

    // NOLINT_NEXT_LINE(whitespace/operators)
    ComplexMaxPoolForward<double, cuDoubleComplex><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
        count, (const cuDoubleComplex*)bottom_data, bottom_abs_data, bottom[0]->shape(0), channels_,
        height_, width_, pooled_height_, pooled_width_, kernel_h_,
        kernel_w_, stride_h_, stride_w_, pad_h_, pad_w_, (cuDoubleComplex*)top_data,
        mask, (cuDoubleComplex*)top_mask);
    this->SyncComplexTopData_gpu(top, 0);
    if (use_top_mask) {
      this->SyncComplexTopData_gpu(top, 1);
    }
    break;
  }
  case PoolingParameter_PoolMethod_AVE:
    // NOLINT_NEXT_LINE(whitespace/operators)
    ComplexAvePoolForward<<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
        count, (const cuDoubleComplex*)bottom_data, bottom[0]->num(), channels_,
        height_, width_, pooled_height_, pooled_width_, kernel_h_,
        kernel_w_, stride_h_, stride_w_, pad_h_, pad_w_, (cuDoubleComplex*)top_data);
    this->SyncComplexTopData_gpu(top, 0);
    break;
  case PoolingParameter_PoolMethod_STOCHASTIC:
    if (this->phase_ == TRAIN) {
      // We need to create the random index as well.
      std::complex<double>* rand_idx_data = this->RealToComplex_mutable_gpu(rand_idx_.mutable_gpu_data());
      caffe_gpu_rng_uniform(count, std::complex<double>(0), std::complex<double>(1),
                            rand_idx_data);
      // NOLINT_NEXT_LINE(whitespace/operators)
      ComplexStoPoolForwardTrain<<<CAFFE_GET_BLOCKS(count),
                                   CAFFE_CUDA_NUM_THREADS>>>(
          count, (const cuDoubleComplex*)bottom_data, bottom[0]->num(), channels_,
          height_, width_, pooled_height_, pooled_width_, kernel_h_,
          kernel_w_, stride_h_, stride_w_,
          (cuDoubleComplex*)rand_idx_data, (cuDoubleComplex*)top_data);
      this->SyncComplex_gpu(rand_idx_data, rand_idx_.mutable_gpu_data());
    } else {
      // NOLINT_NEXT_LINE(whitespace/operators)
      ComplexStoPoolForwardTest<<<CAFFE_GET_BLOCKS(count),
                                  CAFFE_CUDA_NUM_THREADS>>>(
          count, (const cuDoubleComplex*)bottom_data, bottom[0]->num(), channels_,
          height_, width_, pooled_height_, pooled_width_, kernel_h_,
          kernel_w_, stride_h_, stride_w_, (cuDoubleComplex*)top_data);
    }
    this->SyncComplexTopData_gpu(top, 0);
    break;
  default:
    LOG(FATAL) << "Unknown pooling method.";
  }
  CUDA_POST_KERNEL_CHECK;
}


__global__ void ComplexMaxPoolBackward(const int nthreads, const cuComplex* const top_diff,
    const int* const mask, const cuComplex* const top_mask, const int num,
    const int channels, const int height, const int width,
    const int pooled_height, const int pooled_width, const int kernel_h,
    const int kernel_w, const int stride_h, const int stride_w, const int pad_h,
    const int pad_w, cuComplex* const bottom_diff) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    // find out the local index
    // find out the local offset
    const int w = index % width;
    const int h = (index / width) % height;
    const int c = (index / width / height) % channels;
    const int n = index / width / height / channels;
    const int phstart =
         (h + pad_h < kernel_h) ? 0 : (h + pad_h - kernel_h) / stride_h + 1;
    const int phend = min((h + pad_h) / stride_h + 1, pooled_height);
    const int pwstart =
         (w + pad_w < kernel_w) ? 0 : (w + pad_w - kernel_w) / stride_w + 1;
    const int pwend = min((w + pad_w) / stride_w + 1, pooled_width);
    cuComplex gradient = make_cuComplex(0,0);
    const int offset = (n * channels + c) * pooled_height * pooled_width;
    const cuComplex* const top_diff_slice = top_diff + offset;
    if (mask) {
      const int* const mask_slice = mask + offset;
      for (int ph = phstart; ph < phend; ++ph) {
        for (int pw = pwstart; pw < pwend; ++pw) {
          if (mask_slice[ph * pooled_width + pw] == h * width + w) {
            gradient = cuCaddf(gradient, top_diff_slice[ph * pooled_width + pw]);
          }
        }
      }
    } else {
      const cuComplex* const top_mask_slice = top_mask + offset;
      for (int ph = phstart; ph < phend; ++ph) {
        for (int pw = pwstart; pw < pwend; ++pw) {
          if (top_mask_slice[ph * pooled_width + pw].x == h * width + w) {
            gradient = cuCaddf(gradient, top_diff_slice[ph * pooled_width + pw]);
          }
        }
      }
    }
    bottom_diff[index] = gradient;
  }
}

__global__ void ComplexMaxPoolBackward(const int nthreads, const cuDoubleComplex* const top_diff,
    const int* const mask, const cuDoubleComplex* const top_mask, const int num,
    const int channels, const int height, const int width,
    const int pooled_height, const int pooled_width, const int kernel_h,
    const int kernel_w, const int stride_h, const int stride_w, const int pad_h,
    const int pad_w, cuDoubleComplex* const bottom_diff) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    // find out the local index
    // find out the local offset
    const int w = index % width;
    const int h = (index / width) % height;
    const int c = (index / width / height) % channels;
    const int n = index / width / height / channels;
    const int phstart =
         (h + pad_h < kernel_h) ? 0 : (h + pad_h - kernel_h) / stride_h + 1;
    const int phend = min((h + pad_h) / stride_h + 1, pooled_height);
    const int pwstart =
         (w + pad_w < kernel_w) ? 0 : (w + pad_w - kernel_w) / stride_w + 1;
    const int pwend = min((w + pad_w) / stride_w + 1, pooled_width);
    cuDoubleComplex gradient = make_cuDoubleComplex(0,0);
    const int offset = (n * channels + c) * pooled_height * pooled_width;
    const cuDoubleComplex* const top_diff_slice = top_diff + offset;
    if (mask) {
      const int* const mask_slice = mask + offset;
      for (int ph = phstart; ph < phend; ++ph) {
        for (int pw = pwstart; pw < pwend; ++pw) {
          if (mask_slice[ph * pooled_width + pw] == h * width + w) {
            gradient = cuCadd(gradient, top_diff_slice[ph * pooled_width + pw]);
          }
        }
      }
    } else {
      const cuDoubleComplex* const top_mask_slice = top_mask + offset;
      for (int ph = phstart; ph < phend; ++ph) {
        for (int pw = pwstart; pw < pwend; ++pw) {
          if (top_mask_slice[ph * pooled_width + pw].x == h * width + w) {
            gradient = cuCadd(gradient, top_diff_slice[ph * pooled_width + pw]);
          }
        }
      }
    }
    bottom_diff[index] = gradient;
  }
}

__global__ void ComplexAvePoolBackward(const int nthreads, const cuComplex* const top_diff,
    const int num, const int channels, const int height,
    const int width, const int pooled_height, const int pooled_width,
    const int kernel_h, const int kernel_w, const int stride_h,
    const int stride_w, const int pad_h, const int pad_w,
    cuComplex* const bottom_diff) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    // find out the local index
    // find out the local offset
    const int w = index % width + pad_w;
    const int h = (index / width) % height + pad_h;
    const int c = (index / width / height) % channels;
    const int n = index / width / height / channels;
    const int phstart = (h < kernel_h) ? 0 : (h - kernel_h) / stride_h + 1;
    const int phend = min(h / stride_h + 1, pooled_height);
    const int pwstart = (w < kernel_w) ? 0 : (w - kernel_w) / stride_w + 1;
    const int pwend = min(w / stride_w + 1, pooled_width);
    cuComplex gradient = make_cuComplex(0,0);
    cuComplex pool_size = make_cuComplex(0,0);
    const cuComplex* const top_diff_slice =
        top_diff + (n * channels + c) * pooled_height * pooled_width;
    for (int ph = phstart; ph < phend; ++ph) {
      for (int pw = pwstart; pw < pwend; ++pw) {
        // figure out the pooling size
        int hstart = ph * stride_h - pad_h;
        int wstart = pw * stride_w - pad_w;
        int hend = min(hstart + kernel_h, height + pad_h);
        int wend = min(wstart + kernel_w, width + pad_w);
        pool_size.x = (hend - hstart) * (wend - wstart);
        gradient = cuCaddf(gradient, cuCdivf(top_diff_slice[ph * pooled_width + pw], pool_size) );
      }
    }
    bottom_diff[index] = gradient;
  }
}

__global__ void ComplexAvePoolBackward(const int nthreads, const cuDoubleComplex* const top_diff,
    const int num, const int channels, const int height,
    const int width, const int pooled_height, const int pooled_width,
    const int kernel_h, const int kernel_w, const int stride_h,
    const int stride_w, const int pad_h, const int pad_w,
    cuDoubleComplex* const bottom_diff) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    // find out the local index
    // find out the local offset
    const int w = index % width + pad_w;
    const int h = (index / width) % height + pad_h;
    const int c = (index / width / height) % channels;
    const int n = index / width / height / channels;
    const int phstart = (h < kernel_h) ? 0 : (h - kernel_h) / stride_h + 1;
    const int phend = min(h / stride_h + 1, pooled_height);
    const int pwstart = (w < kernel_w) ? 0 : (w - kernel_w) / stride_w + 1;
    const int pwend = min(w / stride_w + 1, pooled_width);
    cuDoubleComplex gradient = make_cuDoubleComplex(0,0);
    cuDoubleComplex pool_size = make_cuDoubleComplex(0,0);
    const cuDoubleComplex* const top_diff_slice =
        top_diff + (n * channels + c) * pooled_height * pooled_width;
    for (int ph = phstart; ph < phend; ++ph) {
      for (int pw = pwstart; pw < pwend; ++pw) {
        // figure out the pooling size
        int hstart = ph * stride_h - pad_h;
        int wstart = pw * stride_w - pad_w;
        int hend = min(hstart + kernel_h, height + pad_h);
        int wend = min(wstart + kernel_w, width + pad_w);
        pool_size.x = (hend - hstart) * (wend - wstart);
        gradient = cuCadd(gradient, cuCdiv(top_diff_slice[ph * pooled_width + pw], pool_size) );
      }
    }
    bottom_diff[index] = gradient;
  }
}

__global__ void ComplexStoPoolBackward(const int nthreads,
    const cuComplex* const rand_idx, const cuComplex* const top_diff,
    const int num, const int channels, const int height,
    const int width, const int pooled_height, const int pooled_width,
    const int kernel_h, const int kernel_w, const int stride_h,
    const int stride_w, cuComplex* const bottom_diff) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    // find out the local index
    // find out the local offset
    const int w = index % width;
    const int h = (index / width) % height;
    const int c = (index / width / height) % channels;
    const int n = index / width / height / channels;
    const int phstart = (h < kernel_h) ? 0 : (h - kernel_h) / stride_h + 1;
    const int phend = min(h / stride_h + 1, pooled_height);
    const int pwstart = (w < kernel_w) ? 0 : (w - kernel_w) / stride_w + 1;
    const int pwend = min(w / stride_w + 1, pooled_width);
    cuComplex gradient = make_cuComplex(0,0);
    const cuComplex* const rand_idx_slice =
        rand_idx + (n * channels + c) * pooled_height * pooled_width;
    const cuComplex* const top_diff_slice =
        top_diff + (n * channels + c) * pooled_height * pooled_width;
    for (int ph = phstart; ph < phend; ++ph) {
      for (int pw = pwstart; pw < pwend; ++pw) {
        if (index == static_cast<int>(rand_idx_slice[ph * pooled_width + pw].x) ) {
          gradient = cuCaddf(gradient, top_diff_slice[ph * pooled_width + pw]);
        }
      }
    }
    bottom_diff[index] = gradient;
  }
}

__global__ void ComplexStoPoolBackward(const int nthreads,
    const cuDoubleComplex* const rand_idx, const cuDoubleComplex* const top_diff,
    const int num, const int channels, const int height,
    const int width, const int pooled_height, const int pooled_width,
    const int kernel_h, const int kernel_w, const int stride_h,
    const int stride_w, cuDoubleComplex* const bottom_diff) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    // find out the local index
    // find out the local offset
    const int w = index % width;
    const int h = (index / width) % height;
    const int c = (index / width / height) % channels;
    const int n = index / width / height / channels;
    const int phstart = (h < kernel_h) ? 0 : (h - kernel_h) / stride_h + 1;
    const int phend = min(h / stride_h + 1, pooled_height);
    const int pwstart = (w < kernel_w) ? 0 : (w - kernel_w) / stride_w + 1;
    const int pwend = min(w / stride_w + 1, pooled_width);
    cuDoubleComplex gradient = make_cuDoubleComplex(0,0);
    const cuDoubleComplex* const rand_idx_slice =
        rand_idx + (n * channels + c) * pooled_height * pooled_width;
    const cuDoubleComplex* const top_diff_slice =
        top_diff + (n * channels + c) * pooled_height * pooled_width;
    for (int ph = phstart; ph < phend; ++ph) {
      for (int pw = pwstart; pw < pwend; ++pw) {
        if (index == static_cast<int>(rand_idx_slice[ph * pooled_width + pw].x) ) {
          gradient = cuCadd(gradient, top_diff_slice[ph * pooled_width + pw]);
        }
      }
    }
    bottom_diff[index] = gradient;
  }
}


template <>
void ComplexPoolingLayer<float>::Backward_gpu(const vector<Blob<float>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<float>*>& bottom) {
  if (!propagate_down[0]) {
    return;
  }
  const std::complex<float>* top_diff = this->RealToComplexTopDiff_gpu(top,0);
  std::complex<float>* bottom_diff = this->RealToComplexBottomDiff_mutable_gpu(bottom,0);
  const int count = bottom[0]->count()/2;
  caffe_gpu_set(count, std::complex<float>(0), bottom_diff);
  // We'll output the mask to top[1] if it's of size >1.
  const bool use_top_mask = top.size() > 1;
  const int* mask = NULL;
  const std::complex<float>* top_mask = NULL;
  switch (this->layer_param_.pooling_param().pool()) {
  case PoolingParameter_PoolMethod_MAX:
    if (use_top_mask) {
      top_mask = this->RealToComplexTopData_gpu(top,1);
    } else {
      mask = max_idx_.gpu_data();
    }
    // NOLINT_NEXT_LINE(whitespace/operators)
    ComplexMaxPoolBackward<<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
        count, (const cuComplex*)top_diff, mask, (const cuComplex*)top_mask, top[0]->shape(0), channels_,
        height_, width_, pooled_height_, pooled_width_,
        kernel_h_, kernel_w_, stride_h_, stride_w_, pad_h_, pad_w_,
        (cuComplex*)bottom_diff);
    break;
  case PoolingParameter_PoolMethod_AVE:
    // NOLINT_NEXT_LINE(whitespace/operators)
    ComplexAvePoolBackward<<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
        count, (const cuComplex*)top_diff, top[0]->shape(0), channels_,
        height_, width_, pooled_height_, pooled_width_, kernel_h_,
        kernel_w_, stride_h_, stride_w_, pad_h_, pad_w_, (cuComplex*)bottom_diff);
    break;
  case PoolingParameter_PoolMethod_STOCHASTIC:
    // NOLINT_NEXT_LINE(whitespace/operators)
    ComplexStoPoolBackward<<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
        count, (const cuComplex*)this->RealToComplex_gpu(rand_idx_.gpu_data()), (const cuComplex*)top_diff,
        top[0]->shape(0), channels_, height_, width_, pooled_height_,
        pooled_width_, kernel_h_, kernel_w_, stride_h_, stride_w_,
        (cuComplex*)bottom_diff);
    break;
  default:
    LOG(FATAL) << "Unknown pooling method.";
  }
  this->SyncComplexBottomDiff_gpu(bottom, 0);

  CUDA_POST_KERNEL_CHECK;
}

template <>
void ComplexPoolingLayer<double>::Backward_gpu(const vector<Blob<double>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<double>*>& bottom) {
  if (!propagate_down[0]) {
    return;
  }
  const std::complex<double>* top_diff = this->RealToComplexTopDiff_gpu(top,0);
  std::complex<double>* bottom_diff = this->RealToComplexBottomDiff_mutable_gpu(bottom,0);
  const int count = bottom[0]->count()/2;
  caffe_gpu_set(count, std::complex<double>(0), bottom_diff);
  // We'll output the mask to top[1] if it's of size >1.
  const bool use_top_mask = top.size() > 1;
  const int* mask = NULL;
  const std::complex<double>* top_mask = NULL;
  switch (this->layer_param_.pooling_param().pool()) {
  case PoolingParameter_PoolMethod_MAX:
    if (use_top_mask) {
      top_mask = this->RealToComplexTopData_gpu(top,1);
    } else {
      mask = max_idx_.gpu_data();
    }
    // NOLINT_NEXT_LINE(whitespace/operators)
    ComplexMaxPoolBackward<<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
        count, (const cuDoubleComplex*)top_diff, mask, (const cuDoubleComplex*)top_mask, top[0]->shape(0), channels_,
        height_, width_, pooled_height_, pooled_width_,
        kernel_h_, kernel_w_, stride_h_, stride_w_, pad_h_, pad_w_,
        (cuDoubleComplex*)bottom_diff);
    break;
  case PoolingParameter_PoolMethod_AVE:
    // NOLINT_NEXT_LINE(whitespace/operators)
    ComplexAvePoolBackward<<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
        count, (const cuDoubleComplex*)top_diff, top[0]->shape(0), channels_,
        height_, width_, pooled_height_, pooled_width_, kernel_h_,
        kernel_w_, stride_h_, stride_w_, pad_h_, pad_w_, (cuDoubleComplex*)bottom_diff);
    break;
  case PoolingParameter_PoolMethod_STOCHASTIC:
    // NOLINT_NEXT_LINE(whitespace/operators)
    ComplexStoPoolBackward<<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
        count, (const cuDoubleComplex*)this->RealToComplex_gpu(rand_idx_.gpu_data()), (const cuDoubleComplex*)top_diff,
        top[0]->shape(0), channels_, height_, width_, pooled_height_,
        pooled_width_, kernel_h_, kernel_w_, stride_h_, stride_w_,
        (cuDoubleComplex*)bottom_diff);
    break;
  default:
    LOG(FATAL) << "Unknown pooling method.";
  }
  this->SyncComplexBottomDiff_gpu(bottom, 0);

  CUDA_POST_KERNEL_CHECK;
}


INSTANTIATE_LAYER_GPU_FUNCS(ComplexPoolingLayer);


}  // namespace caffe
