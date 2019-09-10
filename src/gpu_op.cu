#include "./c_runtime_api.h"
#include <cassert>
#include <cstdio>
#include <cublas_v2.h>
#include <cuda_runtime.h>

/* TODO: Your code here */
/* all your GPU kernel code, e.g. matrix_softmax_cross_entropy_kernel */
__global__ void array_set_kernel(int n, float *arr, float value) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if(idx >= n)
    return;
  arr[idx] = value;
}


__global__ void broadcast_to_kernel(int n_in, int n_out, const float *input, float *output) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n_out)
    return;
  float val = input[idx];
  for (int i = idx; i < n_out; i += n_in)
    output[i] = val;
}


__global__ void reduce_sum_axis_zero_kernel(int n_in, int n_out, const float *input, float *output) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n_out)
    return;
  output[idx] = 0;
  for (int i = idx; i < n_in; i += n_out) {
    output[idx] += input[i];
  }
}


// longer: 2d-matrix; shorter: 1d-matrix
__global__ void matrix_add_kernel (int longer, int shorter,
                                  const float *matA,
                                  const float *matB,
                                  float *output) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= shorter)
    return;
  float val = matB[idx]; // shorter
  for (int i = idx; i < longer; i += shorter)
    output[i] = matA[i] * val;
}


__global__ void matrix_add_by_const_kernel(int n, const float *mat, float val, float *output) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n)
    return;
  output[idx] = mat[idx] + val;
}


// ElementwiseMultiply
// longer: 2d-matrix; shorter: 1d-matrixs
__global__ void matrix_mul_kernel(int longer, int shorter,
                                  const float *matA,
                                  const float *matB,
                                  float *output) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= shorter)
    return;
  float val = matB[idx];
  for (int i = idx; i < longer; i += shorter) {
    output[i] = matA[i] * val;
  }
}


__global__ void matrix_mul_by_const_kernel(int n, const float *mat, float val, float *output) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n)
    return n;
  output[idx] = mat[idx] * val;
}


__global__ void relu_kernel(int n, const float *input, float *output) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n)
    return;
  output[idx] = (input[idx] >= 0) ? input[idx] : 0.0;
}


__global__ void softmax_kernel(int nrow, int ncol, const float *input, float *output) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;  // #row
  if (idx >= nrow)
    return;
  idx_in = idx * ncol;
  float maxval = input[idx_in];
  for (int i = idx_in; i < idx_in + ncol; ++i)
    maxval = max(maxval, input[i]);
  float sum = 0;
  for (int i = idx_in; i < idx_in + ncol; ++i)
    sum += exp(input[i] - maxval);
  for (int i = idx_in; i < idx_in + ncol; ++i)
    output[i] = exp(input[i] - maxval) / sum;
}


__global__ void relu_gradient_kernel(int n, const float *input, const float *grad, float *output) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n)
    return;
  output[idx] = (input[idx] > 0.0) ? grad[idx] : 0.0;
}


// y = inputs[0], y_ = inputs[1]
// np.mean(-np.sum(y_ * np.log(softmax(y)), axis=1), keepdims=True)
__global__ void matrix_softmax_cross_entropy_kernel(int nrow, int ncol,
                                                    const float *input_a,
                                                    const float *input_b,
                                                    float *output) {
  // Dynamic shared memory, size provided at kernel launch.
  extern __shared__ float loss_per_row[];
  // Two dimensional thread blocks.
  int y = threadIdx.y * blockDim.x + threadIdx.x; // # row
  if (y >= nrow) {
    return;
  }
  input_a += y * ncol;
  input_b += y * ncol;
  float maxval = *input_a;
  // Find max for a row.
  for (int x = 1; x < ncol; ++x) {
    maxval = max(maxval, input_a[x]);
  }
  // Deduct by max for a row, and raise to exp.
  float sum = 0;
  for (int x = 0; x < ncol; ++x) {
    sum += exp(input_a[x] - maxval);
  }
  // Compute per-row loss.
  float loss = 0;
  for (int x = 0; x < ncol; ++x) {
    loss -= input_b[x] * log(exp(input_a[x] - maxval) / sum);
  }
  loss_per_row[y] = loss;
  __syncthreads();
  // Compute reduce_mean across rows.
  float mean_loss = 0;
  // Use a single thread to reduce mean across rows.
  if ((threadIdx.x == 0) && (threadIdx.y == 0)) {
    for (int i = 0; i < nrow; ++i) {
      mean_loss += loss_per_row[i];
    }
    mean_loss /= nrow;
    output[0] = mean_loss;
  }
}

int DLGpuArraySet(DLArrayHandle arr, float value) { /* TODO: Your code here */
  int n = 1;
  for (int i = 0; i < arr->ndim; ++i)
    n *= arr->shape[i];
  float *arr_data = (float*)arr->data;
  dim3 threads, blocks;
  if (n <= 1024) {
    threads.x = n;
    blocks.x = 1;
  } else {
    threads.x = 1024;
    blocks.x = (n + 1023) / 1024;
  }
  array_set_kernel<<<blocks, threads>>>(n, arr_data, value);
  return 0;
}

int DLGpuBroadcastTo(const DLArrayHandle input, DLArrayHandle output) {
  /* TODO: Your code here */
  assert(output->ndim >= input->ndim);
  int diff = output->ndim - input->ndim;
  int n_in = 1, n_out = 1;
  for (int i = 0; i < diff; ++i)
    n_out *= output->shape[i];
  for (int i = 0; i < input->ndim; ++i) {
    assert(input->shape[i] == output->shape[i + diff]);
    n_in *= input->shape[i];
    n_out *= output->shape[i];
  }

  float *input_data = (float*)input->data;
  float *output_data = (float*)output->data;
  dim3 blocks, threads;
  if (n_in <= 1024) {
    threads.x = n_in;
    blocks.x = 1;
  } else {
    threads.x = 1024;
    blocks.x = (n_in + 1023) / 1024;
  }
  broadcast_to_kernel<<<blocks, threads>>>(n_in, n_out, input_data, output_data);
  return 0;
}


int DLGpuReduceSumAxisZero(const DLArrayHandle input, DLArrayHandle output) {
  /* TODO: Your code here */
  assert(input->ndim >= output->ndim);
  int diff = input->ndim - output->ndim;
  int n_in = 1, n_out = 1;
  for (int i = 0; i < diff; ++i)
    n_in *= input->shape[i];
  for (int i = 0; i < output->ndim; ++i) {
    assert(output->shape[i] == input->shape[i + diff]);
    n_in *= output->shape[i];
    n_out *= output->shape[i];
  }
  dim3 blocks, threads;
  if (n_out <= 1024) {
    threads.x = n_out;
    blocks.x = 1;
  } else {
    threads.x = 1024;
    blocks.x = (n_out + 1023) / 1024;
  }
  const float *input_data = (const float *)input->data;
  float *output_data = (float *)output->data;
  reduce_sum_axis_zero_kernel<<<blocks, threads>>>(n_in, n_out, input_data, output_data);
  return 0;
}

int DLGpuMatrixElementwiseAdd(const DLArrayHandle matA,
                              const DLArrayHandle matB, DLArrayHandle output) {
  /* TODO: Your code here */
  if (matA->ndim < matB->ndim)
    return DLGpuMatrixElementwiseAdd(matB, matA, output);
  int diff = matA->ndim - matB->ndim;
  int n_a = 1, n_b = 1;
  for (int i = 0; i < diff; ++i)
    n_a *= matA->shape[i];
  for (int i = 0; i < matB->ndim; ++i){
    assert(matB->shape[i] == matA->shape[i + diff]);
    n_a *= matB->shape[i];
    n_b *= matA->shape[i];
  }
  dim3 blocks, threads;
  if (n_b <= 1024) {
    threads.x = n_b;
    blocks.x = 1;
  } else {
    threads.x = 1024;
    blocks.x = (n_b + 1023) / 1024;
  }
  const float *matA_data = (const float *)matA->data;
  const float *matB_data = (const float *)matB->data;
  float *output_data = (float *)output->data;
  matrix_add_kernel<<<blocks, threads>>>(n_a, n_b, matA_data, matB_data, output_data);
  return 0;
}

int DLGpuMatrixElementwiseAddByConst(const DLArrayHandle input, float val,
                                     DLArrayHandle output) {
  /* TODO: Your code here */
  assert(input->ndim == output->ndim);
  int n = 1;
  for (int i = 0; i < input->ndim; ++i) {
    assert(input->shape[i] == output->shape[i]);
    n *= input->shape[i];
  }
  dim3 blocks, threads;
  if (n <= 1024){
    threads.x = n;
    blocks.x = 1;
  } else {
    threads.x = 1024;
    blocks.x = (n + 1023) / 1024;
  }
  const float *input_data = (const float *)input->data;
  float *output_data = (float *)output->data;
  matrix_add_by_const_kernel<<<blocks, threads>>>(n, input_data, val, output_data);
  return 0;
}

int DLGpuMatrixElementwiseMultiply(const DLArrayHandle matA,
                                   const DLArrayHandle matB,
                                   DLArrayHandle output) {
  /* TODO: Your code here */
  if (matA->ndim < matB->ndim)
    return DLGpuMatrixElementwiseMultiply(matB, matA, output);
  int n_a = 1, n_b = 1;
  int diff = matA->ndim - matB->ndim;
  for (int i = 0; i < diff; ++i)
    n_a *= matA->shape[i];
  for (int i = 0; i < matB->ndim; ++i){
    assert(matB->shape[i] == matA->shape[i + diff]);
    n_a *= matB->shape[i];
    n_b *= matB->shape[i];
  }
  dim3 blocks, threads;
  if (n_b <= 1024){
    threads.x = n;
    blocks.x = 1;
  } else {
    threads.x = 1024;
    blocks.x = (n_b + 1023) / 1024;
  }
  const float *matA_data = (const float *)matA->data;
  const float *matB_data = (const float *)matB->data;
  float *output_data = (float *)output->data;
  matrix_mul_kernel<<<blocks, threads>>>(n_a, n_b, matA_data, matB_data, output_data);
  return 0;
}

int DLGpuMatrixMultiplyByConst(const DLArrayHandle input, float val,
                               DLArrayHandle output) {
  /* TODO: Your code here */
  int n = 1;
  for (int i = 0; i < input->ndim; ++i)
    n *= input->shape[i];
  dim3 blocks, threads;
  if (n <= 1024){
    threads.x = n;
    blocks.x = 1;
  } else {
    threads.x = 1024;
    blocks.x = (n + 1023) / 1024;
  }
  const float *input_data = (const float *)input->data;
  float *output_data = (float *)output->data;
  matrix_mul_by_const_kernel<<<blocks, threads>>>(n, input_data, val, output_data);
  return 0;
}

int DLGpuMatrixMultiply(const DLArrayHandle matA, bool transposeA,
                        const DLArrayHandle matB, bool transposeB,
                        DLArrayHandle matC) {
  /* TODO: Your code here */
  // Hint: use cublas
  // cublas assume matrix is column major
  assert(matA->ndim == 2);
  assert(matB->ndim == 2);
  assert(matC->ndim == 2);
  cublasHandle_t handle;
  assert(cublasCreate(&handle) == CUBLAS_STATUS_SUCCESS);
  cublasOperation_t transa = transposeA ? CUBLAS_OP_T : CUBLAS_OP_N;
  cublasOperation_t transb = transposeB ? CUBLAS_OP_T : CUBLAS_OP_N;
  int m = transposeA ? matA->shape[1] : matA->shape[0];
  int k1 = transposeA ? matA->shape[0] : matA->shape[1];
  int k2 = transposeB ? matB->shape[1] : matB->shape[0];
  int n = transposeB ? matB->shape[0] : matB->shape[1];
  assert(k1 == k2);
  assert(m == matC->shape[0]);
  assert(n == matC->shape[1]);
  const float *matA_data = (const float *)matA->data;
  const float *matB_data = (const float *)matB->data;
  float *matC_data = (float *)matC->data;
  float alpha = 1.0f, beta = 0.0f;
  DLGpuArraySet(matC, 0.0f);
  cublasSgemm(handle, transb, transa,
              n, m, k1, &alpha,
              matB_data, matB->shape[1],
              matA_data, matA->shape[1],
              &beta, matC_data, n);
  return 0;
}

int DLGpuRelu(const DLArrayHandle input, DLArrayHandle output) {
  /* TODO: Your code here */
  assert(input->ndim == 2);
  assert(output->ndim == 2);
  assert(input->shape[0] == output->shape[0] &&
        input->shape[1] == input->shape[1]);
  int n = input->shape[0] * input->shape[1];
  dim3 blocks, threads;
  if (n <= 1024){
    threads.x = n;
    blocks.x = 1;
  } else {
    threads.x = 1024;
    blocks.x = (n + 1023) / 1024;
  }
  const float *input_data = (const float *)input->data;
  float *output_data = (float *)output->data;
  relu_kernel<<<blocks, threads>>>(n, input_a, output_data);
  return 0;
}

int DLGpuReluGradient(const DLArrayHandle input, const DLArrayHandle in_grad,
                      DLArrayHandle output) {
  /* TODO: Your code here */
  assert(input->ndim == 2);
  assert(in_grad->ndim == 2);
  assert(output->ndim == 2);
  assert(input->shape[0] == in_grad->shape[0] &&
         input->shape[1] == in_grad->shape[1]);
  assert(input->shape[0] == output->shape[0] &&
         input->shape[1] == output->shape[1]);
  int n = input->shape[0] * input->shape[1];
  dim3 blocks, threads;
  if (n <= 1024) {
    threads.x = n;
    blocks.x = 1;
  } else {
    threads.x = 1024;
    blocks.x = (n + 1023) / 1024;
  }
  const float *input_data = (const float *)input->data;
  const float *grad_data = (const float *)in_grad->data;
  float *output_data = (float *)output->data;
  relu_gradient_kernel<<<blocks, threads>>>(n, input_data, grad_data, output_data);
  return 0;
}

int DLGpuSoftmax(const DLArrayHandle input, DLArrayHandle output) {
  /* TODO: Your code here */
  assert(input->ndim == 2);
  assert(output->ndim == 2);
  assert(input->shape[0] == output->shape[0] &&
         input->shape[1] == output->shape[1]);
  int nrow = input->shape[0];
  int ncol = input->shape[1];
  const float *input_data = (const float *)input->data;
  float *output_data = (float *)output->data;
  dim3 blocks, threads;
  if (nrow <= 1024) {
    threads.x = nrow;
    blocks.x = 1;
  } else {
    threads.x = 1024;
    blocks.x = (nrow + 1023) / 1024;
  }
  softmax_kernel<<<blocks, threads>>>(nrow, ncol, input_data, output_data);
  return 0;
}

int DLGpuSoftmaxCrossEntropy(const DLArrayHandle input_a,
                             const DLArrayHandle input_b,
                             DLArrayHandle output) {
  assert(input_a->ndim == 2);
  assert(input_b->ndim == 2);
  assert(output->ndim == 1);
  assert(input_a->shape[0] == input_b->shape[0] &&
         input_a->shape[1] == input_b->shape[1]);
  int nrow = input_a->shape[0];
  // Maximum x- or y-dimension of a block = 1024
  // But we need 'nrow' shared memory, and max shared memory is 48KB.
  // Conservatively allow max 16KB shared memory.
  assert(nrow <= 1024 * 4);
  int ncol = input_a->shape[1];
  const float *input_data_a = (const float *)input_a->data;
  const float *input_data_b = (const float *)input_b->data;
  float *output_data = (float *)output->data;
  dim3 threads;
  if (nrow <= 1024) {
    threads.x = nrow;
  } else {
    threads.x = 1024;
    threads.y = (nrow + 1023) / 1024;
  }
  // 1 block, each block with 'threads' number of threads with 'nrow' shared
  // memory size
  matrix_softmax_cross_entropy_kernel<<<1, threads, nrow * sizeof(float)>>>(
      nrow, ncol, input_data_a, input_data_b, output_data);
  return 0;
}
