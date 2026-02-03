#ifndef _CORR_CUDA_KERNEL
#define _CORR_CUDA_KERNEL
//#include <THC/THC.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAEvent.h>
#ifdef __cplusplus
extern "C" {
#endif

void blob_rearrange_ongpu(
    const at::Tensor& in,
    at::Tensor& out,
    int padding,
    cudaStream_t stream);

void CorrelateData_ongpu(
    const at::Tensor& rbot1,
    const at::Tensor& rbot2,
    at::Tensor& output,
    int max_displacement,
    int neighborhood_grid_radius_,
    int neighborhood_grid_width_,
    int kernel_radius_,
    int kernel_size,
    int stride1,
    int stride2,
    int corr_type_multiply,
    cudaStream_t stream);

void CorrelateDataBackward_ongpu(
    const at::Tensor& rbot1,
    const at::Tensor& rbot2,
    const at::Tensor& gradOutput,
    at::Tensor& gradInput1,
    at::Tensor& gradInput2,
    int max_displacement,
    int neighborhood_grid_radius_,
    int neighborhood_grid_width_,
    int kernel_radius_,
    int stride1,
    int stride2,
    int pad_size,
    int corr_type_multiply,
    cudaStream_t stream);
#ifdef __cplusplus
}
#endif

#endif
