#include "corr_cuda_kernel.h"
#include <torch/extension.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <cmath>

// ================= Forward =================
int corr_cuda_forward(
    const at::Tensor& input1,
    const at::Tensor& input2,
    at::Tensor& rbot1,
    at::Tensor& rbot2,
    at::Tensor& output,
    int pad_size,
    int kernel_size,
    int max_displacement,
    int stride1,
    int stride2,
    int corr_type_multiply
)
{
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    blob_rearrange_ongpu(input1, rbot1, pad_size, stream);
    blob_rearrange_ongpu(input2, rbot2, pad_size, stream);

    long kernel_radius_ = (kernel_size - 1) / 2;
    long neighborhood_grid_radius_ = max_displacement / stride2;
    long neighborhood_grid_width_ = neighborhood_grid_radius_ * 2 + 1;

    CorrelateData_ongpu(rbot1, rbot2, output, max_displacement,
                        neighborhood_grid_radius_, neighborhood_grid_width_,
                        kernel_radius_, kernel_size, stride1, stride2,
                        corr_type_multiply, stream);

    return 1;
}

// ================= Backward =================
int corr_cuda_backward(
    const at::Tensor& input1,
    const at::Tensor& input2,
    at::Tensor& rbot1,
    at::Tensor& rbot2,
    const at::Tensor& gradOutput,
    at::Tensor& gradInput1,
    at::Tensor& gradInput2,
    int pad_size,
    int kernel_size,
    int max_displacement,
    int stride1,
    int stride2,
    int corr_type_multiply
)
{
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    blob_rearrange_ongpu(input1, rbot1, pad_size, stream);
    blob_rearrange_ongpu(input2, rbot2, pad_size, stream);

    long kernel_radius_ = (kernel_size - 1) / 2;
    long neighborhood_grid_radius_ = max_displacement / stride2;
    long neighborhood_grid_width_ = neighborhood_grid_radius_ * 2 + 1;

    CorrelateDataBackward_ongpu(rbot1, rbot2, gradOutput, gradInput1, gradInput2,
                                max_displacement, neighborhood_grid_radius_, neighborhood_grid_width_,
                                kernel_radius_, stride1, stride2, pad_size,
                                corr_type_multiply, stream);

    return 1;
}

// ================= PyBind11 =================
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("corr_cuda_forward", &corr_cuda_forward, "Correlation forward (CUDA)");
    m.def("corr_cuda_backward", &corr_cuda_backward, "Correlation backward (CUDA)");
}
