//#include <TH/TH.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAEvent.h>

int corr_cpu_forward(at::Tensor input1,
                      at::Tensor input2,
                      at::Tensor rbot1,
                      at::Tensor rbot2,
                      at::Tensor output,
                      int pad_size,
                      int kernel_size,
                      int max_displacement,
                      int stride1,
                      int stride2,
                      int corr_type_multiply)
{
    return 1;
}

int corr_cpu_backward(at::Tensor input1,
                       at::Tensor input2,
                       at::Tensor rbot1,
                       at::Tensor rbot2,
                       at::Tensor gradOutput,
                       at::Tensor gradInput1,
                       at::Tensor gradInput2,
                       int pad_size,
                       int kernel_size,
                       int max_displacement,
                       int stride1,
                       int stride2,
                       int corr_type_multiply)
{
    return 1;
}

