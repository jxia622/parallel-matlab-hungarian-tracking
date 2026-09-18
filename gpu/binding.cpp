#include <torch/extension.h>
std::vector<torch::Tensor> solve_cuda(torch::Tensor,torch::Tensor,torch::Tensor,double);
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){m.def("solve",&solve_cuda);}
