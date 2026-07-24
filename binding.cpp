#include <torch/extension.h>

std::vector<at::Tensor> launch_normalization(at::Tensor input_tensor, float mean, float std, bool use_gpu);
void init_mpi_diagnostics();

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("normalize_images", &launch_normalization, "Custom OpenMP/CUDA Image Normalization Engine");
    m.def("init_mpi_diagnostics", &init_mpi_diagnostics, "Initialize MPI communication hooks");
}
