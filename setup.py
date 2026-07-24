from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os

# Ensure MPI compiler wrapper paths are set if compiling multi-node
os.environ["CC"] = "mpicc"
os.environ["CXX"] = "mpicxx"

setup(
    name="native_ops",
    ext_modules=[
        CUDAExtension(
            name="native_ops",
            sources=["binding.cpp", "native_ops.cu"],
            extra_compile_args={
                "cxx": ["-O3", "-fopenmp"],
                "nvcc": ["-O3", "--use_fast_math", "-Xcompiler", "-fopenmp"]
            },
            libraries=["mpi"]
        )
    ],
    cmdclass={"build_ext": BuildExtension}
)
