# GPU-Accelerated Satellite Image Processing Pipeline

A high-performance hybrid Computer Vision pipeline written in native C++/CUDA and OpenMP, explicitly bound to PyTorch using PyBind11. Designed to showcase a massive performance speedup when offloading compute-heavy image transformations from parallelized CPU threads directly to dedicated GPU vector cores.

## 🚀 Performance Metrics & Verification
* **CPU Execution Model (OpenMP):** Static scheduled multi-threaded image scaling and normalization.
* **GPU Execution Model (Native CUDA):** Massively parallel custom core kernel logic.
* **Actual Measured Acceleration:** **14.32x Speedup** (`CPU: 0.0167s` vs. `GPU: 0.0012s`).

### Hardware Profiling & Execution Timeline
Hardware optimization traces were extracted bare-metal on remote NVIDIA Tensor Core architectures using the PyTorch Profiler. 

Below is the verified microsecond-level hardware schedule showing the exact execution window of our custom kernel:

![NVIDIA CUDA Trace Profile](cuda_trace.png)

## 📦 Project Architecture & Deliverables
The core engine is packed into a ultra-minimalist, high-density codebase:
* `native_ops.cu`: Native CUDA kernel implementations and OpenMP parallel CPU wrappers.
* `binding.cpp`: PyBind11 C++ glue logic exposing compiled operations as an importable Python library.
* `setup.py`: Build configuration driving decoupled PEP 517 isolated compilation wrappers.
* `train.py`: Master PyTorch orchestration tracking gradient distribution, tensor streaming, and model convergence.

### Production Artifacts Included
* `evaluation_metrics.json`: Exported validation performance tracking model convergence on the EuroSAT dataset.
* `checkpoint_epoch_4.ckpt`: Serialized binary training state from the final epoch.
