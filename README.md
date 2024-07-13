# Comparative Performance Study: CPU vs GPU with CUDA

## Algorithms Covered 
- Breadth-First Search
- Bitonic Sort
- Matrix Multiplication
- Radix Sort

## Hardware & Software Required 
- [CUDA capable GPU](https://developer.nvidia.com/cuda-gpus)
- x64 Arch CPU device
- [Visual Studio](https://visualstudio.microsoft.com) :  Install Visual Studio and ensure CUDA support is enabled.
- [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads) : Download and install the CUDA Toolkit compatible with your GPU.

## Usage
### 1. Project Setup:
- Create a new CUDA project in Visual Studio.
- Place your CUDA kernels and main code in kernel.cu.
### 2.	Building and Running:
Build the .sln file to compile and execute the CUDA project.
### 3.	Monitoring Execution:
Visual Studio provides memory usage graphs for both CPU and GPU during execution.
### 4.	Output and Analysis:
Check the shell output for execution times and comparative performance metrics.

>[!CAUTION]
> Ensure all files maintain consistent structures for proper compilation.

> [!NOTE]  
>- Refer to [NVIDIA documentation](https://docs.nvidia.com/cuda/index.html) for GPU-specific optimizations and best practices.
>- For algorithmic understanding and implementation on GPUs, refer to [Udacity’s YouTube](https://youtube.com/playlist?list=PLAwxTw4SYaPnFKojVQrmyOGFCqHTxfdv2&si=5OmSy-aMy-ykL44w) channel.

### [Project Report](docs.pdf)