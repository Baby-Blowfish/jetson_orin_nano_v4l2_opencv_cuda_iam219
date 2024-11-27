#include <cuda_runtime.h>
#include "CUDAGammaCorrection.h"

// CUDA 커널: 감마 보정 수행
__global__ void gamma_correction_kernel(uint16_t* image, int width, int height, float gamma) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    int idx = y * width + x;

    // 감마 보정 공식 적용
    float normalized = image[idx] / 65535.0f;  // 16-bit 정규화
    float corrected = powf(normalized, gamma);
    image[idx] = static_cast<uint16_t>(corrected * 65535.0f);  // 다시 16-bit로 변환
}

// CUDA에서 감마 보정 호출
void apply_gamma_correction_cuda(cv::cuda::GpuMat& gpu_image, float gamma) {
    uint16_t* device_image = reinterpret_cast<uint16_t*>(gpu_image.data);

    dim3 block_size(16, 16);
    dim3 grid_size((gpu_image.cols + block_size.x - 1) / block_size.x,
                   (gpu_image.rows + block_size.y - 1) / block_size.y);

    // CUDA 커널 호출
    gamma_correction_kernel<<<grid_size, block_size>>>(device_image, gpu_image.cols, gpu_image.rows, gamma);

    // CUDA 동기화
    cudaDeviceSynchronize();
}

