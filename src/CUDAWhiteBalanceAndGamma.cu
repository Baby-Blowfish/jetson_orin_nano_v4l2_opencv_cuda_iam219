#include "CUDAWhiteBalanceAndGamma.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

// CUDA 커널 정의
__global__ void whiteBalanceAndGammaKernel(uchar3* image, int width, int height, float rGain, float gGain, float bGain, float gamma) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    int idx = y * width + x;

    uchar3 pixel = image[idx];

    // 화이트 밸런스 적용
    float r = pixel.x * rGain;
    float g = pixel.y * gGain;
    float b = pixel.z * bGain;

    // 감마 보정 적용
    r = powf(r / 255.0f, gamma) * 255.0f;
    g = powf(g / 255.0f, gamma) * 255.0f;
    b = powf(b / 255.0f, gamma) * 255.0f;

    // 값 클램핑
    pixel.x = min(max((int)r, 0), 255);
    pixel.y = min(max((int)g, 0), 255);
    pixel.z = min(max((int)b, 0), 255);

    image[idx] = pixel;
}

// CUDA 기반 화이트 밸런스 및 감마 보정 적용 함수
void applyWhiteBalanceAndGammaCUDA(cv::cuda::GpuMat& gpuImage, float rGain, float gGain, float bGain, float gamma) {
    dim3 block(16, 16);
    dim3 grid((gpuImage.cols + block.x - 1) / block.x, (gpuImage.rows + block.y - 1) / block.y);

    uchar3* d_image = (uchar3*)gpuImage.data;

    whiteBalanceAndGammaKernel<<<grid, block>>>(d_image, gpuImage.cols, gpuImage.rows, rGain, gGain, bGain, gamma);

    cudaDeviceSynchronize(); // 커널 실행 완료 대기
}

