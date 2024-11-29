#include "CUDAimageprocessing.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

__global__ void cropAndReorganizeImageKernel(const uint16_t* src, uint16_t* dst, int srcWidth, int dstWidth, int height) {
    int row = blockIdx.y * blockDim.y + threadIdx.y; // 현재 행
    int col = blockIdx.x * blockDim.x + threadIdx.x; // 현재 열

    if (row >= height || col >= dstWidth) return;

    if (row % 2 == 0) {
        // 짝수 행: 3264개의 RG 픽셀만 복사
        if (col < 3264) {
            int srcIdx = row * srcWidth + col;
            int dstIdx = row * dstWidth + col;
            dst[dstIdx] = src[srcIdx];
        }
    } else {
        // 홀수 행: 16개의 의미 있는 GB 픽셀 + 3248개의 GB 픽셀 복사
        int srcIdx;
        if (col < 16) {
            srcIdx = row * srcWidth + 3264 + col; // 의미 있는 GB
        } else if (col < 3264) {
            srcIdx = row * srcWidth + (col - 16); // GB
        } else {
            return; // 범위를 벗어난 경우
        }
        int dstIdx = row * dstWidth + col;
        dst[dstIdx] = src[srcIdx];
    }
}

// 블록 내 모든 요소의 합을 계산하는 CUDA 커널
__device__ float blockReduceSum(float val) {
    __shared__ float shared[1024];
    int tid = threadIdx.x;

    shared[tid] = val;
    __syncthreads();

    // 병렬 합산
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared[tid] += shared[tid + stride];
        }
        __syncthreads();
    }

    return shared[0];
}

// 이미지의 각 채널 평균 계산 커널
__global__ void computeChannelMean(const uchar3* image, int width, int height, float* mean) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    float rSum = 0.0f, gSum = 0.0f, bSum = 0.0f;

    for (int row = y; row < height; row += gridDim.y) {
        for (int col = x; col < width; col += blockDim.x * gridDim.x) {
            int idx = row * width + col;
            uchar3 pixel = image[idx];

            rSum += pixel.x; // Red 채널
            gSum += pixel.y; // Green 채널
            bSum += pixel.z; // Blue 채널
        }
    }

    // 블록별 합산
    float rBlockSum = blockReduceSum(rSum);
    float gBlockSum = blockReduceSum(gSum);
    float bBlockSum = blockReduceSum(bSum);

    // 첫 번째 스레드만 평균값 기록
    if (threadIdx.x == 0 && threadIdx.y == 0) {
        atomicAdd(&mean[0], rBlockSum);
        atomicAdd(&mean[1], gBlockSum);
        atomicAdd(&mean[2], bBlockSum);
    }
}

// 화이트 밸런스 및 감마 보정 커널
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








//---------------------------------------------------------------------------------------------------//




// CUDA 기반 크롭 및 재구성 함수
void cropAndReorganizeImageCUDA(const uint16_t* src, uint16_t* dst, int srcWidth, int dstWidth, int height) {
    dim3 block(256); // 스레드 블록 크기
    dim3 grid((dstWidth + block.x - 1) / block.x, height); // 각 행을 처리하는 그리드 크기

    // CUDA 커널 호출
    cropAndReorganizeImageKernel<<<grid, block>>>(src, dst, srcWidth, dstWidth, height);
    cudaDeviceSynchronize();
}



// CUDA 기반 화이트 밸런스 및 감마 보정 적용 함수
void applyWhiteBalanceAndGammaCUDA(cv::cuda::GpuMat& gpuImage, float gamma) {
    dim3 block(16, 16);
    dim3 grid((gpuImage.cols + block.x - 1) / block.x, (gpuImage.rows + block.y - 1) / block.y);

    uchar3* d_image = (uchar3*)gpuImage.data;

    // 채널별 평균값 계산
    float* d_mean;
    cudaMalloc(&d_mean, 3 * sizeof(float));
    cudaMemset(d_mean, 0, 3 * sizeof(float));

    computeChannelMean<<<grid, block>>>(d_image, gpuImage.cols, gpuImage.rows, d_mean);
    cudaDeviceSynchronize();

    float h_mean[3];
    cudaMemcpy(h_mean, d_mean, 3 * sizeof(float), cudaMemcpyDeviceToHost);

    float rMean = h_mean[0] / (gpuImage.cols * gpuImage.rows);
    float gMean = h_mean[1] / (gpuImage.cols * gpuImage.rows);
    float bMean = h_mean[2] / (gpuImage.cols * gpuImage.rows);

    // 보정 계수 계산
    float k = (rMean + gMean + bMean) / 3.0f;
    float rGain = k / rMean;
    float gGain = k / gMean;
    float bGain = k / bMean;

    // 화이트 밸런스 및 감마 보정 커널 호출
    whiteBalanceAndGammaKernel<<<grid, block>>>(d_image, gpuImage.cols, gpuImage.rows, rGain, gGain, bGain, gamma);
    cudaDeviceSynchronize();

    cudaFree(d_mean);
}

