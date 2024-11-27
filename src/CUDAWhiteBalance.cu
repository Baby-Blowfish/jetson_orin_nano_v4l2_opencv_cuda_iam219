#include "CUDAWhiteBalance.h"
#include <cuda_runtime.h>

// 커널: 각 채널별 합과 픽셀 수 계산
__global__ void calculate_white_balance_gains(const uint16_t* image, int width, int height,
                                              unsigned long long* sum_r, unsigned long long* sum_g1, unsigned long long* sum_g2, unsigned long long* sum_b,
                                              int* count_r, int* count_g1, int* count_g2, int* count_b) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    int idx = y * width + x;

    __shared__ unsigned long long local_sum_r, local_sum_g1, local_sum_g2, local_sum_b;
    __shared__ int local_count_r, local_count_g1, local_count_g2, local_count_b;

    if (threadIdx.x == 0 && threadIdx.y == 0) {
        local_sum_r = 0;
        local_sum_g1 = 0;
        local_sum_g2 = 0;
        local_sum_b = 0;
        local_count_r = 0;
        local_count_g1 = 0;
        local_count_g2 = 0;
        local_count_b = 0;
    }
    __syncthreads();

    // Red 채널
    if ((y % 2 == 0) && (x % 2 == 0)) {
        atomicAdd(&local_sum_r, image[idx]);
        atomicAdd(&local_count_r, 1);
    }
    // Green 채널 1
    else if ((y % 2 == 0) && (x % 2 == 1)) {
        atomicAdd(&local_sum_g1, image[idx]);
        atomicAdd(&local_count_g1, 1);
    }
    // Green 채널 2
    else if ((y % 2 == 1) && (x % 2 == 0)) {
        atomicAdd(&local_sum_g2, image[idx]);
        atomicAdd(&local_count_g2, 1);
    }
    // Blue 채널
    else if ((y % 2 == 1) && (x % 2 == 1)) {
        atomicAdd(&local_sum_b, image[idx]);
        atomicAdd(&local_count_b, 1);
    }

    __syncthreads();

    if (threadIdx.x == 0 && threadIdx.y == 0) {
        atomicAdd(sum_r, local_sum_r);
        atomicAdd(count_r, local_count_r);
        atomicAdd(sum_g1, local_sum_g1);
        atomicAdd(count_g1, local_count_g1);
        atomicAdd(sum_g2, local_sum_g2);
        atomicAdd(count_g2, local_count_g2);
        atomicAdd(sum_b, local_sum_b);
        atomicAdd(count_b, local_count_b);
    }
}

// 커널: 화이트 밸런스 게인 적용
__global__ void white_balance_kernel(uint16_t* image, int width, int height, double gain_r, double gain_b) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    int idx = y * width + x;

    if ((y % 2 == 0) && (x % 2 == 0)) {  // Red 채널
        image[idx] = min(max(static_cast<uint16_t>(image[idx] * gain_r), 0), 65535);
    } else if ((y % 2 == 1) && (x % 2 == 1)) {  // Blue 채널
        image[idx] = min(max(static_cast<uint16_t>(image[idx] * gain_b), 0), 65535);
    }
}

// 메인 함수: 화이트 밸런스 계산 및 적용
void calculate_and_apply_white_balance(uint16_t* host_image, int width, int height) {
    size_t image_size = width * height * sizeof(uint16_t);
    uint16_t* device_image;
    cudaMalloc(&device_image, image_size);
    cudaMemcpy(device_image, host_image, image_size, cudaMemcpyHostToDevice);

    // 평균값 및 카운트 계산을 위한 변수
    unsigned long long *sum_r, *sum_g1, *sum_g2, *sum_b;
    int *count_r, *count_g1, *count_g2, *count_b;
    cudaMalloc(&sum_r, sizeof(unsigned long long));
    cudaMalloc(&sum_g1, sizeof(unsigned long long));
    cudaMalloc(&sum_g2, sizeof(unsigned long long));
    cudaMalloc(&sum_b, sizeof(unsigned long long));
    cudaMalloc(&count_r, sizeof(int));
    cudaMalloc(&count_g1, sizeof(int));
    cudaMalloc(&count_g2, sizeof(int));
    cudaMalloc(&count_b, sizeof(int));

    // 초기화
    cudaMemset(sum_r, 0, sizeof(unsigned long long));
    cudaMemset(sum_g1, 0, sizeof(unsigned long long));
    cudaMemset(sum_g2, 0, sizeof(unsigned long long));
    cudaMemset(sum_b, 0, sizeof(unsigned long long));
    cudaMemset(count_r, 0, sizeof(int));
    cudaMemset(count_g1, 0, sizeof(int));
    cudaMemset(count_g2, 0, sizeof(int));
    cudaMemset(count_b, 0, sizeof(int));

    // 커널 실행: 평균값 계산
    dim3 block_size(16, 16);
    dim3 grid_size((width + block_size.x - 1) / block_size.x, (height + block_size.y - 1) / block_size.y);
    calculate_white_balance_gains<<<grid_size, block_size>>>(device_image, width, height, sum_r, sum_g1, sum_g2, sum_b, count_r, count_g1, count_g2, count_b);

    // 결과 복사
    unsigned long long h_sum_r, h_sum_g1, h_sum_g2, h_sum_b;
    int h_count_r, h_count_g1, h_count_g2, h_count_b;
    cudaMemcpy(&h_sum_r, sum_r, sizeof(unsigned long long), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_sum_g1, sum_g1, sizeof(unsigned long long), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_sum_g2, sum_g2, sizeof(unsigned long long), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_sum_b, sum_b, sizeof(unsigned long long), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_count_r, count_r, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_count_g1, count_g1, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_count_g2, count_g2, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_count_b, count_b, sizeof(int), cudaMemcpyDeviceToHost);

    // Green 채널 평균 계산
    double avg_r = static_cast<double>(h_sum_r) / h_count_r;
    double avg_g1 = static_cast<double>(h_sum_g1) / h_count_g1;
    double avg_g2 = static_cast<double>(h_sum_g2) / h_count_g2;
    double avg_g = (avg_g1 + avg_g2) / 2.0;
    double avg_b = static_cast<double>(h_sum_b) / h_count_b;

    // Gain 계산 (Green 게인 제한)
    const double MAX_GAIN_LIMIT = 3.0;
    double gain_r = std::min(avg_g / avg_r, MAX_GAIN_LIMIT);
    double gain_b = std::min(avg_g / avg_b, MAX_GAIN_LIMIT);

    // 게인 적용
    white_balance_kernel<<<grid_size, block_size>>>(device_image, width, height, gain_r, gain_b);

    // 결과 복사 및 정리
    cudaMemcpy(host_image, device_image, image_size, cudaMemcpyDeviceToHost);
    cudaFree(device_image);
    cudaFree(sum_r);
    cudaFree(sum_g1);
    cudaFree(sum_g2);
    cudaFree(sum_b);
    cudaFree(count_r);
    cudaFree(count_g1);
    cudaFree(count_g2);
    cudaFree(count_b);
}

