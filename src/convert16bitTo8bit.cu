#include "convert16bitTo8bit.h"
#include <cuda_runtime.h>
#include <stdint.h>

/**
 * @brief CUDA 커널: 16비트 RG10 데이터를 8비트로 변환
 * @param input 16비트 RG10 입력 배열
 * @param output 8비트 출력 배열
 * @param width 이미지 너비
 * @param height 이미지 높이
 */
__global__ void convert16BitTo8BitKernel(const uint16_t* input, uint8_t* output, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    int idx = y * width + x;

    // 10비트 데이터를 8비트로 변환 (비트 시프트)
    uint16_t pixel = input[idx] & 0x03FF;  // 하위 10비트 추출
    output[idx] = static_cast<uint8_t>(pixel >> 2); // 10비트를 8비트로 변환

}

/**
 * @brief 16비트 RG10 데이터를 8비트로 변환하는 CUDA 함수
 * @param d_input 16비트 입력 데이터(GPU 메모리)
 * @param d_output 8비트 출력 데이터(GPU 메모리)
 * @param width 이미지 너비
 * @param height 이미지 높이
 */
void convert16BitTo8BitCUDA(const uint16_t* d_input, uint8_t* d_output, int width, int height) {
    dim3 blockDim(16, 16);
    dim3 gridDim((width + blockDim.x - 1) / blockDim.x, (height + blockDim.y - 1) / blockDim.y);

    convert16BitTo8BitKernel<<<gridDim, blockDim>>>(d_input, d_output, width, height);

    cudaDeviceSynchronize(); // 동기화
}

