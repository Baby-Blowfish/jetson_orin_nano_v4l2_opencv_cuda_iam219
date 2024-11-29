#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include <cassert>

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

// CUDA 기반 크롭 및 재구성 함수
void cropAndReorganizeImageCUDA(const uint16_t* src, uint16_t* dst, int srcWidth, int dstWidth, int height) {
    dim3 block(256); // 스레드 블록 크기
    dim3 grid((dstWidth + block.x - 1) / block.x, height); // 각 행을 처리하는 그리드 크기

    // CUDA 커널 호출
    cropAndReorganizeImageKernel<<<grid, block>>>(src, dst, srcWidth, dstWidth, height);
    cudaDeviceSynchronize();
}


int main() {
    const int srcWidth = 3280;
    const int dstWidth = 3264;
    const int height = 4;

    std::vector<uint16_t> srcData(srcWidth * height, 0);
    std::vector<uint16_t> expectedDstData(dstWidth * height, 0);
    std::vector<uint16_t> dstData(dstWidth * height, 0);

    // 입력 데이터 초기화
    for (int row = 0; row < height; ++row) {
        for (int col = 0; col < srcWidth; ++col) {
            srcData[row * srcWidth + col] = row * 10000 + col;
        }
    }

    // 예상 출력 데이터 초기화
    for (int row = 0; row < height; ++row) {
        for (int col = 0; col < dstWidth; ++col) {
            if (row % 2 == 0) {
                if (col < 3264)
                    expectedDstData[row * dstWidth + col] = row * 10000 + col; // RG 픽셀
            } else {
                if (col < 16) {
                    expectedDstData[row * dstWidth + col] = row * 10000 + 3264 + col; // 의미 있는 GB
                } else {
                    expectedDstData[row * dstWidth + col] = row * 10000 + (col - 16); // 일반 GB
                }
            }
        }
    }

    uint16_t *d_src, *d_dst;
    cudaMalloc(&d_src, srcWidth * height * sizeof(uint16_t));
    cudaMalloc(&d_dst, dstWidth * height * sizeof(uint16_t));

    cudaMemcpy(d_src, srcData.data(), srcWidth * height * sizeof(uint16_t), cudaMemcpyHostToDevice);

    // CUDA 커널 호출
    dim3 block(32, 8); // 블록 크기
    dim3 grid((dstWidth + block.x - 1) / block.x, (height + block.y - 1) / block.y);
    cropAndReorganizeImageKernel<<<grid, block>>>(d_src, d_dst, srcWidth, dstWidth, height);
    cudaDeviceSynchronize();

    cudaMemcpy(dstData.data(), d_dst, dstWidth * height * sizeof(uint16_t), cudaMemcpyDeviceToHost);

    // 결과 비교
    bool testPassed = true;
    for (int row = 0; row < height; ++row) {
        for (int col = 0; col < dstWidth; ++col) {
            int idx = row * dstWidth + col;
            if (dstData[idx] != expectedDstData[idx]) {
                std::cout << "Mismatch at row " << row << ", col " << col
                          << ": expected " << expectedDstData[idx] << ", got " << dstData[idx] << "\n";
                testPassed = false;
            }
        }
    }

    if (testPassed) {
        std::cout << "Test passed: cropped and reorganized data matches expected values.\n";
    } else {
        std::cout << "Test failed: data mismatch found.\n";
    }

    cudaFree(d_src);
    cudaFree(d_dst);

    return 0;
}


