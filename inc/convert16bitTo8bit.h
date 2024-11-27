#ifndef CONVERT16BITTO8BIT_H
#define CONVERT16BITTO8BIT_H

#include <cuda_runtime.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief CUDA 커널을 호출하여 16비트 RG10 Bayer 데이터를 8비트로 변환
 *
 * @param d_input GPU 메모리 상의 16비트 입력 Bayer 데이터
 * @param d_output GPU 메모리 상의 8비트 출력 Bayer 데이터
 * @param width 이미지의 너비
 * @param height 이미지의 높이
 */
void convert16BitTo8BitCUDA(const uint16_t* d_input, uint8_t* d_output, int width, int height);

#ifdef __cplusplus
}
#endif

#endif // CONVERT16BITTO8BIT_H

