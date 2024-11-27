#ifndef CUDA_WHITE_BALANCE_H
#define CUDA_WHITE_BALANCE_H

#include <stdint.h>

// 화이트 밸런스 게인 계산 및 적용 함수
void calculate_and_apply_white_balance(uint16_t* host_image, int width, int height);

#endif // CUDA_WHITE_BALANCE_H

