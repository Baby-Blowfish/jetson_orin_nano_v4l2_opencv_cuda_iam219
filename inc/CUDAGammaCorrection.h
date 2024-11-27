#ifndef CUDA_GAMMA_CORRECTION_H
#define CUDA_GAMMA_CORRECTION_H

#include <opencv2/core/cuda.hpp>

// 감마 보정 함수 선언
void apply_gamma_correction_cuda(cv::cuda::GpuMat& gpu_image, float gamma);

#endif // CUDA_GAMMA_CORRECTION_H

