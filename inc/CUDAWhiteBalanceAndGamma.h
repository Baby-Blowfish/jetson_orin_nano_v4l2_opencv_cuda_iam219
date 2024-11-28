#ifndef CUDA_WHITE_BALANCE_AND_GAMMA_H
#define CUDA_WHITE_BALANCE_AND_GAMMA_H

#include <opencv2/core/cuda.hpp>

/**
 * @brief Applies white balance and gamma correction to a CUDA GpuMat.
 *
 * @param gpuImage Input image on the GPU (CV_8UC3).
 * @param gamma Gamma correction factor.
 */
void applyWhiteBalanceAndGammaCUDA(cv::cuda::GpuMat& gpuImage, float gamma);

#endif // CUDA_WHITE_BALANCE_AND_GAMMA_H

