#ifndef CUDA_IMAGE_PROCESSING_H
#define CUDA_IMAGE_PROCESSING_H

#include <opencv2/core/cuda.hpp>
#include <cuda_runtime.h> // CUDA 런타임 API 함수 선언
#include <cuda.h>         // CUDA 드라이버 API 함수 선언

/**
 * @file CUDAImageProcessing.h
 * @brief Header file for CUDA-based image processing functions, including white balance, gamma correction, and image cropping.
 */

/**
 * @brief Applies white balance and gamma correction to an image stored on the GPU.
 *
 * This function adjusts the white balance and applies gamma correction to an input image
 * represented as a `cv::cuda::GpuMat` with 8-bit 3-channel (RGB) data.
 *
 * @param[in,out] gpuImage The input image on the GPU (CV_8UC3). The white balance and gamma corrections
 *                         are applied in-place.
 * @param[in] gamma The gamma correction factor. A value less than 1 darkens the image, while a value
 *                  greater than 1 brightens it.
 *
 * @note This function assumes that the input image is already in RGB format.
 * @note The input image must be stored in GPU memory and represented as a `cv::cuda::GpuMat`.
 */
void applyWhiteBalanceAndGammaCUDA(cv::cuda::GpuMat& gpuImage, float gamma);

/**
 * @brief Crops and reorganizes a 16-bit Bayer image stored on the GPU.
 *
 * This function crops and reorganizes a Bayer image stored in a single-channel 16-bit format.
 * It processes even and odd rows differently to retain only the meaningful pixel data according
 * to a specific Bayer pattern.
 *
 * @param[in] src Pointer to the input image data in GPU memory. The source image is in Bayer format,
 *                stored as a single channel (16-bit per pixel).
 * @param[out] dst Pointer to the output image data in GPU memory. The reorganized image data
 *                 will be written to this location.
 * @param[in] srcWidth The width (number of columns) of the input image.
 * @param[in] dstWidth The width (number of columns) of the output image.
 * @param[in] height The height (number of rows) of both the input and output images.
 *
 * @note The memory for `dst` must be allocated before calling this function.
 * @note Both `src` and `dst` must point to memory on the GPU.
 */
void cropAndReorganizeImageCUDA(const uint16_t* src, uint16_t* dst, int srcWidth, int dstWidth, int height);

#endif // CUDA_IMAGE_PROCESSING_H

