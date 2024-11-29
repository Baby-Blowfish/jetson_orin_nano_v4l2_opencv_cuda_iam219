#ifndef CAMERA_H
#define CAMERA_H

#include "common.h"
#include "FrameBuffer.h"

// VIDEODEV "/dev/video0"
// WIDTH 640
// HEIGHT 360


class Camera {
public:
    Camera();
    ~Camera();

    int get_fd() const;
    bool captureFrameBuffer(FrameBuffer& framebuffer);
    bool captureOpencv(Camera& camera);


private:
    struct Buffer {
        void* start;
        size_t length;
    };

    int fd;
    int frameCounter = 0, image_count = 1;
    struct Buffer *buffers = NULL;
    volatile unsigned int n_buffers = 0;      /* 버퍼 개수 */

    void initDevice();
    void initMMap();
    void startCapturing();
    void stopCapturing();

    // Frambuffer로 출력
    void processImage(const void* data, FrameBuffer& framebuffer);
    int clip(int value, int min, int max);


    // OpenCV로 출력
    void processRawImage(void* data, int width, int height);
    void apply_white_balance(cv::Mat& bayer_image);
    void apply_white_balance2(cv::Mat& bayer_image);



    // OpenCV & CUDA 출력
    void processRawImageCUDA(void* data, int width, int height);
    void applyWhiteBalance(cv::Mat& image);
    void applyGammaCorrection(cv::Mat& image, double gamma);



    // Test & Check
    void checkFormat();
    void saveFrameToFile(const void* data, size_t size);
    void processRGGBImage(void* data, int width, int height, const std::string& histogramCsvFilename,const std::string& histogramPngFilename);
    void processAndVisualizeRawData(void* data, int width, int height);
    void saveHistogramImage(uint8_t* data, int width, int height, const std::string& histogramImageFilename, const std::string& histogramCsvFilename);
    cv::Scalar computeSSIM(const cv::Mat& img1, const cv::Mat& img2);



};

#endif // CAMERA_H
