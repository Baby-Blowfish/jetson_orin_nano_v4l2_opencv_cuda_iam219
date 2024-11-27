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
    cv::Scalar computeSSIM(const cv::Mat& img1, const cv::Mat& img2);
    void apply_white_balance(cv::Mat& bayer_image);
    void apply_white_balance2(cv::Mat& bayer_image);
    void processRawImage(void* data, int width, int height);
    void processRawImageCUDA(void* data, int width, int height);
    bool captureFrame(FrameBuffer& framebuffer);
    bool captureOpencv(void);
    int get_fd() const;
    void checkFormat();
    int clip(int value, int min, int max);
    void saveFrameToFile(const void* data, size_t size);

    void saveHistogramImage(void* data, int width, int height, const std::string& histogramImageFilename, const std::string& histogramCsvFilename);

    void processAndVisualizeRawData(void* data, int width, int height);

    void applyGammaCorrection(cv::cuda::GpuMat& gpu_image, float gamma);

    void processRGGBImage(void* data, int width, int height, const std::string& histogramCsvFilename,const std::string& histogramPngFilename);

    cv::Mat convertRGGBToMat(void* data, int width, int height);
    void applyWhiteBalance(cv::Mat& image);
    void applyGammaCorrection(cv::Mat& image, double gamma);



private:
    struct Buffer {
        void* start;
        size_t length;
    };

    // RG10 포맷 구조체 정의
    struct rg10_pixel_odd {
        uint16_t r : 10;       // R 값
        uint16_t g : 10;       // G 값
        uint32_t padding : 12; // 패딩 (정렬용)
    };

    struct rg10_pixel_even {
        uint16_t g : 10;       // G 값
        uint16_t b : 10;       // B 값
        uint32_t padding : 12; // 패딩 (정렬용)
    };


    int fd;
    int frameCounter = 0, image_count = 1;
    std::vector<Buffer> buffers;

    void initDevice();
    void initMMap();
    void startCapturing();
    void stopCapturing();
    void processImage(const void* data, FrameBuffer& framebuffer);

};

#endif // CAMERA_H
