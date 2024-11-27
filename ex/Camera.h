#ifndef CAMERA_H
#define CAMERA_H

#include "common.h"
#include "FrameBuffer.h"

class Camera {
public:
    Camera();
    ~Camera();

    bool captureFrame(FrameBuffer& framebuffer);
    int get_fd() const;
    int clip(int value, int min, int max);

private:
    struct Buffer {
        void* start;
        size_t length;
    };

    int fd;
    std::vector<Buffer> buffers;

    void initDevice();
    void initMMap();
    void startCapturing();
    void stopCapturing();
    void processImage(const void* data, FrameBuffer& framebuffer);
    
};

#endif // CAMERA_H
