#include "Camera.h"

#define VIDEODEV "/dev/video0"
#define WIDTH 640
#define HEIGHT 360

Camera::Camera() {
    fd = open(VIDEODEV, O_RDWR | O_NONBLOCK, 0);
    if (fd == -1) {
        throw std::runtime_error("Failed to open video device");
    }

    initDevice();
    initMMap();
    startCapturing();
}

Camera::~Camera() {
    stopCapturing();
    for (size_t i = 0; i < buffers.size(); ++i) {
        munmap(buffers[i].start, buffers[i].length);
    }
    close(fd);
}


int Camera::get_fd() const {
    return fd;
}

bool Camera::captureFrame(FrameBuffer& framebuffer) {
    struct v4l2_buffer buf{};
    buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    buf.memory = V4L2_MEMORY_MMAP;

    if (ioctl(fd, VIDIOC_DQBUF, &buf) == -1) {
        if (errno == EAGAIN) return false;
        throw std::runtime_error("Failed to dequeue buffer");
    }

    processImage(buffers[buf.index].start, framebuffer);

    if (ioctl(fd, VIDIOC_QBUF, &buf) == -1) {
        throw std::runtime_error("Failed to queue buffer");
    }
    return true;
}

void Camera::initDevice() {
    struct v4l2_capability cap{};
    if (ioctl(fd, VIDIOC_QUERYCAP, &cap) == -1) {
        throw std::runtime_error("Failed to query V4L2 device capabilities");
    }

    if (!(cap.capabilities & V4L2_CAP_VIDEO_CAPTURE) || !(cap.capabilities & V4L2_CAP_STREAMING)) {
        throw std::runtime_error("Device does not support required capabilities");
    }

    struct v4l2_format fmt{};
    fmt.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    fmt.fmt.pix.width = WIDTH;
    fmt.fmt.pix.height = HEIGHT;
    fmt.fmt.pix.pixelformat = V4L2_PIX_FMT_YUYV;
            //fmt.fmt.pix.width = 3280;
        //fmt.fmt.pix.height = 2464;
        //fmt.fmt.pix.pixelformat = V4L2_PIX_FMT_SRGGB10;
    fmt.fmt.pix.field = V4L2_FIELD_NONE;

    if (ioctl(fd, VIDIOC_S_FMT, &fmt) == -1) {
        throw std::runtime_error("Failed to set format");
    }
}

void Camera::initMMap() {
    struct v4l2_requestbuffers req{};
    req.count = 4;
    req.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    req.memory = V4L2_MEMORY_MMAP;

    if (ioctl(fd, VIDIOC_REQBUFS, &req) == -1) {
        throw std::runtime_error("Failed to request buffers");
    }

    if (req.count < 2) {
        throw std::runtime_error("Insufficient buffer memory");
    }

    buffers.resize(req.count);
    for (size_t i = 0; i < buffers.size(); ++i) {
        struct v4l2_buffer buf{};
        buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        buf.memory = V4L2_MEMORY_MMAP;
        buf.index = i;

        if (ioctl(fd, VIDIOC_QUERYBUF, &buf) == -1) {
            throw std::runtime_error("Failed to query buffer");
        }

        buffers[i].length = buf.length;
        buffers[i].start = mmap(nullptr, buf.length, PROT_READ | PROT_WRITE, MAP_SHARED, fd, buf.m.offset);
        if (buffers[i].start == MAP_FAILED) {
            throw std::runtime_error("Failed to map buffer");
        }
    }
}

void Camera::startCapturing() {
    for (size_t i = 0; i < buffers.size(); ++i) {
        struct v4l2_buffer buf{};
        buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        buf.memory = V4L2_MEMORY_MMAP;
        buf.index = i;

        if (ioctl(fd, VIDIOC_QBUF, &buf) == -1) {
            throw std::runtime_error("Failed to queue buffer");
        }
    }

    enum v4l2_buf_type type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    if (ioctl(fd, VIDIOC_STREAMON, &type) == -1) {
        throw std::runtime_error("Failed to start streaming");
    }
}

void Camera::stopCapturing() {
    enum v4l2_buf_type type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    if (ioctl(fd, VIDIOC_STREAMOFF, &type) == -1) {
        throw std::runtime_error("Failed to stop streaming");
    }
}

void Camera::processImage(const void* data, FrameBuffer& framebuffer) {
    const unsigned char* in = static_cast<const unsigned char*>(data);
    int height = HEIGHT;                    /* 비디오 데이터의 높이 (픽셀 수) */
    int istride = WIDTH * 2;                /* 한 라인의 데이터 크기 (YUYV 형식은 픽셀당 2바이트 사용) */
    int x, y, j;                            /* 반복문에서 사용할 변수 */
    int y0, u, y1, v, r, g, b, a = 0xff, depth_fb = framebuffer.getScreenInfo().bits_per_pixel / 8;   /* YUYV 데이터를 분리한 후 RGBA 변환에 사용할 변수 */
    long location = 0;                      /* 프레임버퍼에서 현재 위치를 가리킬 인덱스 */

    /* YUYV 데이터를 RGBA로 변환한 후 프레임버퍼에 쓰는 루프 */
    for (y = 0; y < height; ++y) {  /* 각 라인을 반복 */
        for (j = 0, x = 0; x < framebuffer.getScreenInfo().xres; j += 4, x += 2) {  /* 한 라인 내에서 YUYV 데이터를 2픽셀씩 처리 */
            /* 프레임버퍼의 크기를 넘어서지 않는지 검사 */
            if (j >= istride) {  /* 비디오 데이터의 유효 범위를 넘으면 넘어가서 빈 공간을 채움 */
                location += 2*depth_fb;
                continue;
            }


            /* YUYV 데이터에서 Y0, U, Y1, V 성분을 분리 */
            y0 = in[j];              /* 첫 번째 픽셀의 밝기 값 (Y0) */
            u = in[j + 1] - 128;     /* 색차 성분 U, 중앙값 128을 기준으로 보정 */
            y1 = in[j+ 2];          /* 두 번째 픽셀의 밝기 값 (Y1) */
            v = in[j + 3] - 128;     /* 색차 성분 V, 중앙값 128을 기준으로 보정 */

            /* YUV를 RGB로 변환 (첫 번째 픽셀) */
            r = clip((298 * y0 + 409 * v + 128) >> 8, 0, 255);  /* 밝기(Y)와 색차(V)를 사용해 빨강(R) 값 계산 */
            g = clip((298 * y0 - 100 * u - 208 * v + 128) >> 8, 0, 255);  /* Y, U, V를 사용해 초록(G) 값 계산 */
            b = clip((298 * y0 + 516 * u + 128) >> 8, 0, 255);  /* 밝기(Y)와 색차(U)를 사용해 파랑(B) 값 계산 */

            /* 32비트 BMP 저장을 위한 RGB 값 저장 */
            *(framebuffer.get_fbp() + location++) = b;  /* 파란색 값 저장 */
            *(framebuffer.get_fbp() + location++) = g;  /* 초록색 값 저장 */
            *(framebuffer.get_fbp() + location++) = r;  /* 빨간색 값 저장 */
            *(framebuffer.get_fbp() + location++) = a;  /* 알파 값 (투명도) 저장 */


            /* YUV를 RGB로 변환 (두 번째 픽셀) */
            r = clip((298 * y1 + 409 * v + 128) >> 8, 0, 255);  /* 두 번째 픽셀의 R 값 계산 */
            g = clip((298 * y1 - 100 * u - 208 * v + 128) >> 8, 0, 255);  /* 두 번째 픽셀의 G 값 계산 */
            b = clip((298 * y1 + 516 * u + 128) >> 8, 0, 255);  /* 두 번째 픽셀의 B 값 계산 */

            /* BMP 저장을 위한 두 번째 픽셀의 RGB 값 저장 */
            *(framebuffer.get_fbp() + location++) = b;  /* 두 번째 픽셀의 파란색 값 */
            *(framebuffer.get_fbp() + location++) = g;  /* 두 번째 픽셀의 초록색 값 */
            *(framebuffer.get_fbp() + location++) = r;  /* 두 번째 픽셀의 빨간색 값 */
            *(framebuffer.get_fbp() + location++) = a;  /* 두 번째 픽셀의 알파 값 (투명도) */
        }
        in += istride;  /* 한 라인 처리가 끝나면 다음 라인으로 이동 */
    }
}

int Camera::clip(int value, int min, int max) {
    return (value > max ? max : value < min ? min : value);
}
