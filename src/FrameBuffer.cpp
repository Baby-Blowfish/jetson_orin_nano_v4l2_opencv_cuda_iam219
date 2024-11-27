#include "FrameBuffer.h"

#define FBDEV "/dev/fb0"

FrameBuffer::FrameBuffer() {
    fd = open(FBDEV, O_RDWR);
    if (fd == -1) {
        throw std::runtime_error("Failed to open framebuffer device");
    }

    if (ioctl(fd, FBIOGET_VSCREENINFO, &vinfo) == -1) {
        throw std::runtime_error("Failed to get framebuffer info");
    }

    width = vinfo.xres;
    height = vinfo.yres;
    screensize = vinfo.yres * vinfo.xres * (vinfo.bits_per_pixel / 8);

    fbp = static_cast<unsigned char*>(mmap(nullptr, screensize, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0));
    if (fbp == MAP_FAILED) {
        throw std::runtime_error("Failed to map framebuffer device to memory");
    }

    std::memset(fbp, 0, screensize);
}

FrameBuffer::~FrameBuffer() {
    if (fbp) {
        munmap(fbp, screensize);
    }
    if (fd != -1) {
        close(fd);
    }
}

void FrameBuffer::drawPixel(int x, int y, unsigned char r, unsigned char g, unsigned char b, unsigned char a) {
    int depth = vinfo.bits_per_pixel / 8;
    long location = (x + y * vinfo.xres) * depth;

    fbp[location + 0] = b;
    fbp[location + 1] = g;
    fbp[location + 2] = r;
    fbp[location + 3] = a;
}

fb_var_screeninfo FrameBuffer::getScreenInfo() const {
    return vinfo;
}

int FrameBuffer::get_fd() const {
    return fd;
}

unsigned char* FrameBuffer::get_fbp() const {
    return fbp;
}
