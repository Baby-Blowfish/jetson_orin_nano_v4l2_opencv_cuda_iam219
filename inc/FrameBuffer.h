#ifndef FRAMEBUFFER_H
#define FRAMEBUFFER_H

#include "common.h"

class FrameBuffer {
public:
    FrameBuffer();
    ~FrameBuffer();

    void drawPixel(int x, int y, unsigned char r, unsigned char g, unsigned char b, unsigned char a = 0xff);
    fb_var_screeninfo getScreenInfo() const;
    int get_fd() const;
    unsigned char* get_fbp() const;

private:
    int fd;
    int width, height, screensize;
    unsigned char* fbp;
    struct fb_var_screeninfo vinfo;
};

#endif // FRAMEBUFFER_H
