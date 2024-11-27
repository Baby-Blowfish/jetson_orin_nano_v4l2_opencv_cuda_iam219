#include <stdio.h>
#include <unistd.h>
#include <fcntl.h>
#include <linux/fb.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <stdlib.h>

#define FBDEVICE "/dev/fb0"

typedef unsigned char ubyte;

// 프레임 버퍼 정보
struct fb_var_screeninfo vinfo;
struct fb_fix_screeninfo finfo;

// 선 그리기 함수
void draw_line(int x1, int y1, int x2, int y2, ubyte r, ubyte g, ubyte b) {
    int fbfd;
    ubyte *fbp = NULL;

    // 프레임 버퍼 열기
    fbfd = open(FBDEVICE, O_RDWR);
    if (fbfd < 0) {
        perror("Error: cannot open framebuffer device");
        return;
    }

    // 프레임 버퍼 정보 가져오기
    if (ioctl(fbfd, FBIOGET_VSCREENINFO, &vinfo) < 0 || ioctl(fbfd, FBIOGET_FSCREENINFO, &finfo) < 0) {
        perror("Error reading framebuffer information");
        close(fbfd);
        return;
    }

    // 프레임 버퍼 메모리 맵핑
    size_t screensize = vinfo.yres_virtual * finfo.line_length;
    fbp = (ubyte *)mmap(0, screensize, PROT_READ | PROT_WRITE, MAP_SHARED, fbfd, 0);
    if ((long)fbp == -1) {
        perror("Error: failed to map framebuffer device to memory");
        close(fbfd);
        return;
    }

    int depth = vinfo.bits_per_pixel / 8; // 한 픽셀당 바이트 수
    ubyte a = 0xFF; // 알파 값 (불투명)

    // 브레젠험 선 알고리즘
    int dx = abs(x2 - x1), sx = x1 < x2 ? 1 : -1;
    int dy = -abs(y2 - y1), sy = y1 < y2 ? 1 : -1;
    int err = dx + dy, e2;

    while (1) {
        // 현재 픽셀 위치 계산
        if (x1 >= 0 && x1 < vinfo.xres && y1 >= 0 && y1 < vinfo.yres) {
            int offset = (x1 + y1 * vinfo.xres) * depth;
            *(fbp + offset + 0) = b; // Blue
            *(fbp + offset + 1) = g; // Green
            *(fbp + offset + 2) = r; // Red
            *(fbp + offset + 3) = a; // Alpha
        }

        // 종료 조건
        if (x1 == x2 && y1 == y2) break;

        // 다음 픽셀로 이동
        e2 = 2 * err;
        if (e2 >= dy) {
            err += dy;
            x1 += sx;
        }
        if (e2 <= dx) {
            err += dx;
            y1 += sy;
        }
    }

    // 메모리 언맵 및 파일 닫기
    munmap(fbp, screensize);
    close(fbfd);
}


int main() {
    // 빨간 선 그리기 (100, 100)에서 (400, 300)까지
    draw_line(100, 100, 400, 300, 255, 0, 0);

    // 파란 선 그리기 (50, 50)에서 (200, 400)까지
    draw_line(50, 50, 200, 400, 0, 0, 255);

    return 0;
}
