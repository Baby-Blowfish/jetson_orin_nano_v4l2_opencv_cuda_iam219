#include <stdio.h>
#include <stdlib.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/ioctl.h>
#include <linux/videodev2.h>
#include <string.h>

int main() {
    int fd = open("/dev/video0", O_RDWR);
    if (fd < 0) {
        perror("Failed to open video device");
        return -1;
    }

    // 포맷 설정
    struct v4l2_format fmt;
    memset(&fmt, 0, sizeof(fmt));
    fmt.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    fmt.fmt.pix.width = 3280;
    fmt.fmt.pix.height = 2464;
    fmt.fmt.pix.pixelformat = V4L2_PIX_FMT_SRGGB10;
    fmt.fmt.pix.field = V4L2_FIELD_NONE;

    // 포맷 설정 적용
    if (ioctl(fd, VIDIOC_S_FMT, &fmt) < 0) {
        perror("Setting format failed");
        close(fd);
        return -1;
    }

    // 적용된 포맷 확인
    if (ioctl(fd, VIDIOC_G_FMT, &fmt) < 0) {
        perror("Getting format failed");
        close(fd);
        return -1;
    }

    // 포맷 정보 출력
    printf("Format set: Width=%d, Height=%d, Pixel Format=%c%c%c%c, SizeImage=%u\n",
           fmt.fmt.pix.width, fmt.fmt.pix.height,
           fmt.fmt.pix.pixelformat & 0xFF,
           (fmt.fmt.pix.pixelformat >> 8) & 0xFF,
           (fmt.fmt.pix.pixelformat >> 16) & 0xFF,
           (fmt.fmt.pix.pixelformat >> 24) & 0xFF,
           fmt.fmt.pix.sizeimage);

    // 파일 열기
    FILE *file = fopen("frame.raw", "wb");
    if (!file) {
        perror("Failed to open file");
        close(fd);
        return -1;
    }

    // 버퍼 할당
    void *buffer = malloc(fmt.fmt.pix.sizeimage);
    if (!buffer) {
        perror("Failed to allocate memory");
        fclose(file);
        close(fd);
        return -1;
    }

    // 프레임 캡처
    ssize_t bytesRead = read(fd, buffer, fmt.fmt.pix.sizeimage);
    if (bytesRead <= 0) {
        perror("Failed to capture frame");
        free(buffer);
        fclose(file);
        close(fd);
        return -1;
    }

    // 데이터 저장
    fwrite(buffer, 1, fmt.fmt.pix.sizeimage, file);
    printf("Frame captured and saved as 'frame.raw'\n");

    // 리소스 정리
    free(buffer);
    fclose(file);
    close(fd);

    return 0;
}

