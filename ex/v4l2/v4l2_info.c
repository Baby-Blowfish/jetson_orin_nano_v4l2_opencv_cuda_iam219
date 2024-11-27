#include <stdio.h>
#include <stdlib.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/ioctl.h>
#include <linux/videodev2.h>
#include <string.h>
#include <errno.h>

#define VIDEODEV "/dev/video0"  /* V4L2 장치 파일 경로 */

/* ioctl 호출을 위한 헬퍼 함수 */
static int xioctl(int fd, int request, void *arg)
{
    int r;
    do r = ioctl(fd, request, arg);
    while (-1 == r && EINTR == errno);  /* 인터럽트 발생 시 재시도 */
    return r;
}

int main()
{
    int fd;
    struct v4l2_capability cap;

    /* 비디오 장치 파일 열기 */
    fd = open(VIDEODEV, O_RDWR);
    if (fd < 0) {
        perror("Failed to open video device");
        return EXIT_FAILURE;
    }

    /* VIDIOC_QUERYCAP으로 장치의 기본 정보 확인 */
    if (xioctl(fd, VIDIOC_QUERYCAP, &cap) < 0) {
        perror("VIDIOC_QUERYCAP");
        close(fd);
        return EXIT_FAILURE;
    }

    /* 장치 정보 출력 */
    printf("Driver: %s\n", cap.driver);               /* 드라이버 이름 */
    printf("Card: %s\n", cap.card);                   /* 장치 이름 */
    printf("Bus Info: %s\n", cap.bus_info);           /* 버스 정보 (장치 위치) */
    printf("Version: %u.%u.%u\n",
           (cap.version >> 16) & 0xFF,
           (cap.version >> 8) & 0xFF,
           cap.version & 0xFF);                       /* 드라이버 버전 */

    /* 지원 기능 출력 */
    printf("Capabilities: 0x%08x\n", cap.capabilities);
    if (cap.capabilities & V4L2_CAP_VIDEO_CAPTURE)
        printf("  - Video Capture\n");
    if (cap.capabilities & V4L2_CAP_VIDEO_OUTPUT)
        printf("  - Video Output\n");
    if (cap.capabilities & V4L2_CAP_VIDEO_OVERLAY)
        printf("  - Video Overlay\n");
    if (cap.capabilities & V4L2_CAP_STREAMING)
        printf("  - Streaming\n");
    if (cap.capabilities & V4L2_CAP_READWRITE)
        printf("  - Read/Write\n");


    // 지원 pixelformat 출력
    struct v4l2_fmtdesc fmt;
    memset(&fmt, 0, sizeof(fmt));  // 구조체 초기화
    fmt.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;  // 비디오 캡처 타입 설정
    fmt.index = 0;  // 첫 번째 포맷부터 시작

    printf("Supported pixel formats:\n");

    while (xioctl(fd, VIDIOC_ENUM_FMT, &fmt) == 0) {
        printf("%d: %s\n", fmt.index, fmt.description);  // 지원하는 포맷의 설명 출력
        fmt.index++;
    }

    if (errno != EINVAL) {
        perror("VIDIOC_ENUM_FMT");
    }

    // 지원 해상도 확인
    struct v4l2_frmsizeenum frmsize;
    memset(&frmsize, 0, sizeof(frmsize));  // 구조체 초기화
    frmsize.pixel_format = V4L2_PIX_FMT_SRGGB10;  // 사용할 픽셀 포맷 설정
    frmsize.index = 0;  // 첫 번째 해상도부터 시작

    printf("Supported resolutions for 10-bit Bayer RGRG/GBGB:\n");
    while (xioctl(fd, VIDIOC_ENUM_FRAMESIZES, &frmsize) == 0) {
        if (frmsize.type == V4L2_FRMSIZE_TYPE_DISCRETE) {
            printf("%dx%d\n", frmsize.discrete.width, frmsize.discrete.height);
        }
        frmsize.index++;
    }

    if (errno != EINVAL) {
        perror("VIDIOC_ENUM_FRAMESIZES");
    }

    /* 장치 닫기 */
    close(fd);
    return EXIT_SUCCESS;
}

