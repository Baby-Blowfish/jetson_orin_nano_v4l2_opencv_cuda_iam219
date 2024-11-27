#include <stdio.h>
#include <unistd.h>
#include <fcntl.h>
#include <linux/fb.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <stdlib.h>

#define FBDEVICE "/dev/fb0"

typedef unsigned char ubyte;

// 프레임 버퍼 정보 처리를 위한 구조체
struct fb_var_screeninfo vinfo;
struct fb_fix_screeninfo finfo;

// 점을 그리는 함수
static void drawpoint(int fd, int x, int y, ubyte r, ubyte g, ubyte b)
{
	// 알파 값 정의 (불투명)
	ubyte a = 0xff;

	// 영상의 깊이 (바이트 단위)
	int depth = (vinfo.bits_per_pixel/8);

	// x, y 좌표에 해당하는 픽셀 위치 계산
	int offset = (x + y * vinfo.xres) * depth;

	// 파일 포인터를 해당 오프셋으로 이동
	lseek(fd, offset, SEEK_SET);

	// BGRA 순서로 색상 데이터 작성
	write(fd, &b, 1);  // 파랑
	write(fd, &g, 1);  // 초록
	write(fd, &r, 1);  // 빨강
	write(fd, &a, 1);  // 알파 (투명도)
}

// 선을 그리는 함수
static void drawline(int fd, int start_x, int end_x, int y, ubyte r, ubyte g, ubyte b)
{
	ubyte a = 0xFF;  // 알파 값

	// 영상의 깊이 (바이트 단위)
	int depth = (vinfo.bits_per_pixel/8);

	int offset = 0;

	// 주어진 x 좌표 범위 내에서 선을 그림
	for (int x = start_x; x < end_x; x++)
	{
		// 현재 x, y 좌표에 해당하는 픽셀 위치 계산
		offset = (x + y * vinfo.xres) * depth;

		// 파일 포인터를 해당 오프셋으로 이동
		lseek(fd, offset, SEEK_SET);

		// BGRA 순서로 색상 데이터 작성
		write(fd, &b, 1);  // 파랑
		write(fd, &g, 1);  // 초록
		write(fd, &r, 1);  // 빨강
		write(fd, &a, 1);  // 알파 (투명도)
	}
}

// 중점 원 알고리즘을 이용해 원을 그리는 함수
static void drawcircle(int fd, int center_x, int center_y, int radius, ubyte r, ubyte g, ubyte b)
{
	int x = radius, y = 0;
	int radiusError = 1 - x;

	// 원을 그리기 위한 알고리즘 반복
	while (x >= y)
	{
		// 8대칭성을 활용해 원의 점들을 그린다
		drawpoint(fd, x + center_x, y + center_y, r, g, b);
		drawpoint(fd, y + center_x, x + center_y, r, g, b);
		drawpoint(fd, -x + center_x, y + center_y, r, g, b);
		drawpoint(fd, -y + center_x, x + center_y, r, g, b);
		drawpoint(fd, -x + center_x, -y + center_y, r, g, b);
		drawpoint(fd, -y + center_x, -x + center_y, r, g, b);
		drawpoint(fd, x + center_x, -y + center_y, r, g, b);
		drawpoint(fd, y + center_x, -x + center_y, r, g, b);

		// 반지름 오류에 따라 x와 y 값 조정
		y++;
		if (radiusError < 0)
		{
			radiusError += 2 * y + 1;
		}
		else
		{
			x--;
			radiusError += 2 * (y - x + 1);
		}
	}
}

// 화면 전체를 채우는 함수
static void drawface(int fd, int start_x, int start_y, int end_x, int end_y, ubyte r, ubyte g, ubyte b)
{
	ubyte a = 0xFF;  // 알파 값 (불투명)

	// 끝 좌표가 0일 경우, 화면 해상도 전체를 채움
	if (end_x == 0) end_x = vinfo.xres;
	if (end_y == 0) end_y = vinfo.yres;

	// 주어진 영역을 순회하며 각 픽셀을 색상으로 채움
	for (int x = start_x; x < end_x; x++)
	{
		for (int y = start_y; y < end_y; y++)
		{
			// x, y 좌표에 해당하는 픽셀 위치 계산
			int offset = (x + y * vinfo.xres) * vinfo.bits_per_pixel / 8.;

			// 파일 포인터를 해당 오프셋으로 이동
			lseek(fd, offset, SEEK_SET);

			// BGRA 순서로 색상 데이터 작성
			write(fd, &b, 1);  // 파랑
			write(fd, &g, 1);  // 초록
			write(fd, &r, 1);  // 빨강
			write(fd, &a, 1);  // 알파 (투명도)
		}
	}
}

static void drawfacemmap(int fd, int start_x, int start_y, int end_x, int end_y, ubyte r, ubyte g, ubyte b)
{
	ubyte *pfb;

	ubyte a = 0xFF;

	int depth = vinfo.bits_per_pixel/8.;	// bytes_per_pixel

	if(end_x == 0) end_x = vinfo.xres;
	if(end_y == 0) end_y = vinfo.yres;


	// mma() 함수를 이용해서 메모리 맵을 작성
	pfb = (ubyte *)mmap(NULL,vinfo.xres*vinfo.yres*depth,PROT_READ|PROT_WRITE,MAP_SHARED,fd,0);

	for(int x = start_x ; x < end_x*depth; x+=depth)
	{
		for(int y = start_y; y < end_y; y++)
		{
			*(pfb + (x+0) + y*vinfo.xres*depth) = b;
			*(pfb + (x+1) + y*vinfo.xres*depth) = g;
			*(pfb + (x+2) + y*vinfo.xres*depth) = r;
			*(pfb + (x+3) + y*vinfo.xres*depth) = a;
		}
	}
	munmap(pfb, vinfo.xres * vinfo.yres * depth);
}


int main(int argc, char **argv)
{
	int fbfd, status, offset;

	// 프레임 버퍼 장치를 열기
	fbfd = open(FBDEVICE, O_RDWR);
	if (fbfd < 0)
	{
		perror("Error: cannot open framebuffer device");
		return -1;
	}

	// 프레임 버퍼의 화면 정보 가져오기
	if (ioctl(fbfd, FBIOGET_VSCREENINFO, &vinfo) < 0)
	{
		perror("Error reading fixed information");
		return -1;
	}

  // 고정 화면` 정보 가져오기
    if (ioctl(fbfd, FBIOGET_FSCREENINFO, &finfo)) {
        perror("Error reading fixed information");
        close(fbfd);
        return -1;
    }

    // 프레임 버퍼 메모리 맵핑
    size_t screensize = finfo.smem_len;
    unsigned char *fbptr = (unsigned char *)malloc(screensize);
    if (read(fbfd, fbptr, screensize) != screensize) {
        perror("Error reading framebuffer memory");
        free(fbptr);
        close(fbfd);
        return -1;
    }

    // 첫 번째 픽셀 데이터 확인
    unsigned int pixel = *((unsigned int *)fbptr);
    printf("First pixel data (hex): 0x%08X\n", pixel);

    // 엔디안 및 색상 순서 확인
    unsigned char *p = (unsigned char *)&pixel;
    printf("Byte order: %02X %02X %02X %02X\n", p[0], p[1], p[2], p[3]);

    free(fbptr);

	// 현재 프레임 버퍼의 해상도 및 색상 깊이 정보 출력
	printf("Resolution : %dx%d, %dbpp\n", vinfo.xres, vinfo.yres, vinfo.bits_per_pixel);
	printf("Virtual Resolution : %dx%d\n", vinfo.xres_virtual, vinfo.yres_virtual);

	// 각 색상 채널의 오프셋과 길이 출력
	printf("Red: offset = %d, length = %d\n", vinfo.red.offset, vinfo.red.length);
	printf("Green: offset = %d, length = %d\n", vinfo.green.offset, vinfo.green.length);
	printf("Blue: offset = %d, length = %d\n", vinfo.blue.offset, vinfo.blue.length);
	printf("Alpha (transparency): offset = %d, length = %d\n", vinfo.transp.offset, vinfo.transp.length);

	// 테스트를 위한 다양한 도형 그리기
	//drawface(fdfd, 0, 0, 0, 0, 255, 255, 0);  // 화면 전체를 노란색으로 채움
	//drawcircle(fdfd, 200, 200, 100, 255, 0, 255);  // 자홍색 원 그리기
	//drawline(fdfd, 0, 100, 200, 0, 255, 255);  // 청록색 선 그리기


	// 화면 채우기
	//drawfacemmap(fdfd, 0, 0, 0, 0, 255, 0, 0);
	//drawcircle(fdfd, 200, 200, 100, 255, 0, 255);  // 자홍색 원 그리기

	// 프레임 버퍼 장치 닫기
	close(fbfd);

	return 0;
}

