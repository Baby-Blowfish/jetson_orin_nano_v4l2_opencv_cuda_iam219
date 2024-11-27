#include <stdio.h>
#include <fcntl.h>
#include <unistd.h>
#include <linux/fb.h>
#include <sys/ioctl.h>

#define FBDEVICE "/dev/fb0"

int main(int argc, char**argv)
{
	int fbfd = 0;
	// 프레임 버퍼 정보 처리를 위한 구조체
	struct fb_var_screeninfo vinfo, old_vinfo;
	struct fb_fix_screeninfo finfo;

	// 프레임 버퍼를 위한 디바이스 파일을 읽기와 쓰기 모드로 연다
	fbfd = open(FBDEVICE, O_RDWR);
	if(fbfd < 0)
	{
		perror("Error : cannot open framebuffer device");
		return -1;
	}

	// 현재 프레임 버퍼에 대한 화면 정보를 얻어온다
	if(ioctl(fbfd, FBIOGET_FSCREENINFO, &finfo) < 0 )
	{
		perror("Error reading fixed information");
		return -1;
	}

	// 현재 프레임 버퍼에 대한 가상 화면 정보를 얻어온다
	if(ioctl(fbfd, FBIOGET_VSCREENINFO, &vinfo) < 0 )
	{
		perror("Error reading fixed information");
		return -1;
	}

	// 현재 프레임 버퍼에 대한 정보를 출력한다.
	printf("Resolution : %dx%d, %dbpp\n",vinfo.xres,vinfo.yres,vinfo.bits_per_pixel);
	printf("Virtual Resolution : %dx%d\n",vinfo.xres_virtual,vinfo.yres_virtual);
	printf("Length of frame buffer memory : %d\n", finfo.smem_len);

	// 각 색상 채널의 오프셋과 길이를 출력
	printf("Red: offset = %d, length = %d\n", vinfo.red.offset, vinfo.red.length);
	printf("Green: offset = %d, length = %d\n", vinfo.green.offset, vinfo.green.length);
	printf("Blue: offset = %d, length = %d\n", vinfo.blue.offset, vinfo.blue.length);
	printf("Alpha (transparency): offset = %d, length = %d\n", vinfo.transp.offset, vinfo.transp.length);


  // finfo.visual 값을 출력
    printf("finfo.visual: ");
    switch (finfo.visual) {
        case FB_VISUAL_TRUECOLOR:
            printf("FB_VISUAL_TRUECOLOR (True color, likely RGB888 or BGR888)\n");
            break;
        case FB_VISUAL_DIRECTCOLOR:
            printf("FB_VISUAL_DIRECTCOLOR (Direct color, with a separate color map)\n");
            break;
        case FB_VISUAL_PSEUDOCOLOR:
            printf("FB_VISUAL_PSEUDOCOLOR (Pseudo color, uses a color map)\n");
            break;
        case FB_VISUAL_STATIC_PSEUDOCOLOR:
            printf("FB_VISUAL_STATIC_PSEUDOCOLOR (Static pseudo color, fixed color map)\n");
            break;
        default:
            printf("Unknown visual type (%d)\n", finfo.visual);
    }

	// 이전의 값을 백업
	old_vinfo = vinfo;

	// 프레임 버퍼에 새로운 해상도(800x600)을 설정
	vinfo.xres = 800;
	vinfo.yres = 600;

	if(ioctl(fbfd, FBIOPUT_VSCREENINFO, &vinfo)<0)
	{
		perror("fbdev ioctl(PUT)");
		return -1;
	}

	// 설정한 프레임 버퍼에 대한 정보를 출력한다.
	printf("New Resolution : %dx%d, %dbpp\n", vinfo.xres, vinfo.yres, vinfo.bits_per_pixel);

	getchar();	// 사용자 입력 기다림

	ioctl(fbfd,FBIOPUT_VSCREENINFO, &old_vinfo);	// 원래 값으로 다시 설정

	close(fbfd);	// 사용이 끝난 프레임 버퍼의 디바이스 파일 닫기

	return 0;
}

