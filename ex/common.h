#ifndef __COMMON_H__  // 헤더 파일 중복 포함을 방지하기 위한 매크로 정의 시작
#define __COMMON_H__

#include <iostream>
#include <fstream>
#include <vector>
#include <cstring>
#include <stdexcept>
#include <fcntl.h>
#include <unistd.h>

#include <sys/ioctl.h>
#include <sys/types.h>
#include <sys/time.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/select.h>


#include <linux/fb.h>
#include <asm/types.h>              /* videodev2.h에서 필요한 데이터 타입 정의 */
#include <linux/videodev2.h>



#endif  // 헤더 파일 중복 포함 방지를 위한 매크로 정의 끝