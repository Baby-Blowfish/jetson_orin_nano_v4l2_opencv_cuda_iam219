#ifndef __COMMON_H__  // 헤더 파일 중복 포함을 방지하기 위한 매크로 정의 시작
#define __COMMON_H__


#include <iostream>                 // 입출력 스트림을 제공
#include <fstream>                  // 파일 입출력 스트림을 제공
#include <vector>                   // 동적 배열(vector) 컨테이너를 제공
#include <cstring>                  // 문자열 처리 및 메모리 조작 함수 (memcpy, strcmp 등) 제공
#include <stdexcept>                // 예외 처리 클래스(std::runtime_error 등) 제공
#include <fcntl.h>                  // 파일 제어 관련 함수(open, fcntl 등) 제공
#include <unistd.h>                 // POSIX 시스템 호출 함수(read, write, close 등) 제공
#include <sstream>                  // 문자열 스트림(stringstream) 제공
#include <iomanip>                  // 입출력 서식 조정(std::setprecision 등) 제공
#include <ctime>                    // 시간 관련 함수(time, localtime 등) 제공
#include <chrono>                   // C++11 고정밀 시간 측정(chrono::steady_clock 등) 제공
#include <malloc.h>

#include <sys/ioctl.h>              // 장치 제어 함수(ioctl) 제공
#include <sys/types.h>              // 데이터 타입 정의(pid_t, off_t 등) 제공
#include <sys/time.h>               // 시간 관련 구조체 및 함수(timeval, gettimeofday 등) 제공
#include <sys/mman.h>               // 메모리 매핑 관련 함수(mmap, munmap 등) 제공
#include <sys/stat.h>               // 파일 상태 정보(stat) 제공
#include <sys/select.h>             // 다중 I/O 처리(select 함수) 제공

#include <linux/fb.h>               // 프레임버퍼 장치 관련 구조체 및 상수 정의
#include <asm/types.h>              // 하위 수준 데이터 타입 정의(__u8, __u16 등)
#include <linux/videodev2.h>        // V4L2(Video4Linux2) 장치 제어 구조체 및 상수 정의

#include <opencv2/opencv.hpp>       // OpenCV의 주요 기능(cv::Mat, imshow 등) 제공

#include <opencv2/cudaimgproc.hpp>  // CUDA 기반 이미지 처리 함수(cv::cuda::cvtColor 등) 제공
#include <opencv2/cudaarithm.hpp>   // CUDA 기반 수학적 연산 함수(cv::cuda::add 등) 제공
#include <opencv2/core/cuda.hpp>    // CUDA 장치 관리 및 정보 함수(cv::cuda::GpuMat 등) 제공
#include <opencv2/cudafilters.hpp>  // CUDA 기반 필터 함수(cv::cuda::createGaussianFilter 등) 제공
#include <opencv2/cudacodec.hpp>    // CUDA 기반 비디오 디코딩/인코딩(cv::cudacodec::VideoReader 등) 제공
#include <opencv2/cudawarping.hpp>  // CUDA 기반 이미지 왜곡 처리 함수(cv::cuda::resize 등) 제공

#endif  // 헤더 파일 중복 포함 방지를 위한 매크로 정의 끝
