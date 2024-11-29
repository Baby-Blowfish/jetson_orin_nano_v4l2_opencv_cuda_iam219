# CMake 최소 버전 설정
cmake_minimum_required(VERSION 3.10)

# 프로젝트 이름과 버전 설정
project(V4L2CameraCudaProject VERSION 1.0)

# C++ 표준 설정
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# 빌드 타입 설정 (Release 기본값)
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# OpenCV 및 CUDA 설정
find_package(OpenCV REQUIRED)
find_package(CUDA REQUIRED)

# include 디렉토리 설정
include_directories(
    ${OpenCV_INCLUDE_DIRS}     # OpenCV 헤더 경로
    ${CUDA_INCLUDE_DIRS}       # CUDA 헤더 경로
    ${CMAKE_SOURCE_DIR}/inc    # inc 디렉토리 추가
)

# CUDA 및 OpenCV DNN CUDA 관련 플래그 설정
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}
    -O3
    -gencode arch=compute_87,code=sm_87 # Jetson Orin Nano에 맞는 아키텍처 설정
)

# 소스 파일 리스트 정의
set(SOURCES
    ${CMAKE_SOURCE_DIR}/src/main.cpp
    ${CMAKE_SOURCE_DIR}/src/Camera.cpp
    ${CMAKE_SOURCE_DIR}/src/FrameBuffer.cpp
)


# CUDA 소스 파일 정의
set(CUDA_SOURCES
    ${CMAKE_SOURCE_DIR}/src/CUDAimageprocessing.cu
)

# CUDA 파일 컴파일 추가
cuda_add_library(cuda_processing STATIC ${CUDA_SOURCES})

# 실행 파일 이름 설정
set(EXECUTABLE_NAME app)

# 실행 파일 생성
add_executable(${EXECUTABLE_NAME} ${SOURCES})

# 필요한 라이브러리 연결 (OpenCV, CUDA, pthread 등)
target_link_libraries(${EXECUTABLE_NAME}
    cuda_processing
    ${OpenCV_LIBS}             # OpenCV 라이브러리
    ${CUDA_LIBRARIES}          # CUDA 라이브러리
    ${CUDA_CUBLAS_LIBRARIES}   # CUDA BLAS 라이브러리
    pthread                    # pthread 라이브러리
)

# DNN 및 CUDA 관련 설정
target_compile_definitions(${EXECUTABLE_NAME} PRIVATE WITH_CUDA)

