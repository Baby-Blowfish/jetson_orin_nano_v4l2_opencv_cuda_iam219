# CMake 최소 버전 설정
cmake_minimum_required(VERSION 3.10)

# 프로젝트 이름과 버전 설정
project(FrameBufferCameraProject VERSION 1.0)

# C++ 표준 설정
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# 헤더와 소스 파일 디렉토리 정의
set(INC_DIR ${CMAKE_SOURCE_DIR}/inc)
set(SRC_DIR ${CMAKE_SOURCE_DIR}/src)

# 헤더 파일 디렉토리를 include 경로에 추가
include_directories(${INC_DIR})

# 소스 파일 리스트
set(SOURCES
    ${SRC_DIR}/main.cpp
    ${SRC_DIR}/FrameBuffer.cpp
    ${SRC_DIR}/Camera.cpp
)

# 실행 파일 이름
set(EXECUTABLE_NAME app)

# 실행 파일 생성
add_executable(${EXECUTABLE_NAME} ${SOURCES})

# POSIX 시스템에서 필요한 pthread 라이브러리 연결
target_link_libraries(${EXECUTABLE_NAME} pthread)
