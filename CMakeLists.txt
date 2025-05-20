cmake_minimum_required(VERSION 3.5)
project(GTS LANGUAGES CXX CUDA)
set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CUDA_STANDARD 11) 
set(CUDA_SEPARABLE_COMPILATION ON)

# cuda debug
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -g -G")
# CPU代码调试选项
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g")

set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)
aux_source_directory(src SRC_LIST)
include_directories(include)

#find_package(OpenCV 4 REQUIRED)
#include_directories(${OpenCV_INCLUDE_DIRS})
#message(STATUS "include:${OpenCV_INCLUDE_DIRS}")
#link_libraries(${OpenCV_LIBS})

add_executable(${PROJECT_NAME} ${SRC_LIST})

#target_compile_options(${PROJECT_NAME} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-O2>)
# 针对 CUDA 编译器设置不同配置的选项
target_compile_options(${PROJECT_NAME} PRIVATE 
    $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CONFIG:Release>>:-Xcompiler=-O2>
    $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CONFIG:Debug>>:-Xcompiler=-Od -Xcompiler=/RTC1>
)