cmake_minimum_required(VERSION 3.16)

# Set C++ standards
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set CUDA architecture and compiler path
set(CMAKE_CUDA_ARCHITECTURES "86")
set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)

# Project name 
project(RTMPOSE_ONNX LANGUAGES CXX CUDA)

# Set third-party library path
set(3RD_PARTY_PATH "/mnt")

# Set build type and compilation flags
set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_CXX_FLAGS_DEBUG "-Wno-deprecated-declarations -Wno-return-type -g -O0")
set(CMAKE_C_FLAGS_DEBUG "-Wno-deprecated-declarations -Wno-return-type -g -O0")
set(CMAKE_CXX_FLAGS_RELEASE "-Wno-error=deprecated-declarations -Wno-deprecated-declarations -O3")
set(CMAKE_C_FLAGS_RELEASE "-Wno-error=deprecated-declarations -Wno-deprecated-declarations -O3")

# Find and include OpenCV
find_package(OpenCV REQUIRED)

# cuda
include_directories(/usr/local/cuda/include)
link_directories(/usr/local/cuda/lib64)

# Set ONNX Runtime Path
set(ONNXRUNTIME_ROOT_PATH "${3RD_PARTY_PATH}/onnxruntime-linux-x64-gpu-1.18.1")
set(ONNXRUNTIME_INCLUDE_DIRS "${ONNXRUNTIME_ROOT_PATH}/include")
set(ONNXRUNTIME_LIB "${ONNXRUNTIME_ROOT_PATH}/lib/libonnxruntime.so.1.18.1")


# Add executable files and their source files
add_executable(demo
        main.cpp                     # images
        # main_video.cpp             # video 
        src/rtmdetection.cpp
        src/rtmpose.cpp
)

# Specify the directory containing the target
target_include_directories(demo PRIVATE
        include
        ${OpenCV_INCLUDE_DIRS}
        /usr/local/cuda/include
        ${ONNXRUNTIME_INCLUDE_DIRS}

)

# Specify link directory
link_directories(
        /usr/local/cuda/lib64
        ${ONNXRUNTIME_ROOT_PATH}/lib
)

# Link the required libraries
target_link_libraries(demo PRIVATE
        ${OpenCV_LIBS}
        ${ONNXRUNTIME_LIB}
        cudart
)
