# Select basic image 
FROM nvcr.io/nvidia/cuda:12.2.2-cudnn8-devel-ubuntu22.04

# Set environment variables 
ENV DEBIAN_FRONTEND=noninteractive
ENV LANG=C.UTF-8
ENV PATH="/usr/local/cuda/bin:${PATH}"
ENV LD_LIBRARY_PATH="/usr/local/cuda/lib64:/usr/local/lib:${LD_LIBRARY_PATH}"

# Update package and install basic dependencies 
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    wget \
    unzip \
    pkg-config \
    libjpeg-dev \
    libpng-dev \
    libtiff-dev \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    && rm -rf /var/lib/apt/lists/*

# Download, compile, and install opencv
WORKDIR /tmp
COPY ./opencv-4.10.0 ./opencv-4.10.0
COPY ./opencv_contrib-4.10.0 ./opencv_contrib-4.10.0

RUN mkdir -p opencv-4.10.0/build && cd opencv-4.10.0/build && \
    cmake -D CMAKE_BUILD_TYPE=Release \
          -D CMAKE_INSTALL_PREFIX=/usr/local \
          -D OPENCV_EXTRA_MODULES_PATH=/tmp/opencv_contrib-4.10.0/modules \
          -D WITH_CUDA=ON \
          -D WITH_CUDNN=ON \
          -D OPENCV_DNN_CUDA=ON \
          -D ENABLE_FAST_MATH=ON \
          -D CUDA_FAST_MATH=ON \
          -D CUDA_ARCH_BIN=8.6 \      # For specific GPU models
          -D WITH_GTK=OFF \
          -D WITH_QT=OFF \
          -D WITH_OPENGL=OFF \
          -D WITH_V4L=OFF \
          -D BUILD_EXAMPLES=OFF \
          -D BUILD_DOCS=OFF \
          -D BUILD_TESTS=OFF \
          -D BUILD_PERF_TESTS=OFF \
          -D BUILD_opencv_python2=OFF \
          -D BUILD_opencv_python3=OFF .. && \
    make -j$(nproc) && \
    make install && \
    ldconfig && \
    rm -rf /tmp/opencv /tmp/opencv_contrib

# Install ONNX Runtime (C++version with CUDA support)
WORKDIR /tmp
RUN ONNXRT_VERSION=1.18.1 && \
    wget -q https://github.com/microsoft/onnxruntime/releases/download/v${ONNXRT_VERSION}/onnxruntime-linux-x64-gpu-${ONNXRT_VERSION}.tgz && \
    tar -zxf onnxruntime-linux-x64-gpu-${ONNXRT_VERSION}.tgz && \
    cp -r onnxruntime-linux-x64-gpu-${ONNXRT_VERSION}/ /mnt && \
    ldconfig && \
    rm -rf onnxruntime-linux-x64-gpu-${ONNXRT_VERSION}*


# Create a working directory
WORKDIR /app

# Copy file
COPY . .

# Set default startup command
CMD ["bash"]
