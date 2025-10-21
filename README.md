## RTMPose ONNX Deployment

#### Introduction

RTMPose_ONNX converts pre-trained RTMPose and RTMDet models into ONNX models and deploys them using C++.


#### Requirement

- CUDA
- [ONNX Runtime](https://github.com/microsoft/onnxruntime)
- [OpenCV](https://github.com/opencv/opencv)

This project uses cuda，onnxruntime-linux-x64-gpu-1.18.1 and opencv-4.10.0. The compatibility between ONNX Runtime and CUDA at https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html#requirements.



#### Model Conversion

```shell
# go to the mmdeploy folder
cd ${PATH_TO_MMDEPLOY}

# run the command to convert RTMDet
# Model file can be either a local path or a download link
python tools/deploy.py \
    configs/mmdet/detection/detection_onnxruntime_static.py \
    ../mmpose/projects/rtmpose/rtmdet/person/rtmdet_nano_320-8xb32_coco-person.py \
    https://download.openmmlab.com/mmpose/v1/projects/rtmpose/rtmdet_nano_8xb32-100e_coco-obj365-person-05d8511e.pth \
    demo/resources/human-pose.jpg \
    --work-dir rtmpose-ort/rtmdet-nano \
    --device cpu \
    --show \
    --dump-info  # dump sdk info

# run the command to convert RTMPose
# Model file can be either a local path or a download link
python tools/deploy.py \
    configs/mmpose/pose-detection_simcc_onnxruntime_dynamic.py \
    ../mmpose/projects/rtmpose/rtmpose/body_2d_keypoint/rtmpose-m_8xb256-420e_coco-256x192.py \
    https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-m_simcc-aic-coco_pt-aic-coco_420e-256x192-63eb25f7_20230126.pth \
    demo/resources/human-pose.jpg \
    --work-dir rtmpose-ort/rtmpose-m \
    --device cpu \
    --show \
    --dump-info  # dump sdk info
```

MMPose and MMDeploy installation, as well as other detailed information, can be referred to `mmpose/projects.rtmpose/README.md` at https://github.com/open-mmlab/mmpose/blob/main/projects/rtmpose/README.md



#### Demo

You can run the following command.

```cmake
# Compile and Build C++Projects
cd rtmpose_onnx-main
mkdir build
cd build && cmake ..
make

# Run demo (Ensure image path)
cd ..
./build/demo

# You can modify the add.exe command in CmakeLists add_executable (main_video.cpp)
# to achieve real-time human pose estimation in videos.
```



#### Docker 

You can run the following command to implement Docker development.

```dockerfile
# Build image
docker build -t your_image_name .

# Run container (using GPU)
docker run -it --gpus all --name your_container_name your_image_name bash
```

Of course, you can modify the Dockerfile to meet your needs. If you modify NVIDIA Image, it will be visible at https://catalog.ngc.nvidia.com/.



