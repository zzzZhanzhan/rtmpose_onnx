
#ifndef RTMPOSE_H
#define RTMPOSE_H

#include <string>
#include <vector>
#include "opencv2/opencv.hpp"
#include "onnxruntime_cxx_api.h"


struct Meta_pose
{
    cv::Size ori_shape;                 // Original image size
    cv::Size input_shape;               // rtmpose input size
    float ratio;                        // ratio 
    cv::Point2f center;                 // center coordinates
    float padding = 1.5;                // 
    int simcc_split_ratio = 2;
    bool to_rgb;  
    int joint_num = 17;                     
};



class RTMpose
{
private:
    Meta_pose meta;
    Ort::Env env;                            // ONNX environment
    Ort::Session rtmPoseSession;             // rtmpose
public:
    RTMpose();
    ~RTMpose();

    int getJointNum () const
    {
        return meta.joint_num;
    }

    bool loadModel(const std::string &poseModelPath);

    // data preprocess （getBboxCenterScale + Affine + Normalize + ImageToTensor）
    cv::Mat dataPreprocess(const cv::Mat &image_roi, const std::vector<float> & bbox);

    // rtmpose
    std::vector<cv::Point2f> rtmPose(const cv::Mat &image, const std::vector<float> & bbox, float keypointThreshold);
    
    // draw
    void drawPoseResults(cv::Mat & image, const std::vector<std::vector<cv::Point2f>> & allKeypoints);

    // save
    bool savePoseResult(const cv::Mat& poseImage, const std::string& poseOutPath);



    // skeleton coordinate points color (RGB)
    const std::vector<std::array<int, 3>> KEYPOINT_COLORS = {
        {51, 153, 255},  // 0
        {51, 153, 255},  // 1
        {51, 153, 255},  // 2
        {51, 153, 255},  // 3
        {51, 153, 255},  // 4
        {0, 255, 0},     // 5
        {255, 128, 0},   // 6
        {0, 255, 0},     // 7
        {255, 128, 0},   // 8
        {0, 255, 0},     // 9
        {255, 128, 0},   // 10
        {0, 255, 0},     // 11
        {255, 128, 0},   // 12
        {0, 255, 0},     // 13
        {255, 128, 0},   // 14
        {0, 255, 0},     // 15
        {255, 128, 0}    // 16
    };

    // skeleton edge color (RGB)
    const std::vector<std::array<int, 3>> SKELETON_LINK_COLORS = {
        {0, 255, 0},      // 0: 15-13
        {0, 255, 0},      // 1: 13-11
        {255, 128, 0},    // 2: 16-14
        {255, 128, 0},    // 3: 14-12
        {51, 153, 255},   // 4: 11-12
        {51, 153, 255},   // 5: 5-11
        {51, 153, 255},   // 6: 6-12
        {51, 153, 255},   // 7: 5-6
        {0, 255, 0},      // 8: 5-7
        {255, 128, 0},    // 9: 6-8
        {0, 255, 0},      // 10: 7-9
        {255, 128, 0},    // 11: 8-10
        {51, 153, 255},   // 12: 1-2
        {51, 153, 255},   // 13: 0-1
        {51, 153, 255},   // 14: 0-2
        {51, 153, 255},   // 15: 1-3
        {51, 153, 255},   // 16: 2-4
        {51, 153, 255},   // 17: 3-5
        {51, 153, 255}    // 18: 4-6
    };

};


#endif