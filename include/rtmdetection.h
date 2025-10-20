
#ifndef RTMDETECTION_H
#define RTMDETECTION_H

#include <string>
#include <vector>
#include "opencv2/opencv.hpp"
#include "onnxruntime_cxx_api.h"




struct Meta_det
{
    cv::Size ori_shape;                   // Original image size
    cv::Size rtmdet_shape;                // rtmdet network input size
    float ratio;                          // zoom scale
    std::vector<int> pad_param;           // Fill in parameters (up, down, left, right)
    int det_id = 0;                       // Detect Id=0, default detects Person
};



class RTMdetection
{

public:

    RTMdetection();
    ~RTMdetection();

    // load model
    bool loadModel(const std::string &detModelPath);

    // data Preprocess
    cv::Mat dataPreprocess(const cv::Mat &image);

    
    // run model
    std::vector<std::vector<float>> rtmDetection(const cv::Mat &image, float bboxThreshold, float nmsThreshold);

    // compute IOU
    float iou(const std::vector<float> & bbox_a, const std::vector<float> & bbox_b);

    // nms
    std::vector<std::vector<float>> nms(std::vector<std::vector<float>> & bboxes, float nmsThreshold);

    // draw
    void drawDetResults(cv::Mat & image, const std::vector<std::vector<float>> & nms_bboxes);

    // save
    bool saveDetResult(const cv::Mat& detImage, const std::string& detOutPath);



private:

    Meta_det meta_;                          // metadata
    Ort::Env env;                            // ONNX environment
    Ort::Session rtmDetSession;              // MMDetection session
};


#endif

