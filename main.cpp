
#include <iostream>
#include <filesystem>
#include <opencv2/opencv.hpp>
#include <opencv2/imgcodecs.hpp>
#include "rtmdetection.h"
#include "rtmpose.h"


// Load all images from the folder
std::vector<cv::Mat> loadImagesFromFolder(const std::string& folderPath) {
    std::vector<cv::Mat> images;
    std::vector<std::string> imagePaths;


    if (! std::filesystem::exists(folderPath))
    {   std::cerr << "Error: Image Path does not exist!" << std::endl;
    }

    // Retrieve all image paths in the folder
    cv::glob(folderPath + "/*.jpg", imagePaths, false);  // djust file types as needed, such as *. png

    for (const auto& imagePath : imagePaths) {
        cv::Mat img = cv::imread(imagePath);
        if (img.empty()) {
            std::cerr << "Failed to load image: " << imagePath << std::endl;
            continue;
        }
        images.push_back(img);
    }

    return images;
}

int main() {

    // Create RTMdetection instance
    RTMdetection rtmDet;

    // Create RTMpose instance
    RTMpose rtmPose;
    

    // Load rtmdet ONNX model
    std::string rtmdetPath = "./rtmdet-ort/rtmdet-nano/end2end.onnx";

    // Load rtmpose ONNX model
    std::string rtmposePath = "./rtmpose-ort/rtmpose-mm/end2end.onnx";
    
    if (!rtmDet.loadModel(rtmdetPath)) {
        std::cerr << "Failed to load model!" << std::endl;
        return -1;
    }

    if (!rtmPose.loadModel(rtmposePath)) {
        std::cerr << "Failed to load model!" << std::endl;
        return -1;
    }

    // Load images from folder
    std::string folderPath = "./images/";
    std::vector<cv::Mat> images = loadImagesFromFolder(folderPath);

    // Threshold
    float score_thr = 0.3;     // object detection confidence threshold
    float nms_thr = 0.3;       // nms threshold
    float kpt_thr = 0.3;       // key points threshold
    
    // Save path
    std::string detSavePath = "./output_det";        // save det results
    std::string poseSavePath = "./output_pose";      // save pose results


    int count = 0;

    for (auto img : images)
    {
        cv::Mat pose_image = img.clone();
        std::vector<std::vector<cv::Point2f>> allKeypoints;

        auto det_results = rtmDet.rtmDetection(img, score_thr, nms_thr);    // rtmdet

        if (det_results.empty())
        {
            continue;
        }
        

        rtmDet.drawDetResults(img, det_results);

        // Show detection results
        // cv::imshow("Det Results", img);
        // cv::waitKey(0);

        std::string detOutPath = detSavePath + "/det_image_" + std::to_string(count + 1) + ".jpg";
        if (!rtmDet.saveDetResult(img, detOutPath))
        {
            std::cerr << "Failed to save detection result image." << std::endl;
        }
        
        for (const auto& bbox : det_results) 
        {
            auto keypoints = rtmPose.rtmPose(pose_image, bbox, kpt_thr);     // rtmpose

            if (keypoints.size() == rtmPose.getJointNum())
            {
                allKeypoints.push_back(keypoints);
            }
            else
            {
                continue;
            }
            
        }

        if (allKeypoints.empty())
        {
            continue;
        }
        
        rtmPose.drawPoseResults(pose_image, allKeypoints);

        // Show human pose estimation results
        // cv::imshow("Pose Results", pose_image);
        // cv::waitKey(0);
        std::string poseOutPath = poseSavePath + "/pose_image_"+ std::to_string(count + 1) + ".jpg";
        if (!rtmPose.savePoseResult(pose_image, poseOutPath))
        {
            std::cerr << "Failed to save pose estimation result image." << std::endl;
        }
        
        count ++;
        
    }

    return 0;
}