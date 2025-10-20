
#include <iostream>
#include <filesystem>
#include <opencv2/opencv.hpp>
#include <opencv2/imgcodecs.hpp>
#include "rtmdetection.h"
#include "rtmpose.h"




int main() {
    
    // Create instance
    RTMdetection rtmDet;
    RTMpose rtmPose;
    

    // ONNX model file path
    std::string rtmdetPath = "./rtmdet-ort/rtmdet-nano/end2end.onnx";
    std::string rtmposePath = "./rtmpose-ort/rtmpose-mm/end2end.onnx";
    
    // Load onnx model
    if (!rtmDet.loadModel(rtmdetPath)) {
        std::cerr << "Failed to load model!" << std::endl;
        return -1;
    }

    if (!rtmPose.loadModel(rtmposePath)) {
        std::cerr << "Failed to load model!" << std::endl;
        return -1;
    }


    // Video file path
    std::string videoPath = "./videos/walk.mp4";

    // Threshold
    float score_thr = 0.3;
    float nms_thr = 0.3;
    float kpt_thr = 0.3;
    
    std::string detSavePath = "./output_det";
    std::string poseSavePath = "./output_pose";
    int count = 0;


    // Open video file
    cv::VideoCapture cap(videoPath);

    if (!cap.isOpened())
    {
        std::cerr << "Error: Could not open video file " << videoPath << std::endl;
        return 0;
    }

    int frameCount = 0;
    cv::Mat frame;

    // Obtain and calculate frame rate
    double fps = cap.get(cv::CAP_PROP_FPS);
    if (fps <= 0) {
        fps = 30.0;           // If the frame rate cannot be obtained, use a default value 30
    }
    int delay_ms = static_cast<int>(1000.0 / fps);
    std::cout << "Video FPS: " << fps << ", Frame delay: " << delay_ms << " ms" << std::endl;

    // Process videos frame by frame
    while (cap.read(frame))
    {
        if (frame.empty()) break;

        cv::Mat pose_image = frame.clone();
        std::vector<std::vector<cv::Point2f>> allKeypoints;

        // object detection
        auto det_results = rtmDet.rtmDetection(frame, score_thr, nms_thr);

        if (!det_results.empty())
        {
            // draw detection results
            rtmDet.drawDetResults(frame, det_results);
            
            // save detection image
            std::string detOutPath = detSavePath + "/det_frame_" + 
                std::to_string(frameCount + 1) + ".jpg";
            if (!rtmDet.saveDetResult(frame, detOutPath))
            {
                std::cerr << "Failed to save detection result for frame " << frameCount + 1 << std::endl;
            }
            
            // pose estimation
            for (const auto& bbox : det_results) 
            {
                auto keypoints = rtmPose.rtmPose(pose_image, bbox, kpt_thr);

                if (keypoints.size() == rtmPose.getJointNum())
                {
                    allKeypoints.push_back(keypoints);
                }
            }

            if (!allKeypoints.empty())
            {
                // draw pose results
                rtmPose.drawPoseResults(pose_image, allKeypoints);

                cv::imshow("Pose Results", pose_image);
                cv::waitKey(delay_ms);
                
                // save pose image
                std::string poseOutPath = poseSavePath + "/pose_frame_" + 
                    std::to_string(frameCount + 1) + ".jpg";
                if (!rtmPose.savePoseResult(pose_image, poseOutPath))
                {
                    std::cerr << "Failed to save pose estimation result for frame " << frameCount + 1 << std::endl;
                }
            }
        }
        
        frameCount++;
    }

    cap.release();
    std::cout << "Video processing complete. Total frames processed: " << frameCount << std::endl;
    
    return 0;
}