#include "rtmpose.h"
#include <thread>
#include <filesystem>
#include <opencv2/imgproc.hpp>
#include <opencv2/dnn.hpp>



RTMpose::RTMpose() : env(ORT_LOGGING_LEVEL_ERROR, "RTMpose"), rtmPoseSession(nullptr)
{

}

RTMpose::~RTMpose()
{

}

bool RTMpose::loadModel(const std::string &poseModelPath)
{
    try {
    Ort::SessionOptions sessionOptions;
    sessionOptions.SetIntraOpNumThreads(std::thread::hardware_concurrency());
    sessionOptions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

    OrtCUDAProviderOptions options;
    options.device_id = 1;                // Specify GPU device ID
    sessionOptions.AppendExecutionProvider_CUDA(options);

    rtmPoseSession = Ort::Session(env, poseModelPath.c_str(), sessionOptions);          // Load model file, create session object

    return true;
    } catch (const Ort::Exception& e) {
        std::cerr << "Error: Could not load the model. " << e.what() << std::endl;
        return false;
    }
}


cv::Mat RTMpose::dataPreprocess(const cv::Mat &image, const std::vector<float> & bbox)
{
    // image preprocessing
    // onnx model input image size
    int detInputW = 192;      // RTMDet-Nano 192 witdh
    int detInputH = 256;      // RTMDet-Nano 256 height

    meta.input_shape = cv::Size(detInputW, detInputH);    // target size
    meta.ori_shape = image.size();                        // Original image size


    float x1 = bbox[0], y1 = bbox[1], x2 = bbox[2], y2 = bbox[3];


    // Object detection box
    float bbox_w = x2 - x1;   // detection box width
    float bbox_h = y2 - y1;   // detection box height

    // center
    float center_x = x1 + bbox_w / 2;
    float center_y = y1 + bbox_h / 2;
    meta.center = cv::Point2f(center_x, center_y);


    float pad_w = bbox_w * meta.padding;       // expand image size
    float pad_h = bbox_h * meta.padding;      

    // Calculate scaling ratio
    float ratio = std::max(pad_w / detInputW, pad_h / detInputH) * 200.f;

    // Calculate the affine transformation matrix
    float trans_ratio = 200.f / ratio;         // transform ratio
    meta.ratio = trans_ratio;
    cv::Mat affine_mat = cv::getRotationMatrix2D(cv::Point2f(center_x, center_y), 0.0, trans_ratio);  // transform matrix
    affine_mat.at<double>(0, 2) += detInputW  * 0.5 - center_x;
    affine_mat.at<double>(1, 2) += detInputH * 0.5 - center_y;

    // Affine transformation (Crop + Scale + Translate). 
    // Accurately and distortion free extraction of human regions detected in the original image.
    cv::Mat affineTrans_image;
    cv::warpAffine(image, affineTrans_image, affine_mat, meta.input_shape, 
        cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));


    // Normalization
    affineTrans_image.convertTo(affineTrans_image, CV_32F);
    if (meta.to_rgb = true) {
        cv::cvtColor(affineTrans_image, affineTrans_image, cv::COLOR_BGR2RGB);
    }

    std::vector<float> mean = {123.675f, 116.28f, 103.53f};  // RGB 
    std::vector<float> std = {58.395f, 57.12f, 57.375f};

    std::vector<cv::Mat> channels(3);
    cv::split(affineTrans_image, channels);

    for (size_t i = 0; i < 3; i++)
    {
        channels[i] = (channels[i] - mean[i]) / std[i];   
    }

    cv::Mat norm_image;
    cv::merge(channels, norm_image);   // Merge channels RGB

    return norm_image;

}

std::vector<cv::Point2f> RTMpose::rtmPose(const cv::Mat& image, const std::vector<float> & bbox, float keypointThreshold)
{
    cv::Mat pre_image = dataPreprocess(image, bbox);

    std::vector<int64_t> inputDims = {1, 3, pre_image.rows, pre_image.cols};
    std::vector<float> inputTensorValues(pre_image.total() * 3);

    cv::Mat inputChannels[3];
    cv::split(pre_image, inputChannels);
    for (int i = 0; i < 3; ++i) {
        std::memcpy(inputTensorValues.data() + i * pre_image.total(), inputChannels[i].data, pre_image.total() * sizeof(float));
    }

    // Create onnx input tensor
    Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value inputTensor = Ort::Value::CreateTensor<float>(memoryInfo, inputTensorValues.data(), inputTensorValues.size(), inputDims.data(), inputDims.size());

    // // Inference
    std::vector<const char*> inputNodeNames = {"input"};               // Replace with the actual input node name
    std::vector<const char*> outputNodeNames = {"simcc_x", "simcc_y"}; // Replace with the actual output node name
    auto outputTensors = rtmPoseSession.Run(Ort::RunOptions{nullptr}, inputNodeNames.data(), &inputTensor, 1, outputNodeNames.data(), outputNodeNames.size());

    // Parse output tensor
    std::vector<int64_t> simcc_x_dims = outputTensors[0].GetTensorTypeAndShapeInfo().GetShape();
    std::vector<int64_t> simcc_y_dims = outputTensors[1].GetTensorTypeAndShapeInfo().GetShape();

    int joint_num = simcc_x_dims[1];          // Number of key points
    int extend_width = simcc_x_dims[2];       // Expand dimensions
    int extend_height = simcc_y_dims[2];      // Expand dimensions

    

    float* simcc_x_result = outputTensors[0].GetTensorMutableData<float>();  //【1, N, extend_width】 
    float* simcc_y_result = outputTensors[1].GetTensorMutableData<float>();

    std::vector<cv::Point2f> keypoints;
    for (int i = 0; i < joint_num; ++i) {

        // Processing the X-coordinate
        // The maximum value iter of the i-th joint point
        auto x_biggest_iter = std::max_element(simcc_x_result + i * extend_width, simcc_x_result + i * extend_width + extend_width);
        int max_x_pos = std::distance(simcc_x_result + i * extend_width, x_biggest_iter);  // maximum value element index
        int pose_x = max_x_pos / meta.simcc_split_ratio;    // x coordinate
        float score_x = *x_biggest_iter;                    // confidence score of the x-axis of the i-th joint point

        // Processing the Y-coordinate
        auto y_biggest_iter = std::max_element(simcc_y_result + i * extend_height, simcc_y_result + i * extend_height + extend_height);
        int max_y_pos = std::distance(simcc_y_result + i * extend_height, y_biggest_iter);
        int pose_y = max_y_pos / meta.simcc_split_ratio;
        float score_y = *y_biggest_iter;

        float score = std::min(score_x, score_y);   // Ensure both x_point and y_point are reliable simultaneously.

        // Filter low confidence points 
        if (score > keypointThreshold) {
            cv::Point2f temp_point;

            // Coordinate inverse transformation, returning the original image coordinates
            temp_point.x = pose_x * 1.0 / meta.ratio + meta.center.x - meta.input_shape.width * 0.5f / meta.ratio; 
            temp_point.y = pose_y * 1.0 / meta.ratio + meta.center.y - meta.input_shape.height * 0.5f / meta.ratio;
            keypoints.emplace_back(temp_point);
        }
    }

    // std::cout << "Number of key points：" << keypoints.size() << std::endl;

    return keypoints;
}

void RTMpose::drawPoseResults(cv::Mat & image, const std::vector<std::vector<cv::Point2f>> & allKeypoints)
{

    // Define the connection relationships between skeletal points
    std::vector<std::pair<int, int>> skeleton_links {{15, 13}, {13, 11}, {16, 14}, {14, 12}, {11, 12}, 
        {5, 11}, {6, 12}, {5, 6}, {5, 7}, {6, 8}, {7, 9}, 
        {8, 10}, {1, 2}, {0, 1}, {0, 2}, {1, 3}, {2, 4}, {3, 5}, {4, 6}}; 

    for (auto keypoint : allKeypoints)
    {

        // Draw skeletal points
        for (int i = 0; i < keypoint.size(); i++) 
        {
            if (i < KEYPOINT_COLORS.size())
            {
                const auto& color = KEYPOINT_COLORS[i];
                cv::circle(image, keypoint[i], 3, cv::Scalar(color[2], color[1], color[0]), -1);  
            }

        }

        // Drawing skeletal edge connections
        for(int j = 0; j < skeleton_links.size(); j++)
        {
            const auto& link = skeleton_links[j];
            if(link.first < keypoint.size() && link.second < keypoint.size() &&
                j < SKELETON_LINK_COLORS.size())
            {
                const auto& color = SKELETON_LINK_COLORS[j];
                cv::line(image, keypoint[link.first], keypoint[link.second], cv::Scalar(color[2], color[1], color[0]), 1.5);
            }
        }
    }

}


bool RTMpose::savePoseResult(const cv::Mat& poseImage, const std::string& poseOutPath) 
{

    // Save posture detection result image
    if (poseImage.empty()) {
        std::cerr << "Error: No image data to save!" << std::endl;
        return false;
    }

    if (!std::filesystem::exists(std::filesystem::path(poseOutPath).parent_path()))
    {   std::cerr << "Error: Save Detection data Path does not exist!" << std::endl;
        return false;
    }
    
    return cv::imwrite(poseOutPath, poseImage);
}