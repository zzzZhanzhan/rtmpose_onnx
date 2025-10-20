
#include "rtmdetection.h"
#include <thread>
#include <filesystem>
#include <opencv2/imgproc.hpp>
#include <opencv2/dnn.hpp>


RTMdetection::RTMdetection() : env(ORT_LOGGING_LEVEL_ERROR, "RTMdetection"), rtmDetSession(nullptr)
{

}

RTMdetection::~RTMdetection()
{

}

bool RTMdetection::loadModel(const std::string & detModelPath)
{
    try {
    Ort::SessionOptions sessionOptions;
    sessionOptions.SetIntraOpNumThreads(std::thread::hardware_concurrency());
    sessionOptions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    
    OrtCUDAProviderOptions options;
    options.device_id = 1;                    // Specify GPU device ID
    sessionOptions.AppendExecutionProvider_CUDA(options);

    rtmDetSession = Ort::Session(env, detModelPath.c_str(), sessionOptions);          // Load model file, create session object

    return true;
    } catch (const Ort::Exception& e) {
        std::cerr << "Error: Could not load the model. " << e.what() << std::endl;
        return false;
    }
}

float RTMdetection::iou(const std::vector<float> & bbox_a, const std::vector<float> & bbox_b)
{
    float inter_x1 = std::max(bbox_a[0], bbox_b[0]);   // x1
    float inter_y1 = std::max(bbox_a[1], bbox_b[1]);   // y1
    float inter_x2 = std::min(bbox_a[2], bbox_b[2]);   // x2
    float inter_y2 = std::min(bbox_a[3], bbox_b[3]);   // y2

    // 
    float inter = std::max(0.0f, inter_x2 - inter_x1) * std::max(0.0f, inter_y2 - inter_y1);
    float area_a = (bbox_a[2] - bbox_a[0]) * (bbox_a[3] - bbox_a[1]);
    float area_b = (bbox_b[2] - bbox_b[0]) * (bbox_b[3] - bbox_b[1]);

    return inter / (area_a + area_b - inter + 1e-6f);

}

std::vector<std::vector<float>> RTMdetection::nms(std::vector<std::vector<float>> & bboxes, float nmsThreshold)
{

    if (bboxes.empty()) return {};

    // Sort by confidence level
    std::sort(bboxes.begin(), bboxes.end(),
                                [](const std::vector<float> & a, const std::vector<float> & b) { return a[4] > b[4];});

    std::vector<std::vector<float>> keep;
    std::vector<bool> nms_flag(bboxes.size(), false);

    // Suppress highly overlapping boxes
    for (size_t i = 0; i < bboxes.size(); i++)
    {
        if (nms_flag[i])
        {
            continue;
        }
        
        keep.push_back(bboxes[i]);


        for (size_t j = i + 1; j < bboxes.size(); j++)
        {
            if(!nms_flag[j] && iou(bboxes[i], bboxes[j]) > nmsThreshold)
            {
                nms_flag[j] = true;
            }
        }
        
    }
    return keep;

}


cv::Mat RTMdetection::dataPreprocess(const cv::Mat &image)
{
    // image preprocessing
    // onnx model input image size
    const int detInputH = 320;      // RTMDet-Nano 320 input Height
    const int detInputW = 320;      // RTMDet-Nano 320 input Width

    meta_.rtmdet_shape = cv::Size(detInputW, detInputH);    // Resize to target size
    meta_.ori_shape = image.size();                         // Original image size

    float ratio = std::min((float)detInputH / image.rows, (float)detInputW / image.cols);
    meta_.ratio = ratio;


    int resizeH = (int)(image.rows * ratio);   // Height
    int resizeW = (int)(image.cols * ratio);   // Width

    cv::Mat resizedImage;

    cv::resize(image, resizedImage, cv::Size(resizeW, resizeH), 0, 0, cv::INTER_LINEAR);

    // padding
    int top = (detInputH - resizeH) / 2;
    int bottom = detInputH - resizeH - top;    // Handle odd size differences
    int left = (detInputW - resizeW) / 2;
    int right = detInputW - resizeW - left;

    meta_.pad_param = {top, bottom, left, right};

    // Fill to target size
    cv::copyMakeBorder(resizedImage, resizedImage, top, bottom, left, right, 
                        cv::BORDER_CONSTANT, cv::Scalar(114, 114, 114));


    resizedImage.convertTo(resizedImage, CV_32F);        //  Convert to float

    // Normalization，(to_rgb=false)
    std::vector<float> mean = {103.53f, 116.28f, 123.675f};  // BGR 
    std::vector<float> std = {57.375f, 57.12f, 58.395f};

    // Channel normalization
    std::vector<cv::Mat> channels(3);
    cv::split(resizedImage, channels);

    for (size_t i = 0; i < 3; i++)
    {
        channels[i] = (channels[i] - mean[i]) / std[i];   
    }

    cv::Mat normImage;

    cv::merge(channels, normImage);   // Merge channels(HWC) BGR

    return normImage;

}


std::vector<std::vector<float>> RTMdetection::rtmDetection(const cv::Mat &image, float bboxThreshold, float nmsThreshold)
{
    // image preprocessing
    cv::Mat preImage = dataPreprocess(image);

    float ratio = meta_.ratio;

    // padding
    int top = meta_.pad_param[0];
    int bottom = meta_.pad_param[1];    
    int left = meta_.pad_param[2];
    int right = meta_.pad_param[3];

    // Original image size
    int ori_H = meta_.ori_shape.height;
    int ori_W = meta_.ori_shape.width;

    // Create onnx Tensor
    std::vector<int64_t> inputDims = {1, 3, preImage.rows, preImage.cols};
    std::vector<float> inputTensorValues(preImage.total() * 3);

    // HWC ---> CHW
    cv::Mat inputChannels[3];
    cv::split(preImage, inputChannels);
    for (int i = 0; i < 3; ++i) {
        std::memcpy(inputTensorValues.data() + i * preImage.total(), inputChannels[i].data, preImage.total() * sizeof(float));
    }


    // Create onnx input tensor
    Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value inputTensor = Ort::Value::CreateTensor<float>(memoryInfo, inputTensorValues.data(), inputTensorValues.size(), inputDims.data(), inputDims.size());   

    // Inference
    std::vector<const char*> inputNodeNames = {"input"};           // Replace with the actual input node name
    std::vector<const char*> outputNodeNames = {"dets", "labels"}; // Replace with the actual output node name
    auto outputTensors = rtmDetSession.Run(Ort::RunOptions{nullptr}, inputNodeNames.data(), &inputTensor, 1, outputNodeNames.data(), outputNodeNames.size());

    if (outputTensors.empty())
    {
        throw std::runtime_error("Failed to create a tensor");
    }

    // Parse Detection Bounding Boxes - Shape: [num_detections, 5]; 5 indicate [x1, y1, x2, y2, score]
    auto* dets   = outputTensors[0].GetTensorMutableData<float>();    // [N,5]   detection results
    auto* labels = outputTensors[1].GetTensorMutableData<int64_t>();  // [N]     labelsS
    auto  shape  = outputTensors[0].GetTensorTypeAndShapeInfo().GetShape();

    
    int64_t N = shape[1];        // 【batch_size, num_bboxes, 5]


    std::vector<std::vector<float>> bboxes;
    for (int64_t i = 0; i < N; ++i)
    {
        if (labels[i] != meta_.det_id) continue;    // Only save person class id = 0;
        
        float score = dets[i * 5 + 4];              // confidence score
        if (score < bboxThreshold) continue;
        
        // Coordinate mapping to the original image
        float x1 = (dets[i * 5 + 0] - left) / ratio;
        float y1 = (dets[i * 5 + 1] - top) / ratio;
        float x2 = (dets[i * 5 + 2] - left) / ratio;
        float y2 = (dets[i * 5 + 3] - top) / ratio;

        // Ensure coordinates are within the image boundaries
        x1 = std::max(0.0f, std::min(x1, (float)ori_W));
        y1 = std::max(0.0f, std::min(y1, (float)ori_H));
        x2 = std::max(0.0f, std::min(x2, (float)ori_W));
        y2 = std::max(0.0f, std::min(y2, (float)ori_H));

        bboxes.push_back({
                    x1,  // x1
                    y1,  // y1
                    x2,  // x2
                    y2,  // y2
                    score             // confidence score
                });

    }

    // std::cout << "Number of result detection boxes before nmns：" << bboxes.size() << std::endl;

    // Perform Non-Maximum Suppression (nms)
    std::vector<std::vector<float>> nms_results = nms(bboxes, nmsThreshold);

    return nms_results;

}


void RTMdetection::drawDetResults(cv::Mat & image, const std::vector<std::vector<float>> & nms_bboxes)
{
    // Draw object detection results
 
    for (auto bbox : nms_bboxes)
    {
        float x1 = bbox[0];
        float y1 = bbox[1];
        float x2 = bbox[2];
        float y2 = bbox[3];
        float score = bbox[4];

        // draw rectangle
        cv::rectangle(image, 
                      cv::Point(x1, y1),       // Top left
                      cv::Point(x2, y2),       // Top right
                      cv::Scalar(0, 0, 255),   // color
                      2);                      // line width

        // draw confidence score
        std::string label = "Person: " + cv::format("%.2f", score);

        // Label position
        int base_line = 0;
        cv::Size label_size = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &base_line);


        // Draw label background
        cv::rectangle(image, 
                      cv::Point(x1 , y1),
                      cv::Point(x1 + label_size.width + 20 , y1 + label_size.height + base_line), 
                      cv::Scalar(0, 0, 255), 
                      cv::FILLED);         // Fill rectangle

        // Draw label text (white text, font size 0.5)
        cv::putText(image, 
                   label, 
                   cv::Point(x1, y1 + label_size.height), 
                   cv::FONT_HERSHEY_SIMPLEX, 
                   0.6, 
                   cv::Scalar(255, 255, 255),  // white text
                   1.6);                       // font width

    }
    
}


bool RTMdetection::saveDetResult(const cv::Mat& detImage, const std::string& detOutPath)
{
    // Save the target detection result image
    if (detImage.empty()) {
        std::cerr << "Error: No image data to save!" << std::endl;
        return false;
    }

    if (!std::filesystem::exists(std::filesystem::path(detOutPath).parent_path()))
    {   std::cerr << "Error: Save Pose Estimation Data Path does not exist!" << std::endl;
        return false;
    }
    
    // write image
    return cv::imwrite(detOutPath, detImage);
}