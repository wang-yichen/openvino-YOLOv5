#pragma once
#pragma once
#pragma once
#pragma once
#include <opencv2/dnn.hpp>
#include <openvino/openvino.hpp>
#include <opencv2/opencv.hpp>

using namespace std;

class YOLO_OPENVINO
{
public:
    YOLO_OPENVINO();
    ~YOLO_OPENVINO();

public:
    struct Detection
    {
        int class_id;
        float confidence;
        cv::Rect box;
    };

    struct Resize
    {
        cv::Mat resized_image;
        int dw;
        int dh;
    };

    Resize resize_and_pad(cv::Mat& img, cv::Size new_shape);
    void yolov5_compiled(std::string xml_path, ov::CompiledModel& compiled_model);
    void yolov5_detector(ov::CompiledModel compiled_model, cv::Mat input_detect_img, cv::Mat output_detect_img, vector<cv::Rect>& nms_box);

private:

    const float SCORE_THRESHOLD = 0.4;
    const float NMS_THRESHOLD = 0.4;
    const float CONFIDENCE_THRESHOLD = 0.1;

    //vector<cv::Mat>images;//Í¼ÏñÈÝÆ÷ 
    //vector<cv::Rect> boxes;
    //vector<int> class_ids;
    //vector<float> confidences;
    //vector<cv::Rect>output_box;
    Resize resize;

};