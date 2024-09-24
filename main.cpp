#include"openvino.h"
#include<string>
#include<vector>
#include<iostream>
using namespace cv;
using namespace std;
YOLO_OPENVINO yolo_openvino;
std::string path = "D:\\best\\best.xml";
ov::CompiledModel model;
cv::Mat input_img, output_img;
vector<cv::Rect>output_box;
int main()
{
    String path1 = "D:\\best\\images\\";//文件夹路径
    vector<String>src_test;
    glob(path1, src_test, false);//将文件夹路径下的所有图片路径保存到src_test中

    if (src_test.size() == 0) {//判断文件夹里面是否有图片
        printf("error!!!\n");
        exit(1);
    }
    /* input_img = cv::imread("3.jpg");
     yolo_openvino.yolov5_compiled(path, model);
     yolo_openvino.yolov5_detector(model, input_img, output_img, output_box);*/

    yolo_openvino.yolov5_compiled(path, model);
    for (int i = 0; i < src_test.size(); i++) {//依照顺序读取文价下面的每张图片，并显示
        int pos = src_test[i].find_last_of("\\");
        std::string img_name(src_test[i].substr(pos + 1));
        Mat frame = imread(src_test[i]);
        yolo_openvino.yolov5_detector(model, frame, output_img, output_box);
        for (int i = 0; i < output_box.size(); i++)
        {
            cv::rectangle(frame, cv::Point(output_box[i].x, output_box[i].y), cv::Point(output_box[i].x + output_box[i].width, output_box[i].y + output_box[i].height), cv::Scalar(0, 255, 0), 3);
        }
        output_box.clear();
        cv::imwrite("D:\\best\\output\\" + img_name, frame);

    }


}