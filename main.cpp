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
    String path1 = "D:\\best\\images\\";//�ļ���·��
    vector<String>src_test;
    glob(path1, src_test, false);//���ļ���·���µ�����ͼƬ·�����浽src_test��

    if (src_test.size() == 0) {//�ж��ļ��������Ƿ���ͼƬ
        printf("error!!!\n");
        exit(1);
    }
    /* input_img = cv::imread("3.jpg");
     yolo_openvino.yolov5_compiled(path, model);
     yolo_openvino.yolov5_detector(model, input_img, output_img, output_box);*/

    yolo_openvino.yolov5_compiled(path, model);
    for (int i = 0; i < src_test.size(); i++) {//����˳���ȡ�ļ������ÿ��ͼƬ������ʾ
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