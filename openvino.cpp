#include"openvino.h"

YOLO_OPENVINO::YOLO_OPENVINO()
{
}

YOLO_OPENVINO::~YOLO_OPENVINO()
{
}


YOLO_OPENVINO::Resize YOLO_OPENVINO::resize_and_pad(cv::Mat& img, cv::Size new_shape)
{
    float width = img.cols;
    float height = img.rows;
    float r = float(new_shape.width / max(width, height));
    int new_unpadW = int(round(width * r));
    int new_unpadH = int(round(height * r));

    cv::resize(img, resize.resized_image, cv::Size(new_unpadW, new_unpadH), 0, 0, cv::INTER_AREA);

    resize.dw = new_shape.width - new_unpadW;//w����paddingֵ 
    resize.dh = new_shape.height - new_unpadH;//h����paddingֵ 
    cv::Scalar color = cv::Scalar(100, 100, 100);
    cv::copyMakeBorder(resize.resized_image, resize.resized_image, 0, resize.dh, 0, resize.dw, cv::BORDER_CONSTANT, color);

    return resize;
}

void YOLO_OPENVINO::yolov5_compiled(std::string xml_path, ov::CompiledModel& compiled_model)
{
    // Step 1. Initialize OpenVINO Runtime core
    ov::Core core;
    // Step 2. Read a model
    //std::shared_ptr<ov::Model> model = core.read_model("best.xml");
    auto model = core.read_model(xml_path);
    //compiled_model = core.compile_model(model, "CPU");
    // 
       // �޸�ģ�͵�������״��������С����Ϊ 1
    ov::Shape input_shape = model->input().get_shape();
    input_shape[0] = 1; // ������С�� 16 �޸�Ϊ 1
    model->reshape({ {model->input().get_any_name(), input_shape} });


    // Step 4. Inizialize Preprocessing for the model ��ʼ��ģ�͵�Ԥ����
    //ov::preprocess::PrePostProcessor ppp = ov::preprocess::PrePostProcessor(model);
    ov::preprocess::PrePostProcessor ppp(model);
    // Specify input image format ָ������ͼ���ʽ
    //ppp.input().tensor().set_element_type(ov::element::u8).set_layout("NHWC").set_color_format(ov::preprocess::ColorFormat::GRAY);
    ppp.input().tensor().set_element_type(ov::element::u8).set_layout("NHWC").set_color_format(ov::preprocess::ColorFormat::BGR);
    // Specify preprocess pipeline to input image without resizing ָ������ͼ���Ԥ����ܵ�����������С
    ppp.input().preprocess().convert_element_type(ov::element::f32).convert_color(ov::preprocess::ColorFormat::RGB).scale({ 255.,255.,255.});
    //ppp.input().preprocess().convert_element_type(ov::element::f32).scale({ 255.0 });
    //  Specify model's input layout ָ��ģ�͵����벼��
    ppp.input().model().set_layout("NCHW");
    // Specify output results format ָ����������ʽ
    //ppp.output().tensor().set_element_type(ov::element::f32);
    ppp.output().tensor().set_element_type(ov::element::f32);
    // Embed above steps in the graph ��ͼ����Ƕ�����ϲ���
    model = ppp.build();
    compiled_model = core.compile_model(model, "CPU");

    // ��ȡģ�͵�����������Ϣ
    auto inputs = model->inputs();
    auto outputs = model->outputs();

    // ��ӡ������Ϣ
    for (const auto& input : inputs) {
        std::cout << "Input name: " << input.get_any_name() << std::endl;
        std::cout << "Input shape: ";
        for (const auto& dim : input.get_shape()) {
            std::cout << dim << " ";
        }
        std::cout << std::endl;
        std::cout << "Input element type: " << input.get_element_type() << std::endl;
    }

    // ��ӡ�����Ϣ
    for (const auto& output : outputs) {
        std::cout << "Output name: " << output.get_any_name() << std::endl;
        std::cout << "Output shape: ";
        for (const auto& dim : output.get_shape()) {
            std::cout << dim << " ";
        }
        std::cout << std::endl;
        std::cout << "Output element type: " << output.get_element_type() << std::endl;
    }
}



void YOLO_OPENVINO::yolov5_detector(ov::CompiledModel compiled_model, cv::Mat input_detect_img, cv::Mat output_detect_img, vector<cv::Rect>& nms_box)
{
    // Step 3. Read input image
    cv::Mat img = input_detect_img.clone();
    int img_height = img.rows;
    int img_width = img.cols;
    vector<cv::Mat>images;
    vector<cv::Rect> boxes;
    vector<int> class_ids;
    vector<float> confidences;

    if (img_height < 5000 && img_width < 5000)
    {
        images.push_back(img);
    }
    else
    {
        images.push_back(img(cv::Range(0, 0.6 * img_height), cv::Range(0, 0.6 * img_width)));
        images.push_back(img(cv::Range(0, 0.6 * img_height), cv::Range(0.4 * img_width, img_width)));
        images.push_back(img(cv::Range(0.4 * img_height, img_height), cv::Range(0, 0.6 * img_width)));
        images.push_back(img(cv::Range(0.4 * img_height, img_height), cv::Range(0.4 * img_width, img_width)));
    }

    for (int m = 0; m < images.size(); m++)
    {
        auto start1 = std::chrono::system_clock::now();
        // resize image
        Resize res = resize_and_pad(images[m], cv::Size(640, 640));
        // Step 5. Create tensor from image
        // ȷ��ͼ��������������
        if (!res.resized_image.isContinuous()) {
            res.resized_image = res.resized_image.clone();
        }

        if (res.resized_image.type() != CV_8UC3) {
            // ���������ͨ��ͼ�񣬽���ת��
            cv::cvtColor(res.resized_image, res.resized_image, cv::COLOR_GRAY2BGR);
        }


        // ��ȡģ���������״
        ov::Shape input_shape = compiled_model.input().get_shape(); // ��״ӦΪ [1, H, W, C]
        std::cout << "Input shape: ";
        for (auto s : input_shape) {
            std::cout << s << " ";
        }
        std::cout << std::endl;
        // ��ӡ����������Ԫ������
        std::cout << "Input element type: " << compiled_model.input().get_element_type() << std::endl;


        // ������������
        ov::Tensor input_tensor = ov::Tensor(
            ov::element::u8, // Ԫ��������Ԥ��������ָ����һ��
            input_shape,     // ������������״
            res.resized_image.data // ͼ������ָ�룬����Ϊ unsigned char*
        );

        //float* input_data = (float*)res.resized_image.data;//���ź�ͼ������
        //ov::Tensor input_tensor = ov::Tensor(compiled_model.input().get_element_type(), compiled_model.input().get_shape(), input_data);





        // Step 6. Create an infer request for model inference 
        ov::InferRequest infer_request = compiled_model.create_infer_request();
        infer_request.set_input_tensor(input_tensor);


        //��������ʱ��
        auto start = std::chrono::system_clock::now();
        infer_request.infer();
        auto end = std::chrono::system_clock::now();
        std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;



        //Step 7. Retrieve inference results 
        const ov::Tensor& output_tensor = infer_request.get_output_tensor();
        ov::Shape output_shape = output_tensor.get_shape();
        float* detections = output_tensor.data<float>();

        for (int i = 0; i < output_shape[1]; i++)//�������п�
        {
            float* detection = &detections[i * output_shape[2]];//bbox��x y w h obj cls)

            float confidence = detection[4];//��ǰbbox��obj
            if (confidence >= CONFIDENCE_THRESHOLD) //�ж��Ƿ�Ϊǰ��
            {
                float* classes_scores = &detection[5];
                cv::Mat scores(1, output_shape[2] - 5, CV_32FC1, classes_scores);
                cv::Point class_id;
                double max_class_score;
                cv::minMaxLoc(scores, 0, &max_class_score, 0, &class_id);//�������÷ֺ�������

                if (max_class_score > SCORE_THRESHOLD)//����÷�
                {
                    confidences.push_back(confidence);

                    class_ids.push_back(class_id.x);

                    float x = detection[0];//������x 
                    float y = detection[1];//������y 
                    float w = detection[2];//49
                    float h = detection[3];//50

                    float rx = (float)images[m].cols / (float)(res.resized_image.cols - res.dw);//x����ӳ�����
                    float ry = (float)images[m].rows / (float)(res.resized_image.rows - res.dh);//y����ӳ�����

                    x = rx * x;
                    y = ry * y;
                    w = rx * w;
                    h = ry * h;

                    if (m == 0)
                    {
                        x = x;
                        y = y;
                    }
                    else if (m == 1)
                    {
                        x = x + 0.4 * img_width;
                        y = y;

                    }
                    else if (m == 2)
                    {
                        x = x;
                        y = y + 0.4 * img_height;
                    }
                    else if (m == 3)
                    {
                        x = x + 0.4 * img_width;
                        y = y + 0.4 * img_height;
                    }

                    float xmin = x - (w / 2);//bbox���Ͻ�x
                    float ymin = y - (h / 2);//bbox���Ͻ�y
                    boxes.push_back(cv::Rect(xmin, ymin, w, h));
                }
            }
        }
        auto end1 = std::chrono::system_clock::now();
        std::cout <<"��ʱ��"<< std::chrono::duration_cast<std::chrono::milliseconds>(end1 - start1).count() << "ms" << std::endl;
    }

    std::vector<int> nms_result;
    cv::dnn::NMSBoxes(boxes, confidences, SCORE_THRESHOLD, NMS_THRESHOLD, nms_result);
    std::vector<Detection> output;
    auto start2 = std::chrono::system_clock::now();
    for (int i = 0; i < nms_result.size(); i++)
    {
        Detection result;
        int idx = nms_result[i];
        result.class_id = class_ids[idx];
        result.confidence = confidences[idx];
        result.box = boxes[idx];
        nms_box.push_back(result.box);//����Qt NMS���box
        output.push_back(result);
    }


    // Step 9. Print results and save Figure with detections
    for (int i = 0; i < output.size(); i++)
    {
        auto detection = output[i];
        auto box = detection.box;
        auto classId = detection.class_id;
        auto confidence = detection.confidence;


        float xmax = box.x + box.width;
        float ymax = box.y + box.height;

        cv::rectangle(img, cv::Point(box.x, box.y), cv::Point(xmax, ymax), cv::Scalar(0, 255, 0), 3);
        cv::rectangle(img, cv::Point(box.x, box.y - 20), cv::Point(xmax, box.y), cv::Scalar(0, 255, 0), cv::FILLED);
        cv::putText(img, std::to_string(classId), cv::Point(box.x, box.y - 5), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0));

    }
    
    img.copyTo(output_detect_img);
    auto end2 = std::chrono::system_clock::now();
    std::cout << "ʣ��ʱ��" << std::chrono::duration_cast<std::chrono::milliseconds>(end2 - start2).count() << "ms" << std::endl;

}