#include <opencv2/opencv.hpp>
#include <iostream>
#include <thread>

std::vector<int> classIds;
std::vector<float> confidences;
std::vector<cv::Rect> boxes;

cv::Mat frame;

void detection(cv::dnn::DetectionModel model) {
    while (1) {
        // Essentially just constantly get the newest frame.
        cv::Mat copyFrame;
        frame.copyTo(copyFrame);
        model.detect(copyFrame, classIds, confidences, boxes, .2, .4);
    }
}


int main(int, char**) {
    cv::VideoCapture camera(0);
    if (!camera.isOpened()) {
        std::cerr << "Camera not opened" << std::endl;
        return 1;
    }

    cv::namedWindow("Detection", cv::WINDOW_AUTOSIZE);

    cv::dnn::Net net = cv::dnn::readNetFromDarknet("yolov7-tiny.cfg", "yolov7-tiny.weights");
    cv::dnn::DetectionModel model = cv::dnn::DetectionModel(net);

    model.setInputParams(1./255, cv::Size(416, 416), cv::Scalar(), true);

    cv::Mat last_frame;
    camera >> frame;
    std::thread th1(detection, model);
    while (1) {
        camera >> frame;
        for (int i = 0; i < classIds.size(); ++i) {

            auto box = boxes[i];
            if (classIds[i] == 0) {
                cv::rectangle(frame, box, (0, 255, 0), 3);
                cv::rectangle(frame, cv::Point(box.x, box.y - 20), cv::Point(box.x + box.width, box.y), (0,0,255), cv::FILLED);
                cv::putText(frame, "Person", cv::Point(box.x, box.y - 5), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0));
            }
        }
        cv::imshow("Detection", frame);
        
        if (cv::waitKey(1) == 27)
            break;
    }
    return 0;
}
