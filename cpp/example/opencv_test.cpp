#include <opencv2/imgcodecs.hpp>

#include <iostream>

int main() {
    std::string img_path = "../../cpp/example/car.jpg";

    cv::Mat img = cv::imread(img_path, cv::IMREAD_GRAYSCALE);

    cv::imwrite("../../cpp/example/ris.png", img);
        
    return 0;
}
