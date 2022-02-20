#include <cstdio>
#include <iostream>
#include <filesystem>

#include "common/common_headers.hpp"
#include "detection/od_utils.hpp"
#include "detection/object_detection.hpp"

int main(void) {
    const auto cwd = std::filesystem::current_path();
    const std::string model_path{cwd / ".." / "models/ssd_mobilenet_v2_quantized_300x300_coco_2019_01_03.tflite"};
    std::unique_ptr<emb::ObjectDetector<emb::TfLiteObjectDetector>> od = 
        std::make_unique<emb::ObjectDetector<emb::TfLiteObjectDetector>>(model_path, 300, 300);

    // TODO(Filip): build test around this
    std::string img_path{cwd / ".." / "test_data/single_car.jpg"};
    std::cout << img_path << std::endl;
    cv::Mat img = cv::imread(img_path);
    if (!img.data) {
        std::cerr << "Image has not been loaded, properly" << std::endl;
        exit(1);
    }
    std::cout << "Image shape: " << img.cols << " x " << img.rows << std::endl;

    auto det_res = od->Infer(img, true);
    od->Draw(img, det_res);

    return EXIT_SUCCESS;
}
