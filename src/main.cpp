#include <cstdio>
#include <iostream>
#include <filesystem>

#include "common/common_headers.hpp"
#include "detection/detection.hpp"

int main(void) {
    const auto cwd = std::filesystem::current_path();
    const std::string model_path{cwd / ".." / "models/ssd_mobilenet_v2_fpnlite_320x320.tflite"};
    std::unique_ptr<emb::ObjectDetector> od = std::make_unique<emb::ObjectDetector>(model_path, 320, 320);

    // TODO(Filip): build test around this
    std::string img_path{cwd / ".." / "test_data/single_image_test.jpeg"};
    std::cout << img_path << std::endl;
    cv::Mat img = cv::imread(img_path);

    auto det_res = od->Infer(img);

    return EXIT_SUCCESS;
}
