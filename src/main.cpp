#include <cstdio>
#include <iostream>

#include "common/common_headers.hpp"
#include "detection/detection.hpp"

int main(void) {
    const std::string model_path{""};
    std::unique_ptr<emb::ObjectDetector> od = std::make_unique<emb::ObjectDetector>(model_path);

    return EXIT_SUCCESS;
}
