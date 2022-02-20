#pragma once
#include <vector>

#include "common/common_headers.hpp"

namespace emb {
    struct DetectionResult {
        std::string class_lbl{""};
        float score = 0.f;
        float xmin = 0.f, ymin = 0.f, xmax = 0.f, ymax = 0.f;
    };

    using DetectionResults = std::vector<DetectionResult>;

    void Draw(const cv::Mat& input, const emb::DetectionResults& results);

    cv::Mat PreprocessCvMat(const cv::Mat& img, const cv::Size& new_img_dims);
}  // namespace emb
