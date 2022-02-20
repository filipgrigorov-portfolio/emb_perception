#pragma once
#include <vector>

namespace emb {
    struct DetectionResult {
        std::string class_lbl{""};
        float score = 0.f;
        float xmin = 0.f, ymin = 0.f, xmax = 0.f, ymax = 0.f;
    };

    using DetectionResults = std::vector<DetectionResult>;
}  // namespace emb
