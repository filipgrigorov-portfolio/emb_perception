#pragma once

#include <string>

#include "common/common_headers.hpp"

#include "common.h"
#include "c_api.h"

namespace emb {
    template <typename T>
    using DetectionResults = std::vector<T>;

    class ObjectDetector {
     public:
        struct DetectionResult {
            std::string class_lbl{""};
            float score = 0.f;
            float xmin = 0.f, ymin = 0.f, xmax = 0.f, ymax = 0.f;
        };

        ObjectDetector(const std::string& model_path);
        ~ObjectDetector();

        DetectionResults<DetectionResult> Infer(const cv::Mat& input);

     private:
        // Note: Order of destruction
        struct TfLiteInterpreterDeleter { void operator()(TfLiteInterpreter* obj) { TfLiteInterpreterDelete(obj); } };
        struct TfLiteInterpreterOptionsDeleter { void operator()(TfLiteInterpreterOptions* obj) { TfLiteInterpreterOptionsDelete(obj); } };
        struct TfLiteModelDeleter { void operator()(TfLiteModel* obj) { TfLiteModelDelete(obj); } };

        // Note: Order of construction
        std::shared_ptr<TfLiteModel> tf_model_;
        std::shared_ptr<TfLiteInterpreterOptions> tf_opts_;
        std::shared_ptr<TfLiteInterpreter> tf_interpreter_;

        struct TfLiteTensorDeleter { void operator()(TfLiteTensor* obj) { TfLiteTensorFree(obj); } };
        std::shared_ptr<TfLiteTensor> input_tensor_;
    };
}  // namespace emb
