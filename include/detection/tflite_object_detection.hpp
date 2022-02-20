#pragma once

#include <string>

#include "common/common_headers.hpp"
#include "detection/od_utils.hpp"

#include "common.h"
#include "c_api.h"

//TODO(Filip): write abstract class (use dynamic cast for appropriate algorithm)
// or
//TODO(Filip): use pimpl for the algorithms (choose appropriate one)

namespace emb {
    class TfLiteObjectDetector {
     public:
        struct DetectionResult {
            std::string class_lbl{""};
            float score = 0.f;
            float xmin = 0.f, ymin = 0.f, xmax = 0.f, ymax = 0.f;
        };

        TfLiteObjectDetector(const std::string& model_path, int img_width, int img_height);
        ~TfLiteObjectDetector();

        void Draw(const cv::Mat& input, const emd::DetectionResults<DetectionResult>& results) const;

        emd::DetectionResults<DetectionResult> Infer(const cv::Mat& input, bool quantized);

     private:
        // Note: RGB [-1, 1] (not quantized)
        std::pair<void*, size_t> AllocateCvMat(const cv::Mat& img, const std::pair<float, float>& range={0.f, 255.f});
        std::pair<void*, size_t> AllocateQuantizedCvMat(const cv::Mat& img);

        cv::Mat PreprocessCvMat(const cv::Mat& img);

        cv::Size img_dims_;

        char* data_ = nullptr;

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