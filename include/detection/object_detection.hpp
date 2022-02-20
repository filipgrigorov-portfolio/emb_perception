#pragma once
#include <memory>

#include "detection/od_utils.hpp"
// List of detectors
#include "detection/torch_object_detection.hpp"
#include "detection/tflite_object_detection.hpp"

namespace emb {
    template <typename OD>
    class ObjectDetector {
     public:
        ObjectDetector(const std::string& model_path, int img_width, int img_height) {
            pImplOD_ = std::make_unique<OD>(model_path, img_width, img_height);
        }
        ~ObjectDetector() = default;

        void Draw(const cv::Mat& input, const emb::DetectionResults& results) const {
            if (pImplOD_) {
                pImplOD_->Draw(input, results);
            }
        }

        emb::DetectionResults Infer(const cv::Mat& input, bool quantized) {
            if (pImplOD_) {
                return pImplOD_->Infer(input, quantized);
            }
            return emb::DetectionResults();
        }

     private:
        std::unique_ptr<OD> pImplOD_;
    };
}  // namespace
