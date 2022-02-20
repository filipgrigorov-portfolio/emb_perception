#pragma once
#include <string>

#include <torch/script.h>

#include "common/common_headers.hpp"
#include "detection/od_utils.hpp"
namespace emb {
    /*
        Flow: python torch -> torch script (ScriptModule) (.pt) -> c++ frontend
    */
    class TorchObjectDetector {
     public:
        TorchObjectDetector(const std::string& model_path, int img_width, int img_height);
        ~TorchObjectDetector();

        emb::DetectionResults Infer(const cv::Mat& input, bool quantized=false, const std::pair<float, float>& range={0.f, 255.f});

     private:
        std::vector<torch::Tensor> input_tensors_;

        cv::Size img_dims_;

        torch::jit::script::Module module_;
    };
}  // namespace emb
