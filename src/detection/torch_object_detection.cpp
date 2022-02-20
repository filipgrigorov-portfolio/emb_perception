#include "detection/torch_object_detection.hpp"

#define DEBUG

namespace emb {
    TorchObjectDetector::TorchObjectDetector(const std::string& model_path, int img_width, int img_height)
    : img_dims_(img_width, img_height) {
        try {
            module_ = torch::jit::load(model_path);
        } catch(const c10::Error& exc) {
            std::cerr << "Could not load model" << std::endl;
        }
    }

    TorchObjectDetector::~TorchObjectDetector() {

    }

    emb::DetectionResults TorchObjectDetector::Infer(
        const cv::Mat& input, bool quantized/*=false*/, const std::pair<float, float>& range/*={0.f, 255.f}*/) {
        cv::Mat input_img = PreprocessCvMat(input, img_dims_);
        if (!quantized) {
            auto min_intensity = 0.0; 
            auto max_intensity = 0.0;
            cv::minMaxLoc(input_img, &min_intensity, &max_intensity);
            if (range.first != static_cast<float>(min_intensity) && range.second != static_cast<float>(max_intensity)) {
                input_img.convertTo(input_img, CV_32FC3, (1.f - range.first) / range.second, range.first);
            }
        }

        c10::InferenceMode guard(true);

        auto options = torch::TensorOptions().device(torch::kCPU, -1).requires_grad(false);
        options = (quantized) ? options.dtype(torch::kUInt8) : options.dtype(torch::kFloat32);

        torch::Tensor input_tensor = torch::from_blob(
            input.data, {input_img.rows, input_img.cols, input.channels()});
        input_tensors_.push_back(input_tensor);
        // Note: BHWC -> BCHW
        for (auto& tensor : input_tensors_) {
            tensor = torch::reshape(tensor, c10::IntArrayRef{1,  input.channels(), input_img.rows, input_img.cols});
#ifdef DEBUG
            std::cout << "[" << tensor.size(0) << ", " << tensor.size(1) << ", " 
                << tensor.size(2) << ", " << tensor.size(3) << "]" << std::endl;
#endif
        }

        emb::DetectionResults results;
        return results;
    }
}  // namespace emb
