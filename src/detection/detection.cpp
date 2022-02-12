#include <cassert>
#include <fstream>
#include <set>
#include <stdexcept>

#include <iostream>

#include "detection/detection.hpp"

#define DEBUG

// Note: Works for release and debug
#define ASSERT(obj, MSG) \
if (!obj) throw std::runtime_error(MSG); \

namespace {
    void PrintShape(const std::set<int32_t>& shape, const std::string& name) {
        std::string print = name + ": [";
        for (const auto& dim : shape) {
            print += dim + " ";
        }
        print += " ]";
        std::cout << print << std::endl;
    }

    std::pair<void*, size_t> ProcessCvMat(const cv::Mat& input, const cv::Size& img_res) {
        cv::Mat input_img;
        cv::resize(input, input_img, img_res, cv::INTER_AREA);
        int mat_type = input.type();
        size_t img_size = 0;
        size_t size_per_channel = input.rows * input.step;
        if (mat_type == CV_8UC1) {
            img_size = sizeof(uint8_t) * size_per_channel;
        } else if (mat_type == CV_8UC3) {
            img_size = sizeof(uint8_t) * size_per_channel * input.channels();
        } else if (mat_type == CV_16UC1) {
            img_size = sizeof(uint16_t) * size_per_channel;
        } else if (mat_type == CV_16UC3) {
            img_size = sizeof(uint16_t) * size_per_channel * input.channels();
        }

        return std::make_pair((void*)std::move(input.data), img_size);
    }
}  //

namespace emb {
    ObjectDetector::ObjectDetector(const std::string& model_path, int img_width, int img_height)
     : img_res_(img_width, img_height) {
        if (model_path.empty()) throw std::runtime_error("Model path is empty");

        std::ifstream file(model_path, std::ifstream::binary);

        if (!file.is_open() || !file.good()) {
            throw std::runtime_error("Model file has not been open");
        }

        file.seekg(0, file.end);
        int model_bytes_len = file.tellg();
        file.seekg(0, file.beg);

        std::cout << "Loaded tf_lite model: " << model_bytes_len * 1e-6 << " MB" << std::endl;

        char* model_bytes = (char*)malloc(sizeof(char) * model_bytes_len);
        file.read(model_bytes, model_bytes_len);

        tf_model_ = std::shared_ptr<TfLiteModel>(
            TfLiteModelCreate((void*)model_bytes, model_bytes_len), 
            ObjectDetector::TfLiteModelDeleter());

        ASSERT(tf_model_, "TfLite model is not correct")

        tf_opts_ = std::shared_ptr<TfLiteInterpreterOptions>(
            TfLiteInterpreterOptionsCreate(), 
            ObjectDetector::TfLiteInterpreterOptionsDeleter());
        TfLiteInterpreterOptionsSetNumThreads(tf_opts_.get(), 2);

        ASSERT(tf_opts_, "TfLite opts are not correct")

        tf_interpreter_  = std::shared_ptr<TfLiteInterpreter>(
            TfLiteInterpreterCreate(tf_model_.get(), tf_opts_.get()), 
            ObjectDetector::TfLiteInterpreterDeleter());

        ASSERT(tf_interpreter_, "TfLite interpreter is not correct")

        TfLiteInterpreterAllocateTensors(tf_interpreter_.get());

        input_tensor_ = std::shared_ptr<TfLiteTensor>(
            TfLiteInterpreterGetInputTensor(tf_interpreter_.get(), 0), 
            ObjectDetector::TfLiteTensorDeleter());
    }

    ObjectDetector::~ObjectDetector() {
        if (data_) {
            free(data_);
        }
    }

    DetectionResults<ObjectDetector::DetectionResult> ObjectDetector::Infer(const cv::Mat& input) {
        assert(input.data);

        auto img2size = ProcessCvMat(input, img_res_);

        TfLiteTensorCopyFromBuffer(input_tensor_.get(), (void*)img2size.first, img2size.second);
        TfLiteInterpreterInvoke(tf_interpreter_.get());

        // Note: const *, does not need to be free'ed
        const auto number_detections = TfLiteInterpreterGetOutputTensor(tf_interpreter_.get(), 0);
        const auto output_bboxes = TfLiteInterpreterGetOutputTensor(tf_interpreter_.get(), 1);
        const auto output_scores = TfLiteInterpreterGetOutputTensor(tf_interpreter_.get(), 2);
        const auto output_classes = TfLiteInterpreterGetOutputTensor(tf_interpreter_.get(), 3);

        const auto number_detections_dims = TfLiteTensorNumDims(number_detections);
        const auto output_bboxes_dims = TfLiteTensorNumDims(output_bboxes);
        const auto output_scores_dims = TfLiteTensorNumDims(output_scores);
        const auto output_classes_dims = TfLiteTensorNumDims(output_classes);

        std::set<int32_t> number_detections_shape, output_bboxes_shape, output_scores_shape, output_classes_shape;

        for (auto idx = 0; idx <= number_detections_dims; ++idx) { number_detections_shape.insert(TfLiteTensorDim(number_detections, idx)); }
        for (auto idx = 0; idx <= output_bboxes_dims; ++idx) { output_bboxes_shape.insert(TfLiteTensorDim(output_bboxes, idx)); }
        for (auto idx = 0; idx <= output_scores_dims; ++idx) { output_scores_shape.insert(TfLiteTensorDim(output_scores, idx)); }
        for (auto idx = 0; idx <= output_classes_dims; ++idx) { output_classes_shape.insert(TfLiteTensorDim(output_classes, idx)); }

#ifdef DEBUG
        for (auto idx = 0; idx <= number_detections_dims; ++idx) { PrintShape(number_detections_shape, "number_detections_shape"); }
        for (auto idx = 0; idx <= output_bboxes_dims; ++idx) { PrintShape(output_bboxes_shape, "output_bboxes_shape"); }
        for (auto idx = 0; idx <= output_scores_dims; ++idx) { PrintShape(output_scores_shape, "output_scores_shape"); }
        for (auto idx = 0; idx <= output_classes_dims; ++idx) { PrintShape(output_classes_shape, "output_classes_shape"); }

        std::cout.flush();
#endif

        DetectionResults<DetectionResult> results;
        return results;
    }
}  // namespace emb
