#include <cassert>
#include <set>
#include <stdexcept>

#include "detection/detection.hpp"

// Note: Works for release and debug
#define ASSERT(obj, MSG) \
if (!obj) throw std::runtime_error(MSG); \

namespace emb {
    ObjectDetector::ObjectDetector(const std::string& model_path) {
        if (model_path.empty()) throw std::runtime_error("Model path is empty");

        tf_model_ = std::shared_ptr<TfLiteModel>(
            TfLiteModelCreateFromFile(model_path.c_str()), 
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

    ObjectDetector::~ObjectDetector() {}

    DetectionResults<ObjectDetector::DetectionResult> ObjectDetector::Infer(const cv::Mat& input) {
        assert(input.data);

        TfLiteTensorCopyFromBuffer(input_tensor_.get(), input.data, input.rows * input.step1() * sizeof(input.type()));
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

        DetectionResults<DetectionResult> results;
        return results;
    }
}  // namespace emb
