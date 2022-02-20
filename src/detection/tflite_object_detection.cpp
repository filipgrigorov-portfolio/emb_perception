#include <cassert>
#include <fstream>
#include <set>
#include <stdexcept>

#include <iostream>

#include <opencv2/imgcodecs.hpp>

#include "detection/tflite_object_detection.hpp"

#define DEBUG

// Note: Works for release and debug
#define ASSERT(obj, MSG) \
if (!obj) throw std::runtime_error(MSG); \

namespace {
    const std::vector<std::string> labels = {
        "person",
        "bicycle",
        "car",
        "motorcycle",
        "airplane",
        "bus",
        "train",
        "truck",
        "boat",
        "traffic light",
        "fire hydrant",
        "street sign",
        "stop sign",
        "parking meter"
    };

    void PrintShape(const std::set<int32_t>& shape, const std::string& name) {
        std::string print = name + ": [";
        for (const auto& dim : shape) {
            print += dim + " ";
        }
        print += " ]";
        std::cout << print << std::endl;
    }
}  //

namespace emb {
    TfLiteObjectDetector::TfLiteObjectDetector(const std::string& model_path, int img_width, int img_height)
     : img_dims_(img_width, img_height) {
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
            TfLiteObjectDetector::TfLiteModelDeleter());

        ASSERT(tf_model_, "TfLite model is not correct")

        tf_opts_ = std::shared_ptr<TfLiteInterpreterOptions>(
            TfLiteInterpreterOptionsCreate(), 
            TfLiteObjectDetector::TfLiteInterpreterOptionsDeleter());
        TfLiteInterpreterOptionsSetNumThreads(tf_opts_.get(), 2);

        ASSERT(tf_opts_, "TfLite opts are not correct")

        tf_interpreter_  = std::shared_ptr<TfLiteInterpreter>(
            TfLiteInterpreterCreate(tf_model_.get(), tf_opts_.get()), 
            TfLiteObjectDetector::TfLiteInterpreterDeleter());

        ASSERT(tf_interpreter_, "TfLite interpreter is not correct")

        TfLiteInterpreterAllocateTensors(tf_interpreter_.get());

        input_tensor_ = std::shared_ptr<TfLiteTensor>(
            TfLiteInterpreterGetInputTensor(tf_interpreter_.get(), 0), 
            TfLiteObjectDetector::TfLiteTensorDeleter());

        // Note: Verify dimensions of the input tensor
        assert(input_tensor_->dims->data[0] == 1);
        assert(input_tensor_->dims->data[1] == img_width);
        assert(input_tensor_->dims->data[2] == img_height);
        assert(input_tensor_->dims->data[3] == 3);
    }

    TfLiteObjectDetector::~TfLiteObjectDetector() {
        if (data_) {
            free(data_);
        }
    }

    void TfLiteObjectDetector::Draw(const cv::Mat& input, const emb::DetectionResults& results) const {
        cv::Mat canvas = input.clone();
        for (const auto& result : results) {
            cv::Rect_<int> bbox{cv::Point_<int>(result.xmin, result.ymin), cv::Point_<int>(result.xmax, result.ymax)};
            cv::rectangle(canvas, bbox, cv::Scalar(0, 0, 255), 1);
            
            cv::imwrite("annotated.jpg", canvas);
        }
    }

    emb::DetectionResults TfLiteObjectDetector::Infer(const cv::Mat& input, bool quantized) {
        assert(input.data);

        auto img2size = (quantized) ? AllocateQuantizedCvMat(PreprocessCvMat(input)) : AllocateCvMat(PreprocessCvMat(input));
        TfLiteTensorCopyFromBuffer(input_tensor_.get(), img2size.first, img2size.second);

        if (TfLiteInterpreterInvoke(tf_interpreter_.get()) != kTfLiteOk) {
            throw std::runtime_error("Invoke did not work");
        }

        // tf.int tensor with only one value, the number of detections [N]
        auto num_detections = TfLiteInterpreterGetOutputTensor(tf_interpreter_.get(), 3);
        const int num_detections_f = static_cast<int>(*(num_detections->data.f));

        std::cout << "\nNumber outputs: " << TfLiteInterpreterGetOutputTensorCount(tf_interpreter_.get()) << std::endl;
        std::cout << "Number dets: " << num_detections_f << std::endl;

        // Note: const *, does not need to be free'ed

        // tf.float32 tensor of shape [N, 4] containing bounding box coordinates in the following order: [ymin, xmin, ymax, xmax]
        auto detection_boxes = TfLiteInterpreterGetOutputTensor(tf_interpreter_.get(), 0);
        if (!detection_boxes) {
            throw std::runtime_error("Badly caught detection_boxes tensor");
        }

        // tf.int tensor of shape [N] containing detection class index from the label file
        auto detection_classes = TfLiteInterpreterGetOutputTensor(tf_interpreter_.get(), 1);

        // tf.float32 tensor of shape [N] containing detection score
        auto detection_scores = TfLiteInterpreterGetOutputTensor(tf_interpreter_.get(), 2);

#ifdef DEBUG
        const auto num_detections_dims = TfLiteTensorNumDims(num_detections);
        const auto detection_boxes_dims = TfLiteTensorNumDims(detection_boxes);
        const auto detection_classes_dims = TfLiteTensorNumDims(detection_classes);
        const auto detection_scores_dims = TfLiteTensorNumDims(detection_scores);

        std::cout << "\"" << num_detections->name << "\"" << " : " << num_detections_dims << std::endl;
        std::cout << "\"" << detection_boxes->name << "\"" << " : " << detection_boxes_dims << std::endl;
        std::cout << "\"" << detection_classes->name << "\"" << " : " << detection_classes_dims << std::endl;
        std::cout << "\"" << detection_scores->name << "\"" << " : " << detection_scores_dims << std::endl << std::endl;

        std::cout.flush();
#endif

        emb::DetectionResults results;
        const float* detection_boxes_f = detection_boxes->data.f;
        const float* detection_classes_f = detection_classes->data.f;
        const float* detection_scores_f = detection_scores->data.f;

        // Note: [ymin, xmin, ymax, xmax]
        for (auto idx = 0; idx < num_detections_f; ++idx) {
            emb::DetectionResult result;
            const auto index = idx * 4;
            result.ymin = detection_boxes_f[index] * input.rows;
            result.xmin = detection_boxes_f[index + 1] * input.cols;
            result.ymax = detection_boxes_f[index + 2] * input.rows;
            result.xmax = detection_boxes_f[index + 3] * input.cols;
            
            if (detection_classes_f[idx] > labels.size()) {
                std::cerr << "Unsupported label; skipping" << std::endl;
                continue;
            }

            if (detection_scores_f[idx] <= 0.3) {
                std::cerr << "Low confidence= " << detection_scores_f[idx] * 100 << "%; skipping" << std::endl;
                continue;
            }

            result.class_lbl = labels[static_cast<int>(detection_classes_f[idx])];
            result.score = detection_scores_f[idx];

            std::cout << "[" << result.xmin << ", " << result.ymin <<
                ", " << result.xmax - result.xmin << ", " << result.ymax - result.ymin << "] - " <<
                result.score << " - " << result.class_lbl << std::endl;
            results.push_back(result);
        }

        return results;
    }

    //private
    std::pair<void*, size_t> TfLiteObjectDetector::AllocateCvMat(
        const cv::Mat& img, const std::pair<float, float>& range/*={0.f, 255.f}*/) {
        /* 
            Note: Only if the input needs to be in [-1, 1]
                  img' = (img / 255) * 2 - 1 <=> [0, 1] * 2 = [0, 2] - 1 = [-1, 1]
        */
        cv::Mat normalized_img = img.clone();
        auto min_intensity = 0.0; 
        auto max_intensity = 0.0;
        cv::minMaxLoc(normalized_img, &min_intensity, &max_intensity);
        if (range.first != static_cast<float>(min_intensity) && range.second != static_cast<float>(max_intensity)) {
            normalized_img.convertTo(normalized_img, CV_32FC3, (1.f - range.first) / range.second, range.first);
        }
        
        float* data = input_tensor_->data.f;
        size_t size = normalized_img.cols * normalized_img.rows * normalized_img.channels() * sizeof(float);
        std::memcpy(data, (void*)normalized_img.data, size);

        std::stringstream ss;
        for (auto idx = 0; idx < normalized_img.cols * normalized_img.rows * normalized_img.channels() - 1; ++idx) {
            ss << data[idx] << ",";
        }
        ss << data[normalized_img.cols * normalized_img.rows * normalized_img.channels() - 1];
        std::ofstream os("img.npy", std::ios::binary);
        if (os.is_open()) {
            std::cout << "Serializing img into npy" << std::endl;
            os.write(ss.str().c_str(), ss.str().size());
        }
        //std::cout << ss.str();

        return std::make_pair((void*)data, size);
    }

    std::pair<void*, size_t> TfLiteObjectDetector::AllocateQuantizedCvMat(const cv::Mat& img) {
        if (!input_tensor_.get()) {
            throw std::runtime_error("Badly allocated input tensor");
        }

        uint8_t* data = input_tensor_->data.uint8;
        size_t size = img.cols * img.rows * img.channels() * sizeof(uint8_t);
        /*for (auto idx = 0; idx < size; ++idx) {
            std::cout << img.data[idx] << " ";
        }
        std::cout << std::endl;*/
        std::memcpy(data, (void*)img.data, size);

        std::stringstream ss;
        for (auto idx = 0; idx < img.cols * img.rows * img.channels() - 1; ++idx) {
            ss << data[idx] << ",";
        }
        ss << data[img.cols * img.rows * img.channels() - 1];
        std::ofstream os("img.npy", std::ios::binary);
        if (os.is_open()) {
            std::cout << "Serializing img into npy" << std::endl;
            os.write(ss.str().c_str(), ss.str().size());
        }
        //std::cout << ss.str();

        return std::make_pair((void*)data, size);
    }

    cv::Mat TfLiteObjectDetector::PreprocessCvMat(const cv::Mat& img) {
        cv::Mat processed_img;
        /* 
            Note: fx=0 and fy=0 play crucial role for the accuracy of the result
            If (0, 0), the image is scaled by fx and fy along the horizontal and
            vertical axis
            If (0, 0), the scales are :
                (1) dsize.width / image.cols
                (2) dsize.height / image.rows
            dsize = Size(fx * image.cols, fy * image.rows) if no size is specified
            If (0, 0) => we keep the aspect ratio intact
        */
        cv::resize(img, processed_img, img_dims_, 0, 0, cv::INTER_AREA);
        cv::cvtColor(processed_img, processed_img, cv::COLOR_BGR2RGB);
        std::cout << "\nImage resized shape: " << processed_img.cols << " x " 
            << processed_img.rows << " x " << processed_img.channels() << std::endl;
        return processed_img;
    }

}  // namespace emb
