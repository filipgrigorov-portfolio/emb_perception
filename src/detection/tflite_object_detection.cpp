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

    void TfLiteObjectDetector::Draw(const cv::Mat& input, const emd::DetectionResults<TfLiteObjectDetector::DetectionResult>& results) const {
        cv::Mat canvas = input.clone();
        for (const auto& result : results) {
            cv::Rect bbox{cv::Point2i(result.xmin, result.ymin), cv::Point2i(result.xmax, result.ymax)};
            cv::rectangle(canvas, bbox, cv::Scalar(0, 0, 255), 1);
            
            cv::imwrite("annotated.jpeg", canvas);
        }
    }

    emd::DetectionResults<TfLiteObjectDetector::DetectionResult> TfLiteObjectDetector::Infer(const cv::Mat& input, bool quantized) {
        assert(input.data);

        auto img2size = (quantized) ? ProcessQuantizedCvMat(input, img_dims_) : ProcessQuantizedCvMat(input, img_dims_);
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

        emd::DetectionResults<TfLiteObjectDetector::DetectionResult> results;
        const float* detection_boxes_f = detection_boxes->data.f;
        const float* detection_classes_f = detection_classes->data.f;
        const float* detection_scores_f = detection_scores->data.f;

        for (auto idx = 0; idx < num_detections_f; ++idx) {
            TfLiteObjectDetector::DetectionResult result;
            const auto index = idx * 4;
            result.xmin = detection_boxes_f[index] * input.cols;
            result.ymin = detection_boxes_f[index + 1] * input.rows;
            result.xmax = detection_boxes_f[index + 2] * input.cols;
            result.ymax = detection_boxes_f[index + 3] * input.rows;
            std::cout << detection_classes_f[idx] << std::endl;
            if (detection_classes_f[idx] > labels.size() || detection_scores_f[idx] <= 0.4) {
                std::cerr << "Unsupported label; skipping" << std::endl;
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
    std::pair<float*, size_t> TfLiteObjectDetector::ProcessCvMat(const cv::Mat& input, const cv::Size& img_res) {
        cv::Mat input_img;
        cv::resize(input, input_img, img_res, cv::INTER_AREA);
        std::cout << "\nImage resized shape: " << input_img.cols << " x " << input_img.rows << " x " << input_img.channels() << std::endl;

        // Note: img' = (img / 255) * 2 - 1 <=> [0, 1] * 2 = [0, 2] - 1 = [-1, 1]
        // Note: Only if the input needs to be in [-1, 1]
        //input_img.convertTo(input_img, CV_32FC3, 2.f / 255.f, -1);
        float* data = input_tensor_->data.f;
        size_t size = input_img.cols * input_img.rows * input_img.channels() * sizeof(float);
        std::memcpy(data, (void*)input_img.data, size);

        std::stringstream ss;
        for (auto idx = 0; idx < input_img.cols * input_img.rows * input_img.channels() - 1; ++idx) {
            ss << data[idx] << ",";
        }
        ss << data[input_img.cols * input_img.rows * input_img.channels() - 1];
        std::ofstream os("img.npy", std::ios::binary);
        if (os.is_open()) {
            std::cout << "Serializing img into npy" << std::endl;
            os.write(ss.str().c_str(), ss.str().size());
        }
        std::cout << ss.str();

        return std::make_pair(data, size);
    }

    std::pair<uint8_t*, size_t> TfLiteObjectDetector::ProcessQuantizedCvMat(const cv::Mat& input, const cv::Size& img_res) {
        cv::Mat input_img;
        cv::resize(input, input_img, img_res, cv::INTER_AREA);
        std::cout << "\nImage resized shape: " << input_img.cols << " x " << input_img.rows << " x " << input_img.channels() << std::endl;

        if (!input_tensor_.get()) {
            throw std::runtime_error("Badly allocated input tensor");
        }

        uint8_t* data = input_tensor_->data.uint8;
        size_t size = input_img.cols * input_img.rows * input_img.channels() * sizeof(uint8_t);
        /*for (auto idx = 0; idx < size; ++idx) {
            std::cout << input_img.data[idx] << " ";
        }
        std::cout << std::endl;*/
        void* copiedbytes = std::memcpy(data, (void*)input_img.data, size);

        std::stringstream ss;
        for (auto idx = 0; idx < input_img.cols * input_img.rows * input_img.channels() - 1; ++idx) {
            ss << data[idx] << ",";
        }
        ss << data[input_img.cols * input_img.rows * input_img.channels() - 1];
        std::ofstream os("img.npy", std::ios::binary);
        if (os.is_open()) {
            std::cout << "Serializing img into npy" << std::endl;
            os.write(ss.str().c_str(), ss.str().size());
        }
        //std::cout << ss.str();

        return std::make_pair(data, size);
    }

}  // namespace emb
