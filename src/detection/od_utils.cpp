#include <iostream>

#include "detection/od_utils.hpp"

namespace emb {
    void Draw(const cv::Mat& input, const emb::DetectionResults& results) {
        cv::Mat canvas = input.clone();
        for (const auto& result : results) {
            cv::Rect_<int> bbox{cv::Point_<int>(result.xmin, result.ymin), cv::Point_<int>(result.xmax, result.ymax)};
            cv::rectangle(canvas, bbox, cv::Scalar(0, 0, 255), 1);
            
            cv::imwrite("annotated.jpg", canvas);
        }
    }

    cv::Mat PreprocessCvMat(const cv::Mat& img, const cv::Size& new_img_dims) {
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
        cv::resize(img, processed_img, new_img_dims, 0, 0, cv::INTER_AREA);
        cv::cvtColor(processed_img, processed_img, cv::COLOR_BGR2RGB);
        std::cout << "\nImage resized shape: " << processed_img.cols << " x " 
            << processed_img.rows << " x " << processed_img.channels() << std::endl;
        return processed_img;
    }
}  // namespace emb
