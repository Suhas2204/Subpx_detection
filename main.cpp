#include <opencv2/opencv.hpp>
#include "libSubpx.hpp"
#include <iostream>

int main() {
    // Load test image
    cv::Mat image = cv::imread("test_image.png", cv::IMREAD_GRAYSCALE);
    if (image.empty()) {
        std::cerr << "Error: Could not load image!" << std::endl;
        return -1;
    }

    // Initialize Subpx detector with parameters
    Subpx detector;
    detector.canny_thresholdMin = 50;    // Example parameters (adjust as needed)
    detector.canny_thresholdMax = 150;
    detector.minRadius = 5;
    detector.maxRadius = 50;
    detector.minCircularity = 0.7;
    detector.useAdaptiveThreshold = false;
    // ... Set other parameters ...

    // Process the image
    std::ostringstream OSD_text;
    detector.detect(image, OSD_text);

    // Visualize results
    cv::Mat annotated_image;
    cv::cvtColor(image, annotated_image, cv::COLOR_GRAY2BGR);
    detector.annotate(annotated_image, image, 1.0 /*scale*/, 1.0 /*downscale*/, OSD_text);

    // Display output
    cv::imshow("Detected Features", annotated_image);
    cv::waitKey(0);

    return 0;
}
