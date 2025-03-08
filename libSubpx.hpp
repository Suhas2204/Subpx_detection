
/* Original git: https://github.com/raymondngiam/subpixel-edge-contour-in-opencv.git

The subpixel edge extraction implementation in this repo is based on the following two papers:

- C. Steger, "An unbiased detector of curvilinear structures", IEEE Transactions on Pattern Analysis and Machine Intelligence, 20(2): pp. 113-125, (1998)
- H. Farid and E. Simoncelli, "Differentiation of Discrete Multi-Dimensional Signals" IEEE Trans. Image Processing. 13(4): pp. 496-508 (2004)

MIT License

Copyright (c) 2022 Raymond Ngiam

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.*/

//  This code is based on ros-perception/image_pipeline
//  and http://wiki.ros.org/message_filters#Policy-Based_Synchronizer_.5BROS_1.1.2B-.5D
//  https://github.com/jsk-ros-pkg/jsk_common/blob/master/jsk_ros_patch/image_view2/msg/PointArrayStamped.msg
//  catkin build -DCMAKE_BUILD_TYPE=Debug

// ccc detection should be performed after undistortion, but without rectification. This ensures circles are real circles/ellipses (lines are projected straight). Rectification would alter round circles.

#ifndef LIBSUBPX_H
#define LIBSUBPX_H
#include <opencv2/opencv.hpp>
#include <iostream>
#include <fmt/core.h>
#include <vector>
#include <execution>
#include <Eigen/Dense>
#include "definitions.hpp"

class Subpx
{
private:
  const std::vector<cv::Scalar_<double>> COLORS // Thanks ChatGPT for rainbow color palette
      {
          cv::Scalar_<double>(148, 0, 211), // Violet 1
          cv::Scalar_<double>(0, 255, 255), // Cyan 4
          cv::Scalar_<double>(255, 127, 0), // Orange 7
          cv::Scalar_<double>(0, 0, 255),   // Blue 3
          cv::Scalar_<double>(255, 255, 0), // Yellow 6
          cv::Scalar_<double>(75, 0, 130),  // Indigo 2
          cv::Scalar_<double>(0, 255, 0),   // Green 5
          cv::Scalar_<double>(255, 0, 0)    // Red 8
      };

  cv::Mat img_canny;
  cv::Mat img_adaptive_threshold;
  std::vector<std::vector<cv::Point>> contours;
  std::vector<cv::Vec4i> contours_hierarchy;
  std::vector<std::vector<cv::Point>> selectedContours;
  std::vector<int> externalIndices;
  std::vector<int> internalIndices;
  std::vector<std::vector<cv::Point>> filteredCont;
  std::vector<cv::Point2d> filteredContCenter;
  std::vector<std::vector<std::shared_ptr<cv::Point2d>>> filteredCont_subpx;

public:
  bool blobXY_on, centerPoint_on, blobID_on, showContours_on, detailView_on;

  unsigned int detailView_pointID, detailView_scale;

  int canny_thresholdMin, canny_thresholdMax, canny_apertureSize;
  //int  canny_apertureSize,adaptive_blockSize;
  double maxRadius, minRadius, minCircularity, minArea, maxArea; // adaptive_C;
  int centerColorThresh;

  std::vector<cv::Point3d> filteredContCenter_subpx;

  bool useAdaptiveThreshold;         // Toggle for enabling and disabling the adaptive thresholding.
  int adaptiveBlockSize;             // Block Size for adaptive thresholding.
  double adaptiveC;                  // C value for adaptive thresholding.

  void annotate(cv::Mat &image_downsc, const cv::Mat &image_orig, const double &annotationScale, const double &outputDownscale, std::ostringstream &OSD_text);

  void detect(const cv::Mat &image,std::ostringstream &OSD_text); // (std::ostringstream &OSD_text);

  void GetEdgeContourValidIndices(const std::vector<cv::Vec4i> &hierarchy, std::vector<int> &internalIndices, std::vector<int> &externalIndices);

  void SubPixelEdgeContour(const cv::Mat &image_in, const std::vector<cv::Point> &filteredCont, std::vector<std::shared_ptr<cv::Point2d>> &contSubPix);

  // Subpixel edge extraction method according to
  // C. Steger, "An unbiased detector of curvilinear structures",
  // IEEE Transactions on Pattern Analysis and Machine Intelligence,
  // 20(2): pp. 113-125, (1998)
  std::shared_ptr<cv::Point2d> SubPixelFacet(const cv::Point &p, cv::Mat &gyMat, cv::Mat &gxMat, cv::Mat &gyyMat, cv::Mat &gxxMat, cv::Mat &gxyMat);

  void applyAdaptiveThreshold(const cv::Mat &inputImage);
};

#endif