#include "libSubpx.hpp"

void Subpx::annotate(cv::Mat &image_downsc, const cv::Mat &image_orig, const double &annotationScale, const double &outputDownscale, std::ostringstream &OSD_text)
{
  if (showContours_on)
  {
    // show all countours in dark gray
    /*for (size_t i = 0; i < contours.size(); ++i)
    {
      for (size_t j = 0; j < contours[i].size(); ++j)
      {
        cv::Point point(contours[i][j].x / outputDownscale, contours[i][j].y / outputDownscale);
        cv::circle(image_downsc, point, annotationScale / outputDownscale, cv::Scalar(80, 80, 80), -1); // Draw each point as a small circle
      }
    }*/
    for (int i = 0; i < selectedContours.size(); i++)
    {
      for (size_t j = 0; j < selectedContours[i].size(); ++j)
      {
        cv::Point point(selectedContours[i][j].x / outputDownscale, selectedContours[i][j].y / outputDownscale);
        cv::circle(image_downsc, point, annotationScale / outputDownscale, COLORS[i % COLORS.size()], -1); // Draw each point as a small circle
      }
    }

    OSD_text << "Selected contours: " << selectedContours.size() << " out of " << internalIndices.size() + externalIndices.size() << " are identified" << std::endl;

    OSD_text << "Filtered points: " << filteredContCenter_subpx.size() << std::endl;
  }
  if (blobXY_on)
  {
    for (auto it : filteredContCenter) // draw crosses for extracted points from cv:simpleBlobDetector
    {
      cv::putText(image_downsc, "(x|y) = (" + std::to_string(it.x) + " | " + std::to_string(it.y) + ")", cv::Point((it.x) / outputDownscale + image_downsc.cols * 0.01, (it.y) / outputDownscale - image_downsc.cols * 0.01), cv::FONT_HERSHEY_PLAIN, annotationScale / outputDownscale, cv::Scalar(0, 0, 255), annotationScale / outputDownscale + 1.0, false);
    }
  }
  if (centerPoint_on)
  {
    for (auto it : filteredContCenter_subpx) // draw circles for filtered ccc-points
    {
      cv::circle(image_downsc, cv::Point(it.x / outputDownscale, it.y / outputDownscale), it.z / 2.0f / outputDownscale, cv::Scalar(0, 0, 255), -1);
    }
  }
  if (blobID_on)
  {
    int id = 0;
    for (auto it : filteredContCenter) // draw crosses for extracted points from cv:simpleBlobDetector
    {
      cv::putText(image_downsc, std::to_string(id), cv::Point((it.x) / outputDownscale - image_downsc.cols * 0.01, (it.y) / outputDownscale - image_downsc.cols * 0.01), cv::FONT_HERSHEY_PLAIN, annotationScale / outputDownscale, cv::Scalar(0, 0, 255), annotationScale / outputDownscale + 1.0, false);
      id++;
    }
  }
 
  if (detailView_on)
  {
    double viewerSize = 0.25;
    // get the size of the mini view be 1/4 of image_downsc size
    cv::Size viewer(image_downsc.cols * viewerSize, image_downsc.rows * viewerSize);
    // reserve viewer image
    cv::Mat viewer_image(viewer.height, viewer.width, CV_8UC3, cv::Scalar(0, 0, 0));
    // check if selected display ID is within range
    if (0 <= detailView_pointID && detailView_pointID < filteredCont.size())
    {
      // get the data corresponding to selected display ID
      std::vector<cv::Point> &fC = filteredCont[detailView_pointID];
      cv::Point2d &fCCenter = filteredContCenter[detailView_pointID];
      std::vector<std::shared_ptr<cv::Point2d>> &fC_sub = filteredCont_subpx[detailView_pointID];
      cv::Point3d &fCCenter_sub = filteredContCenter_subpx[detailView_pointID];

      int extractWidth = static_cast<double>(image_orig.cols * viewerSize) / static_cast<double>(detailView_scale) + 0.5; // add half pixel for modulo division rest
      int extractHeight = static_cast<double>(image_orig.rows * viewerSize) / static_cast<double>(detailView_scale) + 0.5;
      cv::Rect ROI(fCCenter_sub.x - extractWidth / 2, fCCenter_sub.y - extractHeight / 2, extractWidth, extractHeight);
      ROI.x = std::max(ROI.x, 0);
      ROI.y = std::max(ROI.y, 0);
      ROI.width = std::min(ROI.width, image_orig.cols - ROI.x);
      ROI.height = std::min(ROI.height, image_orig.rows - ROI.y);
      cv::Mat ROI_image = image_orig(ROI);
      cv::Mat ROI_upscaled;
      cv::resize(ROI_image, ROI_upscaled, cv::Size(), detailView_scale / outputDownscale, detailView_scale / outputDownscale, cv::INTER_NEAREST);
      cv::Mat ROI_up_col;

      if (ROI_upscaled.type() == CV_8UC1)
      {
        cv::cvtColor(ROI_upscaled, ROI_up_col, cv::COLOR_GRAY2BGR); // Converts single-channel grayscale to 3-channel BGR
      }
      else if (ROI_upscaled.type() == CV_8UC3)
      {
        ROI_up_col = ROI_upscaled; // If already 3-channel, just convert the pixel depth if needed
      }
      else
      {
        CV_Assert(false); // There's a problem ... the image format is neither CV_8UC1 nor CV_8UC3
      }

      // draw pixel information
      std::vector<cv::Point> displayContour2;
      for (const auto &p : fC)
      {
        int x = floor((p.x - ROI.x + 0.5) * detailView_scale / outputDownscale);
        int y = floor((p.y - ROI.y + 0.5) * detailView_scale / outputDownscale);
        displayContour2.emplace_back(x, y);
        cv::circle(ROI_up_col, cv::Point(x, y), annotationScale / outputDownscale * 3.0, cv::Scalar(0, 255, 0), -1);
      }
      std::vector<std::vector<cv::Point>> displayContourFull2{displayContour2};
      cv::drawMarker(ROI_up_col, cv::Point((fCCenter.x - ROI.x) * detailView_scale / outputDownscale, (fCCenter.y - ROI.y) * detailView_scale / outputDownscale), cv::Scalar(0, 255, 0), cv::MARKER_TILTED_CROSS, annotationScale / outputDownscale * 5.0, annotationScale / outputDownscale * 2.0);
      cv::drawContours(ROI_up_col, displayContourFull2, 0, cv::Scalar(0, 255, 0), annotationScale / outputDownscale * 1.0);

      // draw subpixel information
      std::vector<cv::Point> displayContour;
      for (const auto &p : fC_sub)
      {
        int x = floor((p->x - ROI.x + 0.5) * detailView_scale / outputDownscale);
        int y = floor((p->y - ROI.y + 0.5) * detailView_scale / outputDownscale);
        cv::circle(ROI_up_col, cv::Point(x, y), annotationScale / outputDownscale * 3.0, cv::Scalar(0, 0, 255), -1);
        displayContour.emplace_back(x, y);
      }
      cv::circle(ROI_up_col, cv::Point((fCCenter_sub.x - ROI.x) * detailView_scale / outputDownscale, (fCCenter_sub.y - ROI.y) * detailView_scale / outputDownscale), annotationScale / outputDownscale * 3.0, cv::Scalar(0, 0, 255), -1);

      std::vector<std::vector<cv::Point>> displayContourFull{displayContour};
      cv::drawContours(ROI_up_col, displayContourFull, 0, cv::Scalar(0, 0, 255), annotationScale / outputDownscale * 1.0);
      cv::putText(ROI_up_col, "ID =  " + std::to_string(detailView_pointID), cv::Point2d(20 / outputDownscale, 200 / outputDownscale), cv::FONT_HERSHEY_PLAIN, annotationScale / outputDownscale, cv::Scalar(0, 0, 255), annotationScale / outputDownscale + 1.0, false);
      double delta = std::sqrt((fCCenter_sub.x - fCCenter.x) * (fCCenter_sub.x - fCCenter.x) + (fCCenter_sub.y - fCCenter.y) * (fCCenter_sub.y - fCCenter.y));
      cv::putText(ROI_up_col, "centerDelta =  " + std::to_string(delta) + "px", cv::Point2d(20 / outputDownscale, 100 / outputDownscale), cv::FONT_HERSHEY_PLAIN, annotationScale / outputDownscale, cv::Scalar(0, 0, 255), annotationScale / outputDownscale + 1.0, false);

      // put everything to viewer_image
      // Calculate the top-left corner to place the small image in the center of the large image
      int x_offset = std::max(0, (viewer_image.cols - ROI_up_col.cols) / 2);
      int y_offset = std::max(0, (viewer_image.rows - ROI_up_col.rows) / 2);
      // Calculate how much of the small image fits into the large image
      int width = std::min(ROI_up_col.cols, viewer_image.cols - x_offset);
      int height = std::min(ROI_up_col.rows, viewer_image.rows - y_offset);
      // Define the ROI in both images
      cv::Rect roiLargeImage(x_offset, y_offset, width, height);
      int x_small = std::max(0, (ROI_up_col.cols - width) / 2); // the worst case for the integer division is a shift to a smaller x_small, which definately is inside the image.
      int y_small = std::max(0, (ROI_up_col.rows - height) / 2);
      cv::Rect roiROI_up_col(x_small, y_small, width, height);
      // Copy the portion of the small image that fits into the large image
      ROI_up_col(roiROI_up_col).copyTo(viewer_image(roiLargeImage));
    }
    else // draw text to indicate selected display ID is out of bounds
    {
      cv::putText(viewer_image, "selected point ID " + std::to_string(detailView_pointID), cv::Point(10, viewer.height / 3.0), cv::FONT_HERSHEY_PLAIN, annotationScale / outputDownscale, cv::Scalar(0, 0, 255), annotationScale / outputDownscale, false);
      cv::putText(viewer_image, "is out of bounds", cv::Point(10, viewer.height / 3.0 * 2.0), cv::FONT_HERSHEY_PLAIN, annotationScale / outputDownscale, cv::Scalar(0, 0, 255), annotationScale / outputDownscale, false);
    }
    // draw an rectangle border
    cv::rectangle(viewer_image, cv::Point(0, 0), cv::Point(viewer_image.cols - 2, viewer_image.rows - 2), cv::Scalar(0, 0, 255), 3);
    // put viewer_image to global image
    cv::Rect roi_image(image_downsc.cols - viewer_image.cols, image_downsc.rows - viewer_image.rows, viewer_image.cols, viewer_image.rows);
    viewer_image.copyTo(image_downsc(roi_image));
  }
  // cv::imwrite("/home/raptor/suhas_new/src/ccc_blob_detector/output/final_detection.png", image_downsc);
}

void Subpx::detect(const cv::Mat &image, std::ostringstream &OSD_text)//std::ostringstream &OSD_text)
{
  // We detect ccc-markers directly on the distorted image. Unidistorting the whole image is computationally very expensive (especially if use high resolution map and cubic interpolation) and neither calib.io or GOM seem to do that. Since lenses for optical metrology will always use low distortion lenses, and the other aforementioned dependencies don't undistort either, we'll match the bias. There will be miniscule differences, but whole image distortion introduces small errors anyway.
  cv::TickMeter timer;
  timer.start();

  START("threshold");

  /// TODO: Use a smaller blur kernel for speed
  //cv::GaussianBlur(image, img_canny, cv::Size(3, 3), 0);

  cv::GaussianBlur(image, img_canny, cv::Size(5, 5), 0); // What could happen without denoising is leading to edge tracing twice the diameter line, but not closing.
  
  cv::Canny(img_canny, img_canny, canny_thresholdMin, canny_thresholdMax, canny_apertureSize, true);  

  START("FilterNonCircularEdges");
  cv::Mat filteredEdges = cv::Mat::zeros(img_canny.size(), CV_8U);
  std::vector<std::vector<cv::Point>> tempContours;
  cv::findContours(img_canny, tempContours, cv::RETR_LIST, cv::CHAIN_APPROX_NONE);     // RETR_LIST all contours are simply stored as a flat list, and no parent-child relationships are inferred.

  for (const auto& contour : tempContours) {
      double perimeter = cv::arcLength(contour, true);
      double area = cv::contourArea(contour);
      if (perimeter == 0 || area < 50) continue;

      // Circularity check (closer to 1 = more circular)
      double circularity = 4 * CV_PI * area / (perimeter * perimeter);
      if (circularity > 0.5) { // Adjust threshold as needed
          cv::drawContours(filteredEdges, std::vector<std::vector<cv::Point>>{contour}, -1, cv::Scalar(255), 1);    // I will use this to make some changes.
      }
  }
  // cv::imwrite("/home/raptor/suhas_new/src/ccc_blob_detector/output/filtered_edges.png", filteredEdges);

  filteredEdges.copyTo(img_canny); // Replace raw edges with filtered edges
  STOP("FilterNonCircularEdges");
  
  filteredEdges.copyTo(img_canny); 
  
  STOP("threshold");
  START("FindContours");

  /// TODO: Use simplified approximation for contours
  //cv::findContours(img_canny, contours, contours_hierarchy, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE);

  cv::findContours(img_canny, contours, contours_hierarchy, cv::RETR_TREE, cv::CHAIN_APPROX_NONE); // outputs are cleared automatically
  

  STOP("FindContours");

  // START("selectedContours");
  externalIndices.clear();
  internalIndices.clear();
  GetEdgeContourValidIndices(contours_hierarchy, internalIndices, externalIndices); // only retrieve outer contours
  selectedContours.clear();
  selectedContours.reserve(internalIndices.size());
  for (auto element : internalIndices)
  {
    selectedContours.push_back(contours[element]);
  }

  // START("filtering");
  
  std::vector<double> contLengths, contAspectRatio; 
  contLengths.resize(selectedContours.size());   
  std::transform(std::execution::par, selectedContours.begin(), selectedContours.end(), contLengths.begin(), [](const std::vector<cv::Point> &contour)
                 { return contour.size(); });
  cv::Mat_<double> contLengthMat(contLengths); 

  // extract properties
  std::vector<cv::Point2d> contCenter;
  std::vector<double> contAxisA, contAxisB, contArea, centerCol;


  for (const auto &c : selectedContours)
  {
    auto M = cv::moments(c);
    double area = M.m00;
    
    // handle the case where the counter is a line without area, see https://stackoverflow.com/questions/62392240/opencv-cv2-moments-returns-all-moments-to-zero
    if (area <= 0)
    {
      contArea.push_back(-1.0);
      contCenter.emplace_back(-1.0, -1.0);
      contAxisA.emplace_back(-1);
      contAxisB.emplace_back(-1);
      contAspectRatio.emplace_back(-1); 
      centerCol.push_back(static_cast<double>(-1));
      continue;
    }
    contArea.push_back(area);
    auto centerX = M.m10 / area;
    auto centerY = M.m01 / area;
    double m20 = M.mu20 / area;
    double m02 = M.mu02 / area;
    double m11 = M.mu11 / area;
    double c1 = m20 - m02;
    double c2 = c1 * c1;
    double c3 = 4 * m11 * m11;

    contCenter.emplace_back(centerX, centerY);

    auto ra = sqrt(2.0 * (m20 + m02 + sqrt(c2 + c3)));
    contAxisA.emplace_back(ra);
    auto rb = sqrt(2.0 * (m20 + m02 - sqrt(c2 + c3)));
    contAxisB.emplace_back(rb);
    contAspectRatio.emplace_back(std::min(ra / rb, rb / ra)); 

    // get center color
    int x = static_cast<int>(centerX);
    int y = static_cast<int>(centerY);

    // Ensure the coordinates are within the image boundaries
    CV_Assert(x >= 0 && y >= 0 && x <= image.cols && y <= image.rows);

    // Return the intensity value at the rounded coordinates
    auto grayValue = image.at<uchar>(y, x);
    
    centerCol.push_back(static_cast<double>(grayValue));
  }

  cv::Mat_<double> contRadiusAMat(contAxisA), contRadiusBMat(contAxisB), contAreaMat(contArea), contCenterColMat(centerCol);
  cv::Mat_<double> contAspectRatioMat(contAspectRatio); 



  cv::Mat thresAspectRatio, thresRadiusAMax, thresRadiusAMin, thresRadiusBMax, thresRadiusBMin, thresAreaMin, thresAreaMax, thresCenterCol;
  cv::threshold(contAspectRatioMat, thresAspectRatio, minCircularity, 0.3, cv::ThresholdTypes::THRESH_BINARY);
  cv::threshold(contRadiusAMat, thresRadiusAMax, maxRadius, 1.0, cv::ThresholdTypes::THRESH_BINARY_INV);
  cv::threshold(contRadiusAMat, thresRadiusAMin, minRadius, 1.0, cv::ThresholdTypes::THRESH_BINARY);
  cv::threshold(contRadiusBMat, thresRadiusBMax, maxRadius, 1.0, cv::ThresholdTypes::THRESH_BINARY_INV);
  cv::threshold(contRadiusBMat, thresRadiusBMin, minRadius, 1.0, cv::ThresholdTypes::THRESH_BINARY);
  cv::threshold(contCenterColMat, thresCenterCol, centerColorThresh, 1.0, cv::ThresholdTypes::THRESH_BINARY);
  cv::threshold(contAreaMat, thresAreaMin, minArea, 1.0, cv::ThresholdTypes::THRESH_BINARY);
  cv::threshold(contAreaMat, thresAreaMax, maxArea, 1.0, cv::ThresholdTypes::THRESH_BINARY_INV);
  cv::Mat and0, and1, and2, and3, and4, and5, and6, and7;
  cv::bitwise_and(thresAspectRatio, thresRadiusAMax, and0);
  cv::bitwise_and(and0, thresRadiusAMax, and1);
  cv::bitwise_and(and1, thresRadiusAMin, and2);
  cv::bitwise_and(and2, thresRadiusBMax, and3);
  cv::bitwise_and(and3, thresRadiusBMin, and4);
  cv::bitwise_and(and4, thresAreaMin, and5);
  cv::bitwise_and(and5, thresAreaMax, and6);
  cv::bitwise_and(and6, thresCenterCol, and7);
  cv::Mat filteredIdx;
  cv::findNonZero(and7, filteredIdx);
  /// TODO: Add cv::isContourConvex() filter?

  filteredCont.clear();
  filteredContCenter.clear();
  for (int i = 0; i < filteredIdx.rows; i++)
  {
    int index = filteredIdx.at<cv::Point>(i).y;
    filteredCont.emplace_back(selectedContours[index]);
    std::vector<std::shared_ptr<cv::Point2d>> contSubPix; 
    filteredContCenter.emplace_back(contCenter[index]);
  }

  cv::Mat contourImage = cv::Mat::zeros(image.size(), CV_8UC3);

// Draw all filtered contours
  for (size_t i = 0; i < filteredCont.size(); i++) {
      cv::drawContours(contourImage, filteredCont, i, cv::Scalar(0, 255, 0), 2); // Draw in green
  }

  // START("supixelDetection");
  filteredCont_subpx.clear();
  filteredContCenter_subpx.clear();
  for (auto single_contour : filteredCont)
  {
    std::vector<std::shared_ptr<cv::Point2d>> contSubPix;

    auto roi = cv::boundingRect(single_contour);
    const int extension = 3; // extend 3 px for cv::sepFilter2D()
    cv::Rect extendedROI(roi.x - extension, roi.y - extension, roi.width + 2 * extension, roi.height + 2 * extension);
    // Ensure the extended ROI is within the image boundaries
    extendedROI.x = std::max(0, extendedROI.x);
    extendedROI.y = std::max(0, extendedROI.y);
    extendedROI.width = std::min(extendedROI.width, image.cols - extendedROI.x);
    extendedROI.height = std::min(extendedROI.height, image.rows - extendedROI.y);

    cv::Mat image_roi = image(extendedROI);

    for (cv::Point &point : single_contour)
    {
      // Subtract ROI top-left corner coordinates
      point.x -= extendedROI.x;
      point.y -= extendedROI.y;
    }
    
    // Create a copy of the original image to draw centers on
    cv::Mat centerImage = image.clone();

    // Draw centers of filtered contours
    for (const auto& center : filteredContCenter) {
        cv::circle(centerImage, cv::Point(center.x, center.y), 5, cv::Scalar(0, 0, 255), -1); // Draw a red circle at each center
    }

    SubPixelEdgeContour(image_roi, single_contour, contSubPix); /// Only send ROI, because gradients and 2nd order gradients take a long time to calculate

    for (auto el : contSubPix)
    {
      el->x += extendedROI.x;
      el->y += extendedROI.y;
    }
    filteredCont_subpx.push_back(contSubPix);

    // get subpixel centroid position. Careful, small numbers addition could introduce errors at some point. Maybe use Kahan Summation? Be careful of compiler optimizations!
    size_t totalPoints = contSubPix.size(); // already calculated before ...
    double sumX = std::accumulate(contSubPix.begin(), contSubPix.end(), 0.0, [](double sum, const std::shared_ptr<cv::Point2d> &point)
                                  { return sum + point->x; });
    double sumY = std::accumulate(contSubPix.begin(), contSubPix.end(), 0.0, [](double sum, const std::shared_ptr<cv::Point2d> &point)
                                  { return sum + point->y; });

    double centerX = sumX / totalPoints;
    double centerY = sumY / totalPoints;
    double size = 0.5 * (roi.width + roi.height);
    filteredContCenter_subpx.emplace_back(centerX, centerY, size);
  }
  timer.stop();
  OSD_text << "Processing Time of the detect function: " <<timer.getTimeMilli() << " ms" <<std::endl;
}


void Subpx::GetEdgeContourValidIndices(const std::vector<cv::Vec4i>& hierarchy, 
                                      std::vector<int>& internalIndices, 
                                      std::vector<int>& externalIndices) {
    internalIndices.clear();
    externalIndices.clear();
    const int PARENT = 3, FIRST_CHILD = 2;

    for (int i = 0; i < hierarchy.size(); ++i) {
        // Check if contour is outer (no parent)
        if (hierarchy[i][PARENT] == -1) {
            externalIndices.push_back(i);
        }
        // Check if contour is inner (no children)
        if (hierarchy[i][FIRST_CHILD] == -1) {
            internalIndices.push_back(i);
        }
    }
}

void Subpx::SubPixelEdgeContour(const cv::Mat &image_in, const std::vector<cv::Point> &filteredCont, std::vector<std::shared_ptr<cv::Point2d>> &contSubPix)
{
  // 7-tap interpolant and 1st and 2nd derivative coefficients according to
  // H. Farid and E. Simoncelli, "Differentiation of Discrete Multi-Dimensional Signals"
  // IEEE Trans. Image Processing. 13(4): pp. 496-508 (2004)
  // computeGradients(image_in);  
  // contSubPix.resize(filteredCont.size());

  // #pragma omp parallel for
  // for (size_t i = 0; i < filteredCont.size(); ++i) {
  //     contSubPix[i] = this->SubPixelFacet(filteredCont[i], gy, gx, gyy, gxx, gxy);
  const std::vector<double> p_vec{0.004711, 0.069321, 0.245410, 0.361117, 0.245410, 0.069321, 0.004711};
  const std::vector<double> d1_vec{-0.018708, -0.125376, -0.193091, 0.000000, 0.193091, 0.125376, 0.018708};
  const std::vector<double> d2_vec{0.055336, 0.137778, -0.056554, -0.273118, -0.056554, 0.137778, 0.055336};

  auto p = cv::Mat_<double>(p_vec);
  auto d1 = cv::Mat_<double>(d1_vec);
  auto d2 = cv::Mat_<double>(d2_vec);

  cv::Mat image_gray;
  image_in.convertTo(image_gray, CV_8UC1); // make sure the conversion is a 8 bit single channel image for the gradient methods to work correctly

  if (image_in.channels() > 1) // Convert to grayscale if it's not already a single-channel image
  {
    cv::cvtColor(image_in, image_gray, cv::COLOR_BGR2GRAY);
  }
  else // If it's already a single-channel image, use it directly
  {
    image_gray = image_in;
  }

  cv::Mat dx, dy, grad;
  cv::sepFilter2D(image_gray, dy, CV_64F, p, d1);
  cv::sepFilter2D(image_gray, dx, CV_64F, d1, p);
  cv::pow(dy.mul(dy, 1.0) + dx.mul(dx, 1.0), 0.5, grad); // element wise sqrt(dx²+dy²) -> grad

  cv::Mat gy, gx, gyy, gxx, gxy;
  cv::sepFilter2D(grad, gy, CV_64F, p, d1);
  cv::sepFilter2D(grad, gx, CV_64F, d1, p);
  cv::sepFilter2D(grad, gyy, CV_64F, p, d2);
  cv::sepFilter2D(grad, gxx, CV_64F, d2, p);
  cv::sepFilter2D(grad, gxy, CV_64F, d1, d1);




  
  contSubPix.resize(filteredCont.size());
  std::transform(std::execution::seq,
                 filteredCont.cbegin(),
                 filteredCont.cend(),
                 contSubPix.begin(),
                 [this, &gy, &gx, &gyy, &gxx, &gxy](const cv::Point &p)
                 { return this->SubPixelFacet(p, gy, gx, gyy, gxx, gxy); });



// #pragma omp parallel for
//   for (size_t i = 0; i < filteredCont.size(); ++i) {
//     contSubPix[i] = this->SubPixelFacet(filteredCont[i], gy, gx, gyy, gxx, gxy);               




//   }


}

std::shared_ptr<cv::Point2d> Subpx::SubPixelFacet(const cv::Point &p, cv::Mat &gyMat, cv::Mat &gxMat, cv::Mat &gyyMat, cv::Mat &gxxMat, cv::Mat &gxyMat)
{

  // Subpixel edge extraction method according to
  // C. Steger, "An unbiased detector of curvilinear structures",
  // IEEE Transactions on Pattern Analysis and Machine Intelligence,
  // 20(2): pp. 113-125, (1998)

  auto row = p.y;
  auto col = p.x;
  auto gy = gyMat.at<double>(row, col);
  auto gx = gxMat.at<double>(row, col);
  auto gyy = gyyMat.at<double>(row, col);
  auto gxx = gxxMat.at<double>(row, col);
  auto gxy = gxyMat.at<double>(row, col);

  Eigen::Matrix<double, 2, 2> hessian;
  hessian << gyy, gxy, gxy, gxx;
  Eigen::JacobiSVD<Eigen::MatrixXd> svd(hessian, Eigen::ComputeFullV);
  auto v = svd.matrixV();
  // first column vector of v, corresponding to largest eigen value
  // is the direction perpendicular to the line
  auto ny = v(0, 0);
  auto nx = v(1, 0);
  auto t = -(gx * nx + gy * ny) / (gxx * nx * nx + 2 * gxy * nx * ny + gyy * ny * ny);
  auto px = t * nx;
  auto py = t * ny;

  return std::make_shared<cv::Point2d>(col + px, row + py);
}