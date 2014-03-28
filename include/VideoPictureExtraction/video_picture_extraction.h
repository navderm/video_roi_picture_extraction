#ifndef VIDEO_PICTURE_EXTRACTION_H
#define VIDEO_PICTURE_EXTRACTION_H

#include <iostream>
#include <string>
#include <fstream>


#include <opencv2/opencv.hpp>

#include <boost/lexical_cast.hpp>

class VideoPictureExtraction
{
public:
  VideoPictureExtraction();
  virtual ~VideoPictureExtraction();
  void LoadAndExtract();
  void LoadVideoFileAndExtract(std::string);
  void LoadAnnotationFile(std::string);
  std::string ExtractAnnotationFileName(std::string videoFile);

private:
  cv::VideoCapture m_openVideo;
  std::vector<std::pair<int , cv::Rect> > allExtractions;
};
#endif //VIDEO_PICTURE_EXTRACTION_H
