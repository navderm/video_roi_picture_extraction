#ifndef VIDEO_PICTURE_EXTRACTION_H
#define VIDEO_PICTURE_EXTRACTION_H

#include <iostream>
#include <string>
#include <fstream>
#include <map>
#include <utility>

#include <opencv2/opencv.hpp>

#include <boost/lexical_cast.hpp>

template <typename Pair>
struct Equal : public std::binary_function<Pair, Pair, bool>
{
  bool operator()(const Pair &x, const Pair &y) const
  {
    return x.first == y.first;
  }
};


class VideoPictureExtraction
{
public:
  VideoPictureExtraction();
  virtual ~VideoPictureExtraction();
  void LoadAndExtract();
  void LoadVideoFileAndExtract(std::string);
  void LoadAnnotationFile(std::string);
  std::string ExtractAnnotationFileName(std::string videoFile);

  void BinImagesOnSize();
  void FindMeanOfBins(const std::multimap<std::string, cv::Mat>& imageBins,
                      std::multimap<std::string, cv::Mat>& meanedImages);
private:
  cv::VideoCapture m_openVideo;
  std::vector<std::pair<int , cv::Rect> > allExtractions;
};
#endif //VIDEO_PICTURE_EXTRACTION_H
