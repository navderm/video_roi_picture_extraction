#ifndef VIDEO_PICTURE_EXTRACTION_H
#define VIDEO_PICTURE_EXTRACTION_H

#include <iostream>
#include <string>
#include <fstream>
#include <map>
#include <utility>

#include <opencv2/opencv.hpp>
#include <opencv2/ml/ml.hpp>

#include <boost/lexical_cast.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/shared_array.hpp>

// include all the vlfeat stuff in vlfeat namespace

namespace vlfeat
{
extern "C" {
#include <vlfeat/vl/generic.h>
#include <vlfeat/vl/hog.h>
}
}

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

  void BinImagesOnSize(std::string filename, int classType);
  void FindMeanOfBins(const std::multimap<std::string, cv::Mat>& imageBins,
                      std::multimap<std::string, cv::Mat>& meanedImages);
  void ExtractHOGFeatures(const std::multimap<std::string, cv::Mat>& images,
                          std::vector< std::pair<cv::Mat, boost::shared_array<float> > >& HOGFeatures, int classType);
  void TestHOG();
  void TrainLinearSVM(std::vector<std::pair<cv::Mat, boost::shared_array<float> > >& HOGFeatures);
  void TestImgConversion();
  void ConvertFeaturesToSVMMat(float* ptr, int height, int width, int dimen,
                               cv::Mat& featureMat, int classType,
                               cv::Mat& classMat);
  void PrepareSVMData();
private:
  cv::VideoCapture m_openVideo;
  std::vector<std::pair<int , cv::Rect> > allExtractions;

  // SVM Training
  CvSVM m_svm;
  cv::Mat m_trainingData;
  cv::Mat m_classification;
};
#endif //VIDEO_PICTURE_EXTRACTION_H
