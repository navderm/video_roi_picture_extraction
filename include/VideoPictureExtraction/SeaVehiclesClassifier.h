#ifndef VIDEO_PICTURE_EXTRACTION_H_
#define VIDEO_PICTURE_EXTRACTION_H_

#include <iostream>
#include <string>
#include <fstream>
#include <map>
#include <utility>
#include <algorithm>

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

// template <typename Pair>
// struct Equal : public std::binary_function<Pair, Pair, bool>
// {
//   bool operator()(const Pair &x, const Pair &y) const
//   {
//     return x.first == y.first;
//   }
// };


class SeaVehiclesClassifier
{
  struct HOGDescription
  {
    float * descriptors;
    int width;
    int height;
    int dimensions;
    void clearAll()
    {
      free (descriptors);
    }
  };

public:
  SeaVehiclesClassifier();
  virtual ~SeaVehiclesClassifier();
  void RunAll();
  void LoadAndExecute(int);
  void LoadPositiveImages(std::string listName, int imgSize);
  void LoadNegativeImages(std::string listName, int imgSize);
  void GenerateHOGFeatures(std::vector<cv::Mat> & imageVector,
                           std::vector<HOGDescription>& descriptions);
  void CombineTrainingSets();
  void ConvertFeaturesToSVMMat(float* ptr, int height, int width, int dimen,
                               cv::Mat& featureMat, int classType,
                               cv::Mat& classMat);
  void TrainSVM(int imgSize);


  void SaveBinedImage(std::string filename);

private:
  std::vector<cv::Mat> m_posImages;
  std::vector<cv::Mat> m_negImages;
  std::vector<HOGDescription> m_posExamples;
  std::vector<HOGDescription> m_negExamples;

  cv::Mat m_trainingData;
  cv::Mat m_classifyingData;
};
#endif //VIDEO_PICTURE_EXTRACTION_H
