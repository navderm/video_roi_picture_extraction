#include <VideoPictureExtraction/SeaVehiclesClassifier.h>

SeaVehiclesClassifier::SeaVehiclesClassifier()
{

}

SeaVehiclesClassifier::~SeaVehiclesClassifier()
{

}

void SeaVehiclesClassifier::RunAll()
{
  for (int i = 16; i < 321; i = i + 16)
    LoadAndExecute(i);
}

void SeaVehiclesClassifier::LoadAndExecute(int imgSize)
{
  if (imgSize % 16 != 0)
    {
      std::cout << "We don't deal with images not multiple of 16" << std::endl;
    }
  std::string posFile = "/other/aesop_data/AESOP/phase2data/TrainingPositives/ResizedImages/posImageList.txt";
  LoadPositiveImages(posFile, imgSize);
  // create filename based on the imgSize;
  std::string negFile = "/other/aesop_data/mtt_results/phase2data/temp/bg_chips";
  // negFile += boost::lexical_cast<std::string>(imgSize) + "_" + boost::lexical_cast<std::string>(imgSize)+
  negFile += "/negImageList.txt";
  LoadNegativeImages(negFile, imgSize);

  // Only take some of neg imges for now
  std::random_shuffle(m_negImages.begin(), m_negImages.end());

  std::vector<cv::Mat> new_m_negImages;
  if (m_posImages.size() * 2 <= m_negImages.size())
    new_m_negImages = std::vector<cv::Mat> (m_negImages.begin(),
                                            m_negImages.begin() + m_posImages.size()*2);
  else
    new_m_negImages = m_negImages;

  m_negImages.clear();
  m_negImages = new_m_negImages;
  std::cout << "The size of the negImages is : " << m_negImages.size() << std::endl;
  std::cout << "The size of the posImages is : " << m_posImages.size() << std::endl;

  if (m_negImages.size() == 0 || m_posImages.size() == 0)
    return;

  GenerateHOGFeatures(m_posImages, m_posExamples);
  std::cout << "Pos Examples : " << m_posExamples.size() << std::endl;
  GenerateHOGFeatures(m_negImages, m_negExamples);
  std::cout << "Neg Examples : " << m_negExamples.size() << std::endl;

  if (m_posExamples.size() == 0 || m_negExamples.size() == 0)
    return;
  CombineTrainingSets();
  if (m_trainingData.rows ==0 || m_trainingData.cols == 0 ||
      m_classifyingData.rows == 0 || m_classifyingData.cols == 0)
    return;
  std::cout << "M_trainingData:" << m_trainingData.rows << ", " << m_trainingData.cols << ", " << m_trainingData.channels() << std::endl;
  std::cout << "m_class" << m_classifyingData.rows << ", " << m_classifyingData.cols << ", " << m_classifyingData.channels() << std::endl;
  TrainSVM(imgSize);
}


void SeaVehiclesClassifier::LoadPositiveImages(std::string listName, int imgSize)
{
  // clear the previous stash
  m_posImages.clear();

  std::ifstream posFiles(listName.c_str());
  if (!posFiles.is_open())
    {
      std::cout << "Could not open list file ! Have to quit ! " << std::endl;
      return;
    }
  std::vector<std::string> allImagesName;
  std::vector<cv::Mat> allImages;
  std::string imgFile;
  int dummy;
  posFiles >> dummy;
  while (1)
    {
      posFiles >> dummy >> imgFile;
      if (posFiles.eof())
        break;
      cv::Mat image = cv::imread(imgFile, 0);
      assert(!(image.rows % 16)); // no remainder left
      if (image.rows == imgSize)
        {
          // Only work for this particular image size in this iteration
          // leave the rest be
          m_posImages.push_back(image);
        }
    }
  std::cout << "Loaded all the positive images. The size is : " << m_posImages.size() << std::endl;
}

void SeaVehiclesClassifier::LoadNegativeImages(std::string listName, int imgSize)
{

  // clear the previous stash
  m_negImages.clear();

  std::ifstream posFiles(listName.c_str());
  if (!posFiles.is_open())
    {
      std::cout << "Could not open list file ! Have to quit ! " << std::endl;
      return;
    }
  std::vector<std::string> allImagesName;
  std::vector<cv::Mat> allImages;
  std::string imgFile;
  int dummy;
  posFiles >> dummy;
  while (1)
    {
      posFiles >> dummy >> imgFile;
      if (posFiles.eof())
        break;
      cv::Mat image = cv::imread(imgFile, 0);
      std::cout << "Image rows : " << image.rows << std::endl;
      assert(! (image.rows % 16)); // no remainder left
      if (image.rows == imgSize)
        {
          // Only work for this particular image size in this iteration
          // leave the rest be
          m_negImages.push_back(image);
        }
    }
  std::cout << "Loaded all the negative images. The size is : " << m_negImages.size() << std::endl;
}


void SeaVehiclesClassifier::GenerateHOGFeatures(std::vector<cv::Mat> & imageVector,
                                                std::vector<HOGDescription> & descriptions)
{

  std::vector<HOGDescription>::iterator des_it = descriptions.begin();
  std::vector<HOGDescription>::iterator des_it_end = descriptions.end();

  // clearing the space taken by pointers
  for (; des_it != des_it_end; des_it++)
    des_it->clearAll();
  // Then clearing the vector;
  descriptions.clear();

  // will generate float* descriptors
  std::vector<cv::Mat>::iterator it = imageVector.begin();
  std::vector<cv::Mat>::iterator it_end = imageVector.end();
  for (; it != it_end; it++)
    {
      vlfeat::VlHog *hog = vlfeat::vl_hog_new(vlfeat::VlHogVariantDalalTriggs,
                                              (vlfeat::vl_size)8, false);
      cv::Mat_<float> fImg;
      it->convertTo(fImg, CV_32FC3);
      vlfeat::vl_hog_put_image(hog, (float*)fImg.ptr(0), fImg.cols, fImg.rows,
                               fImg.channels(), 8);
      int hogWidth = vlfeat::vl_hog_get_width(hog);
      int hogHeight = vlfeat::vl_hog_get_height(hog);
      int hogDimensions = vlfeat::vl_hog_get_dimension(hog);
      float * features = (float*)malloc(hogWidth * hogHeight * hogDimensions * sizeof(float));
      vlfeat::vl_hog_extract(hog, features);
      HOGDescription newDesc;
      newDesc.descriptors = features;
      newDesc.width = hogWidth;
      newDesc.height = hogHeight;
      newDesc.dimensions = hogDimensions;
      descriptions.push_back(newDesc);

      // visualize the HOG features

      if (SHOWIMAGES)
        {
          int glyphSize = vlfeat::vl_hog_get_glyph_size(hog);
          std::cout << "GlyphSize : " << glyphSize << std::endl;
          int imageHeight = glyphSize * hogHeight;
          int imageWidth = glyphSize * hogWidth;
          std::cout << "show image data :" << imageHeight << ", " << imageWidth << std::endl;
          float * image = (float*)malloc (sizeof(float) * imageWidth * imageHeight);
          vlfeat::vl_hog_render(hog, image, features, hogWidth, hogHeight);

          cv::Mat_<float> tempImg(imageHeight, imageWidth, image);
          std::cout << tempImg.rows << " : " << tempImg.cols  << std::endl;

          cv::imshow("HOG", tempImg);
          cv::imshow("IMAGE", *it);
          cv::waitKey(-1);
        }
      vlfeat::vl_hog_delete(hog);
    }
}


void SeaVehiclesClassifier::CombineTrainingSets()
{
  // take all positive and negative examples and generate
  // m_trainingData and m_classyingData

  m_trainingData.release();
  m_classifyingData.release();// = cv::Mat();

  std::vector<HOGDescription>::iterator posIt = m_posExamples.begin();
  std::vector<HOGDescription>::iterator posItEnd = m_posExamples.end();

  for (; posIt != posItEnd; posIt++)
    {
      cv::Mat featureMat;
      cv::Mat classMat;
      ConvertFeaturesToSVMMat(posIt->descriptors,
                              posIt->height, posIt->width, posIt->dimensions,
                              featureMat,
                              1, classMat);
      m_trainingData.push_back(featureMat);
      m_classifyingData.push_back(classMat);
    }

  std::vector<HOGDescription>::iterator negIt = m_negExamples.begin();
  std::vector<HOGDescription>::iterator negItEnd = m_negExamples.end();

  for(; negIt != negItEnd; negIt++)
    {
      cv::Mat featureMat;
      cv::Mat classMat;
      ConvertFeaturesToSVMMat(negIt->descriptors,
                              negIt->height, negIt->width, negIt->dimensions,
                              featureMat,
                              -1, classMat);
      m_trainingData.push_back(featureMat);
      m_classifyingData.push_back(classMat);
    }
}

void SeaVehiclesClassifier::ConvertFeaturesToSVMMat(float* ptr,
                                                    int height, int width, int dimen,
                                                    cv::Mat& featureMat,
                                                    int classType, cv::Mat& classMat)
{
  featureMat.create(1, height * width * dimen, CV_32FC1);
  for (int col  =  0 ; col < (height * width * dimen); col++)
    {
      featureMat.at<float>(0, col) = ptr[col];
    }
  classMat.create(1, 1, CV_32FC1);
  classMat.at<float>(0,0) = (float)classType;
  return;
}


void SeaVehiclesClassifier::TrainSVM(int imgSize)
{
  // use the trainingData and m_classifyingData to train and save the SVM
  CvSVM newsvm;

  CvSVMParams params;
  params.svm_type = CvSVM::C_SVC;
  params.kernel_type = CvSVM::RBF;
  params.term_crit = cvTermCriteria(CV_TERMCRIT_ITER, 100,  1e-6);
  params.C = 5.0;
  params.gamma = 0.5;

  newsvm.train(m_trainingData, m_classifyingData, cv::Mat(), cv::Mat(), params);
  std::string file = "./" + boost::lexical_cast<std::string>(imgSize)+".svm";
  newsvm.save(file.c_str());
  std::cout << "Saved File ! " << std::endl;
}

void SeaVehiclesClassifier::PredictSVM(std::string filename)
{
  cv::Mat testImg = cv::imread(filename, 0);
  if (testImg.empty())
    std::cout << "Could not find the image !" << std::endl;
  int row = testImg.rows;
  if (row == 0)
    return;
  int rowFactor = testImg.rows % 16;
  int finalRow = row + (16 - rowFactor);

  cv::Mat resizedTestIm;
  cv::resize(testImg, resizedTestIm, cv::Size(finalRow, finalRow));
  std::cout << "Final test image size is : " << resizedTestIm.rows << ", " << resizedTestIm.cols << std::endl;

  std::vector<cv::Mat> testImgVec;
  testImgVec.push_back(resizedTestIm);

  std::vector<HOGDescription> testImgHOGs;
  GenerateHOGFeatures(testImgVec, testImgHOGs);

  cv::Mat featureMat; cv::Mat classMatUnuse;
  ConvertFeaturesToSVMMat(testImgHOGs[0].descriptors,
                          testImgHOGs[0].height,
                          testImgHOGs[0].width,
                          testImgHOGs[0].dimensions,
                          featureMat,
                          0, classMatUnuse);
  // open the right svm file for testing
  std::string folder = "/other/workspace/VideoPictureExtraction/svms/" +
    boost::lexical_cast<std::string>(testImg.rows) + ".svm";
}


void SeaVehiclesClassifier::SaveBinedImage(std::string listName)
{
  std::ifstream imagelistfile(listName.c_str());
  if (!imagelistfile.is_open())
    {
      std::cout << "Could not open list file ! Have to quit ! " << std::endl;
      return ;
    }
  std::vector<cv::Mat> allImages;
  std::vector<std::string> allImgNames;
  std::string imgFile;
  int dummy;
  imagelistfile >> dummy;
  int cntr = 0 ;
  while (1)//
    {
      imagelistfile >> dummy >> imgFile;
      if (imagelistfile.eof())
        break;
      cv::Mat image = cv::imread(imgFile, 0);
      if (image.empty())
        {
          std::cout << "The file could not be opened ! " << std::endl;
          continue;
        }
      assert(image.rows == image.cols);

      // resize here, save, and leave
      int row = image.rows;
      if (row == 0)
        continue;
      int rowFactor = image.rows % 16;
      int finalRow = row + (16 - rowFactor);
      std::cout << "Starting row : " << row << std::endl;
      std::cout << "Final Row : " << finalRow << std::endl;
      cv::Mat resizedIm;
      cv::resize(image, resizedIm, cv::Size(finalRow, finalRow));
      std::cout << "New image:" << resizedIm.rows << ", " << resizedIm.cols << std::endl;
      std::string fiel = "/other/aesop_data/AESOP/phase2data/TrainingPositives/ResizedImages/";
      std::string saveFile = fiel + boost::lexical_cast<std::string>(cntr) + ".jpg";
      std::cout << saveFile << std::endl;
      cv::imwrite(saveFile, resizedIm);
      cntr++;
    }
}
