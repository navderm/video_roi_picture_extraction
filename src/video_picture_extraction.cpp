#include <VideoPictureExtraction/video_picture_extraction.h>

VideoPictureExtraction::VideoPictureExtraction ()
{
}

VideoPictureExtraction::~VideoPictureExtraction ()
{
}

void VideoPictureExtraction::LoadAndExtract()
{
  std::cout << "Enter the video file to open :" ;
  std::string videoFile;
  std::cin >> videoFile;
  if (videoFile.compare("NULL") == 0)
    {
      std::cout << "Blast you ! The necromancer is right behind you ! It shall haunt you until Pi reaches of age !!" << std::endl;
      exit(0);
    }
  std::string annotationFile = ExtractAnnotationFileName(videoFile);
  LoadAnnotationFile(annotationFile);
  LoadVideoFileAndExtract(videoFile);
}


void VideoPictureExtraction::LoadVideoFileAndExtract(std::string videoFile)
{
  std::cout << "The video file is : " << videoFile << std::endl;
  cv::VideoCapture openVideo;
  openVideo.open(videoFile);
  if (!openVideo.isOpened())
    {
      std::cout << "Could not open the vide ! "<< std::endl;
      return;
    }
  for (int cntr = 0 ; cntr < allExtractions.size(); cntr++)
    {
      std::cout << "Going to : position " << allExtractions[cntr].first << std::endl;

      openVideo.set(CV_CAP_PROP_POS_FRAMES, allExtractions[cntr].first);
      cv::Mat newImg;
      openVideo.read(newImg);
      if ((newImg.empty()) || (newImg.rows == 0 || newImg.cols == 0))
        {
          std::cout << "Could not open file ! " << std::endl;
          continue;
        }
      cv::Rect ROI  = allExtractions[cntr].second;
      std::cout << allExtractions[cntr].second.x << ", " << allExtractions[cntr].second.y << ", " <<
        allExtractions[cntr].second.width << ", " << allExtractions[cntr].second.height << std::endl;
      std::cout << "Size : " << newImg.rows << "," << newImg.cols << std::endl;

      // cv::Mat showOldImg = newImg.clone();
      // cv::rectangle(showOldImg, cv::Point(ROI.x, ROI.y), cv::Point(ROI.x + ROI.width, ROI.y + ROI.height), cv::Scalar(255, 0, 0), 3);
      // cv::imshow("Old Image", showOldImg);

      bool widthInc = false;
      bool heightInc = false;
      int maxWidth = std::max(allExtractions[cntr].second.width, allExtractions[cntr].second.height);
      if (maxWidth == allExtractions[cntr].second.width)
        heightInc = true;
      if (maxWidth == allExtractions[cntr].second.height)
        widthInc = true;
      int minWidth = std::min(allExtractions[cntr].second.width, allExtractions[cntr].second.height);
      int useableWidth, useableHeight;
      if ((allExtractions[cntr].second.x + maxWidth > newImg.cols) || (allExtractions[cntr].second.y + maxWidth > newImg.rows))
        {
          // I'll have to pad here
          int diff = maxWidth - minWidth;
          cv::Mat padded;
          int padding = diff + 10 ; // just use 10 . 1 should be safe enough
          std::cout << padding << std::endl;
          padded.create(newImg.rows + 2*padding, newImg.cols + 2*padding, newImg.type());
          padded.setTo(cv::Scalar::all(0));
          newImg.copyTo(padded(cv::Rect(padding, padding, newImg.cols, newImg.rows)));
          // padded.copyTo(newImg);
          newImg = padded;
          ROI.x = ROI.x + padding;
          ROI.y = ROI.y + padding;
        }
      cv::Mat cutPad_x, cutPad_y;
      if (widthInc)
        ROI.x = ROI.x - (maxWidth - minWidth)/2;
      if (ROI.x < 0)
        {
          // basically create a background padding
          ROI.x = ROI.x + (maxWidth - minWidth)/2;
          cutPad_x = newImg(cv::Rect(ROI.x + minWidth, ROI.y, (maxWidth- minWidth)/2, maxWidth));
        }
      if (heightInc)
        ROI.y = ROI.y - (maxWidth - minWidth)/2;
      if (ROI.y < 0)
        {
          // basically create a background padding
          ROI.y = ROI.y + (maxWidth - minWidth)/2;
          cutPad_y = newImg(cv::Rect(ROI.x + minWidth, ROI.y, maxWidth, (maxWidth - minWidth)/2));
        }

      cv::Rect newROI = cv::Rect(ROI.x, ROI.y, maxWidth, maxWidth);
      cv::Mat roiImg = newImg(newROI);

      if (cutPad_x.rows > 0)
        {
          roiImg = roiImg(cv::Rect(0,0,(int)(roiImg.cols - (maxWidth - minWidth)/2), (int)(roiImg.rows)));
          cv::hconcat(cutPad_x, roiImg, roiImg);
        }
      if (cutPad_y.rows > 0)
        {
          roiImg = roiImg(cv::Rect(0,0,(int)(roiImg.cols), (int)(roiImg.rows  - (maxWidth - minWidth)/2)));
          cv::vconcat(cutPad_y, roiImg,  roiImg);
        }

      if (roiImg.rows < 5 && roiImg.cols < 5)
        continue;
      // cv::Mat showNewImg = newImg;
      // cv::rectangle(showNewImg, cv::Point(newROI.x, newROI.y), cv::Point(newROI.x + newROI.width, newROI.y + newROI.height) , cv::Scalar(0, 255, 0), 3);
      // cv::imshow("New Image", showNewImg);
      // cv::imshow("ROIImg", roiImg);
      std::cout << "Size of ROI is : " << roiImg.rows << " , " << roiImg.cols << std::endl;
      // cv::waitKey(100);
      std::string videoFilePath = videoFile.substr(0, videoFile.find_last_of("/"));
      std::string videoFileName = videoFile.substr(videoFile.find_last_of("/") + 1, videoFile.size());
      // I hard code the directory here !
      videoFilePath = "/other/aesop_data/AESOP/phase2data/TrainingPositives/";
      std::string filename = videoFilePath + videoFileName + "_" +
        boost::lexical_cast<std::string>(allExtractions[cntr].first) + "_" +
        boost::lexical_cast<std::string>(allExtractions[cntr].second.x) + "_" +
        boost::lexical_cast<std::string>(allExtractions[cntr].second.y) + ".jpg";
      std::cout << "filename is : "<< filename << std::endl;
      cv::imwrite(filename, roiImg);
    }
}

std::string VideoPictureExtraction::ExtractAnnotationFileName(std::string videoFile)
{
  std::string annotationFile = videoFile + ".ROI_annotations.txt";
  std::cout << "The annotation file : " << annotationFile << std::endl;
  return annotationFile;
}
void VideoPictureExtraction::LoadAnnotationFile(std::string annotationFile)
{
  allExtractions.clear();
  std::ifstream fs(annotationFile.c_str());
  if (!fs.is_open())
    {
      std::cout << "Could not open annotation file ! " << std::endl;
      fs.close();
      return;
    }
  std::string line;
  while(1)
    {
      int a, b, c, d, e, f, g;
      fs >> a >> b >> c >> d >> e >> f >> g;
      if (fs.eof())
        break;
      cv::Rect newRect(d, e, f, g);
      allExtractions.push_back(std::pair<int, cv::Rect>(b, newRect));
    }
  fs.close();
  std::cout << "Total no of ROIs is : " << allExtractions.size() << std::endl;
  for (int cntr = 0 ; cntr < allExtractions.size(); cntr++)
    std::cout << allExtractions[cntr].first << " : " <<
      allExtractions[cntr].second.x << ", " << allExtractions[cntr].second.y << std::endl;
}

void VideoPictureExtraction::BinImagesOnSize()
{
  /*
 ***Create a list of all the images in the directory
 ***Open all the images in the directory
    Create a good bin
    Bin them in ratio of their size
    Extract a mean for each bin
    resize all the images
    Extract HOG features out of the images
  */
  std::string filename= "/other/aesop_data/AESOP/phase2data/TrainingPositives/imagelist.txt";
  std::ifstream imagelistfile(filename.c_str());
  if (!imagelistfile.is_open())
    {
      std::cout  << "Could not open list file ! Have to quit ! " << std::endl;
      return;
    }

  std::vector<cv::Mat> allPositiveDataImages;
  std::string imgFile;
  int dummy ;
  imagelistfile >> dummy;
  while(!imagelistfile.eof())
    {
      imagelistfile >> dummy >> imgFile;
      std::cout << "Image is : " << imgFile << std::endl;
      cv::Mat image = cv::imread(imgFile, 0);
      if (image.empty())
        {
          std::cout << "The file could not be opened" << std::endl;
          continue;
        }
      // our special case where the images are rectangular
      assert(image.rows == image.cols) ;
      allPositiveDataImages.push_back(image);
    }
  std::cout << "The size of all PositiveData images is : " << allPositiveDataImages.size() << std::endl;

  // all the images are in the vector. Put them in the multimap.
  std::multimap<std::string, cv::Mat> imageBins;
  for (int cntr=0; cntr < allPositiveDataImages.size(); cntr++)
    {
      int putIndex =(int)(allPositiveDataImages[cntr].rows / 16);
      std::string putIndexStr = boost::lexical_cast<std::string>(putIndex);
      imageBins.insert(std::make_pair(putIndexStr, allPositiveDataImages[cntr]));
      //       imageBins[putIndexStr] = allPositiveDataImages[cntr];
    }

  // For each multimap, find the mean size of the images and the no of images in the same
  int totalcount  = 0;
  for(std::multimap<std::string,cv::Mat>::iterator it = imageBins.begin(), end = imageBins.end(); it != end; it = imageBins.upper_bound(it->first))
    {
      totalcount += imageBins.count(it->first);
      std::cout << it->first << ":" << imageBins.count(it->first) << std::endl;
    }
  std::cout << "Total count is : " << totalcount << std::endl;
  std::multimap<std::string, cv::Mat> meanedImages;
  FindMeanOfBins(imageBins, meanedImages);

  for (std::multimap<std::string, cv::Mat>::iterator it = meanedImages.begin(), end = meanedImages.end(); it != end; it++)
    {
      std::cout << it->first << " : (" << it->second.rows << ", " << it->second.cols << ")" << std::endl;
    }
}

void VideoPictureExtraction::FindMeanOfBins(const std::multimap<std::string, cv::Mat>& imageBins,
                                            std::multimap<std::string, cv::Mat>& meanedImages)
{
  std::multimap<std::string, cv::Mat>::const_iterator it = imageBins.begin();
  std::multimap<std::string, cv::Mat>::const_iterator it_end = imageBins.end();

  std::vector<std::string> uniqueMembers;
  for (; it != it_end; it++)
    {
      if (uniqueMembers.size() == 0)
        uniqueMembers.push_back(it->first);
      if (uniqueMembers.back() != it->first)
        uniqueMembers.push_back(it->first);
    }

  // Define a bin with means:
  std::map <std::string, std::pair<int, int> > meanDim;

  for (int cntr = 0 ; cntr < uniqueMembers.size(); cntr++)
    {
      std::cout << uniqueMembers[cntr] << ", ";
      std::string currentIndex = uniqueMembers[cntr];
      std::multimap<std::string, cv::Mat>::const_iterator it = imageBins.begin();
      std::multimap<std::string, cv::Mat>::const_iterator it_end = imageBins.end();
      int totalcols =0, totalrows = 0, imgCntr = 0;
      for (; it != it_end; it++)
        {
          if (it->first == currentIndex)
            {
              totalcols += it->second.cols;
              totalrows += it->second.rows;
              imgCntr++;
            }
        }
      totalcols /= imgCntr;
      totalrows /= imgCntr;
      std::pair<int ,int> rowcolval = std::pair<int, int>(totalrows, totalcols);
      std::string cntrStr = boost::lexical_cast<std::string>(cntr);
      meanDim[cntrStr] = std::pair<int, int> (rowcolval);
    }

  for (int cntr = 0 ; cntr < uniqueMembers.size(); cntr++)
    {
      it = imageBins.begin();
      it_end = imageBins.end();
      for (; it != it_end; it++)
        {
          std::pair<int, int> rowcolval = meanDim[it->first];
          cv::Mat resizedIm;
          if (rowcolval.first == 0 || rowcolval.second == 0)
            resizedIm = it->second.clone();
          else
            cv::resize(it->second, resizedIm, cv::Size(rowcolval.second, rowcolval.first));
          meanedImages.insert(std::make_pair(it->first, resizedIm));
        }
    }
}

