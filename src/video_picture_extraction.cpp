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

