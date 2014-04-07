#include <VideoPictureExtraction/video_picture_extraction.h>
#include <VideoPictureExtraction/SeaVehiclesClassifier.h>


int main()
{
   VideoPictureExtraction vpe;
  while(1)
    vpe.LoadAndExtract();

  // SeaVehiclesClassifier svc;
  // std::string filename= "/other/aesop_data/AESOP/phase2data/TrainingPositives/imagelist.txt";
  // svc.SaveBinedImage(filename);
  svc.RunAll();
}









