#include <VideoPictureExtraction/video_picture_extraction.h>
#include <VideoPictureExtraction/SeaVehiclesClassifier.h>


int main()
{
  //   VideoPictureExtraction vpe;
  SeaVehiclesClassifier svc;
  // while(1)
  //   vpe.LoadAndExtract();
  std::string filename= "/other/aesop_data/AESOP/phase2data/TrainingPositives/imagelist.txt";
  // vpe.BinImagesOnSize(filename, 1);
  // vpe.PrepareSVMData();
  // vpe.TestHOG();

  // svc.SaveBinedImage(filename);
  svc.RunAll();
}









