#include <cv.h>
#include <highgui.h>

using namespace cv;

int main( int argc, char** argv )
{
   if(argc < 2)
   {
     printf("Please provide an image path.\n");
     return -1;
   }

   char* imageName = argv[1];

   Mat image;
   image = imread( imageName, 1 );

   if( !image.data )
   {
     printf( " No image data \n " );
     return -1;
   }

   Mat filtered_image;
   for(int i = 0; i < 3; i++) {
      bilateralFilter(image, filtered_image, 0, 40, 3, 0);
      image = filtered_image.clone();
   }

   Mat canny_edge;
   int lowThreshold = 10;
   Canny( filtered_image, canny_edge, lowThreshold, lowThreshold*3, 3 );
   int erosion_size = 6;   
   Mat element = getStructuringElement(MORPH_CROSS,
                  Size(2 * erosion_size + 1, 2 * erosion_size + 1), 
                  Point(erosion_size, erosion_size) );
   erode(canny_edge, canny_edge, element);
   dilate(canny_edge, canny_edge, element);
   image.copyTo(filtered_image, canny_edge);

   if(argc > 2)
      imwrite(argv[2], image);
   else
      imwrite(argv[1], image);

   /*
   namedWindow( "Final Image", CV_WINDOW_AUTOSIZE );

   imshow( "Final Image", image );

   waitKey(0);
   */

   return 0;
}
