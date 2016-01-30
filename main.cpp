#include <cstdlib>
#include <cstdio>


#include <OpenCV\cv.h>
#include <OpenCV\highgui.h>

#include <stdio.h>

#define OPENCV_ROOT  "E:\bachelor\opencv" 

void cropImage(IplImage * image, IplImage * crop, CvSeq * faceRectSquare, char* new_name);
using namespace std;

int main(int argc, char* argv[])
{
		// variables
	   IplImage * image = 0;						// upload img
	   IplImage* crop = 0;							// crop img
	   CvHaarClassifierCascade * cascade = 0;		// the face detector
	   CvMemStorage * storage = 0;					// memory for detector to use
	   CvSeq* faceRectSeq;							// memory-access interface
	   CvRect* roi;									// detected region rectangle	
	   char* filename;								// upload image name
	   char* new_name;								// name for detected file

        // check param
		if (argc != 3) {
			cout << "Prog usage: " << argv[0] << " <image.ext>" << "crop image name" << endl;
			exit(1);
		}

		filename = argv[1];
		new_name = argv[2];

        //init
        image = cvLoadImage(filename, 1);
		storage = cvCreateMemStorage(0);
		cascade = (CvHaarClassifierCascade *) cvLoad(("E:\\bachelor\\opencv\\data\\haarcascades\\haarcascade_frontalface_default.xml"), 0, 0, 0); // set path to cascade

		//valid init
		if( !image || !storage || !cascade )
		{
			cout << "Initialization failed: " << (!image) ? "can't load image file" : (!cascade) ? "can't load haar-cascade -- check path" : "unable to allocate memory for data storage";
			exit(1);
		}

		// detect faces in image
		faceRectSeq = cvHaarDetectObjects
									  (image, cascade, storage,
									  1.1,                       // increase search scale by 10% each pass
									  3,                         // merge groups of three detections
									  CV_HAAR_DO_CANNY_PRUNING,  // skip regions unlikely to contain a face
									  cvSize(40,40));            // smallest size face to detect = 40x40

		//crop image
		cropImage(image, crop, faceRectSeq, new_name);

		// clean up and release resources
		cvReleaseImage(&image);
		cvReleaseImage(&crop);
		if(cascade) cvReleaseHaarClassifierCascade(&cascade);
		if(storage) cvReleaseMemStorage(&storage);
        return 0;
}

void cropImage(IplImage * image, IplImage * crop, CvSeq * faceRectSquare, char* new_name)
{
   CvRect* roi = (CvRect*)cvGetSeqElem(faceRectSquare, 1);
   
   // create a window to display detected faces
   cvNamedWindow("Croped img", CV_WINDOW_AUTOSIZE);

   crop = cvCreateImage(cvSize(roi->width, roi->height), image->depth, image->nChannels);

   //set detected square
   cvSetImageROI(image, *roi);
   cvCopy(image, crop);
   cvResetImageROI(image);
   cvSaveImage (new_name, crop);
  
   // display face detections
   cvShowImage("Original", image);
   cvShowImage("Croped img", crop);
   cvWaitKey(0);
   cvDestroyWindow("Croped img");
   cvDestroyWindow("Original");
}