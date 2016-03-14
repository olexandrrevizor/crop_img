#include "opencv2/core/core.hpp"
#include "opencv2/contrib/contrib.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/objdetect/objdetect.hpp"

#include <Windows.h>
#include <string.h>

#include <iostream>
#include <fstream>
#include <sstream>

using namespace cv;
using namespace std;

void detectAndDisplay(Mat* frame, string* img_name);


string fn_haar = string("E:/bachelor/opencv-2.4/sources/data/haarcascades/haarcascade_frontalface_alt_tree.xml");

static string *readTxt(string filename, Mat* images) {
	streampos begin, end;
	std::ifstream file(filename.c_str(), ifstream::in);

	if (!file) {
		string error_message = "No valid input file was given, please check the given filename.";
		CV_Error(CV_StsBadArg, error_message);
	}

	string line, path;
	string *name = new string[15];
	int i = 0;
	while (getline(file, line)) {
		stringstream liness(line);
		getline(liness, path, '\n');

		if (!path.empty()) {
			name[i] = path;
			images[i] = imread(path, CV_LOAD_IMAGE_COLOR);
		}

		if (i >= 14) {
			break;
		}

		i++;
	}
	return name;
}

bool subdirForCrop(const char* filePath)
{
	//This will get the file attributes bitlist of the file
	DWORD fileAtt = GetFileAttributesA(filePath);
	int flag = 0;
	if (fileAtt == INVALID_FILE_ATTRIBUTES)
		flag = -1;
	else if (fileAtt == FILE_ATTRIBUTE_DIRECTORY)
		flag = 1;
	if (flag)
		CreateDirectory(filePath, NULL);
	return  flag;
}

int main(int argc, char** argv)
{
	string* img_name;

	if (argc != 2)
	{
		cout << " Usage: display_image ImageToLoadAndDisplay" << endl;
		exit(1);
	}

	Mat images[15];

	img_name = readTxt(argv[1], images);

	detectAndDisplay(images, img_name);

	waitKey(0);                                     // Wait for a keystroke in the window
	return 0;
}

string formatName(string img_name) {
	size_t pos = img_name.find_last_of("\\");
	string path = img_name.substr(0, pos) + "\\crop\\";
	subdirForCrop(path.c_str());
	string name = img_name.substr(pos + 1, img_name.length());
	name = name.substr(0, name.find(".")) + ".pgm";
	return path + name;
}

void detectAndDisplay(Mat* frame, string* img_name)
{	
	std::vector<Rect> faces;
	cv::Mat frame_gray;
	cv::Mat crop;
	cv::Mat res;
	cv::Mat gray;
	CascadeClassifier face_cascade;

	string name = "";
	
	if (!face_cascade.load(fn_haar)) {
		cout << "Error casacade init!" << std::endl;
		exit(1);
	}

	for (int i = 0; i <= 14; i++) {
		
		if (!frame[i].data) {
			continue;
		}

		name = formatName(img_name[i]);

		cvtColor(frame[i], frame_gray, CV_BGR2GRAY);
		equalizeHist(frame_gray, frame_gray);

		// Set Region of Interest
		cv::Rect roi_c;

		int ac = 0;

		//-- Detect faces
		face_cascade.detectMultiScale(frame_gray, faces, 1.1, 2, 0 | CV_HAAR_SCALE_IMAGE, Size(30, 30));

		for (size_t j = 0; j < faces.size(); j++)
		{
			roi_c.x = faces[j].x;
			roi_c.y = faces[j].y;
			roi_c.width = (faces[j].width);
			roi_c.height = (faces[j].height);

			crop = frame[i](roi_c);
			resize(crop, res, Size(128, 128), 0, 0, INTER_LINEAR); // This will be needed later while saving images
			cvtColor(crop, gray, CV_BGR2GRAY); // Convert cropped image to Grayscale

			imwrite(name, gray);

			Point pt1(faces[j].x, faces[j].y);
			Point pt2(faces[j].x + faces[j].width, faces[j].y + faces[j].height);
			rectangle(frame[i], pt1, pt2, Scalar(226, 65, 7), 1, 8, 0);

		}
	}
	waitKey(0);
	exit(0);
}