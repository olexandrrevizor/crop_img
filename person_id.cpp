#include "opencv2/core/core.hpp"
#include "opencv2/contrib/contrib.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/objdetect/objdetect.hpp"

#include <iostream>
#include <fstream>
#include <sstream>

using namespace cv;
using namespace std;

static void read_csv(const string& filename, vector<Mat>& images, vector<int>& labels, char separator = ';') {
	ifstream file(filename.c_str(), ifstream::in);
	if (!file) {
		string error_message = "No valid input file was given, please check the given filename.";
		CV_Error(CV_StsBadArg, error_message);
	}
	string line, path, classlabel;
	while (getline(file, line)) {
		stringstream liness(line);
		getline(liness, path, separator);
		getline(liness, classlabel);
		if (!path.empty() && !classlabel.empty()) {
			images.push_back(imread(path, 0));
			labels.push_back(atoi(classlabel.c_str()));
		}
	}
}

int main(int argc, const char *argv[]) {
	string fn_haar = string("E:/bachelor/opencv-2.4/sources/data/haarcascades/haarcascade_frontalface_default.xml");
	string fn_csv = string("E:/bachelor/crop_face/at.txt");

	vector<Mat> images;
	vector<int> labels;
	Mat image;

	image = imread("E:/download/1.jpg", CV_LOAD_IMAGE_COLOR);
	

	try {
		read_csv(fn_csv, images, labels);
	}
	catch (cv::Exception& e) {
		cerr << "Error opening file \"" << fn_csv << "\". Reason: " << e.msg << endl;
		exit(1);
	}

	int im_width = images[0].cols;
	int im_height = images[0].rows;
	
	Ptr<FaceRecognizer> model = createLBPHFaceRecognizer();//createFisherFaceRecognizer(); //createEigenFaceRecognizer();  2700
	model->train(images, labels);

	CascadeClassifier haar_cascade;
	haar_cascade.load(fn_haar);

	
	vector< Rect_<int> > faces;

	Mat gray;
	cvtColor(image, gray, CV_BGR2GRAY);
	haar_cascade.detectMultiScale(gray, faces);

	for (int i = 0; i < faces.size(); i++) {

		Rect face_i = faces[i];
		Mat face = gray(face_i);

		Mat face_resized;
		cv::resize(face, face_resized, Size(im_width, im_height), 1.0, 1.0, INTER_CUBIC);
		int prediction = model->predict(face_resized);
		rectangle(image, face_i, CV_RGB(255, 0, 0), 1);
		
		string box_text = format("Person = %d", prediction);

		int pos_x = std::max(face_i.tl().x - 10, 0);
		int pos_y = std::max(face_i.tl().y - 10, 0);
		
		putText(image, box_text, Point(pos_x, pos_y), FONT_HERSHEY_PLAIN, 1.0, CV_RGB(0, 255, 0), 2.0);
	}

	namedWindow("face_recognizer", WINDOW_AUTOSIZE);
	imshow("face_recognizer", image);
	cv::waitKey(0);
	return 0;
}
