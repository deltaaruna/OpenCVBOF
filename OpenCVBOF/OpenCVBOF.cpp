// OpenCVBOF.cpp : Defines the entry point for the console application.
// pqj647

#include "stdafx.h"

#include "stdafx.h"
#include <opencv/cv.h>
#include <opencv/highgui.h>
#include <opencv/ml.h>
#include <stdio.h>
#include <iostream>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/nonfree/features2d.hpp>
using namespace cv;
using namespace std;

using std::cout;
using std::cerr;
using std::endl;
using std::vector;

char ch[30];
char ch2[30];

Ptr<DescriptorMatcher> descriptorMatcher = DescriptorMatcher::create("FlannBased");
Ptr<DescriptorExtractor> descriptorExtractor = new SurfDescriptorExtractor();
SurfFeatureDetector surfDetector(500);

int dictionarySize = 1500;
TermCriteria tc(CV_TERMCRIT_ITER, 10, 0.001);
int retries = 1;
int flags = KMEANS_PP_CENTERS;
BOWKMeansTrainer bowTrainer(dictionarySize, tc, retries, flags);
BOWImgDescriptorExtractor bowDescriptorExtractor(descriptorExtractor, descriptorMatcher);

void collectclasscentroids() {
	IplImage *img;
	int i, j;
	
	for (i = 1; i <= 17; i++){
		printf(ch, "%s%d%s", "dataset/train/aeroplane/",  i,".jpg");
		cout << endl;
		sprintf(ch, "%s%d%s", "dataset/train/aeroplane/", i, ".jpg");
		const char* imageName = ch;
		img = cvLoadImage(imageName, 0);
		vector<KeyPoint> keypoint;
		surfDetector.detect(img, keypoint);
		Mat features;
		descriptorExtractor->compute(img, keypoint, features);
		bowTrainer.add(features);

		printf(ch, "%s%d%s", "dataset/train/bicycle/", i, ".jpg");
		cout << endl;
		sprintf(ch, "%s%d%s", "dataset/train/bicycle/", i, ".jpg");
		imageName = ch;
		img = cvLoadImage(imageName, 0);
		surfDetector.detect(img, keypoint);
		descriptorExtractor->compute(img, keypoint, features);
		bowTrainer.add(features);

		printf(ch, "%s%d%s", "dataset/train/car/", i, ".jpg");
		cout << endl;
		sprintf(ch, "%s%d%s", "dataset/train/car/", i, ".jpg");
		imageName = ch;
		img = cvLoadImage(imageName, 0);
		surfDetector.detect(img, keypoint);
		descriptorExtractor->compute(img, keypoint, features);
		bowTrainer.add(features);
	}
	return;
}

void printResponse(int response)
{
	switch (response) {
	case 1: cout << "Aeroplane detected"; 
		break;
	case 2: cout << "Bicycle detected"; 
		break;
	case 3: cout << "Car detected"; 
		break;
	}
}

int _tmain(int argc, _TCHAR* argv[])
{
	int i, j;
	IplImage *crntImg;
	cout << "Vector quantization..." << endl;
	collectclasscentroids();
	vector<Mat> descriptors = bowTrainer.getDescriptors();
	int count = 0;
	for (vector<Mat>::iterator iter = descriptors.begin(); iter != descriptors.end(); iter++)
	{
		count += iter->rows;
	}
	cout << "Clustering " << count << " features" << endl;
	Mat dictionary = bowTrainer.cluster();
	bowDescriptorExtractor.setVocabulary(dictionary);
	cout << "extracting histograms in the form of BOW for each image " << endl;
	Mat labels(0, 1, CV_32FC1);
	Mat trainingData(0, dictionarySize, CV_32FC1);
	int k = 0;
	vector<KeyPoint> currentKeyPoint;
	Mat currentBowDescriptor;

	for (i = 1; i <= 17; i++){
		printf(ch, "%s%d%s", "dataset/train/aeroplane/", i, ".jpg");
		cout << endl;
		sprintf(ch, "%s%d%s", "dataset/train/aeroplane/", i, ".jpg");
		const char* imageName = ch;
		crntImg = cvLoadImage(imageName, 0);
		surfDetector.detect(crntImg, currentKeyPoint);
		bowDescriptorExtractor.compute(crntImg, currentKeyPoint, currentBowDescriptor);
		trainingData.push_back(currentBowDescriptor);
		labels.push_back((float)1);
	}

	for (i = 1; i <= 17; i++){
		printf(ch, "%s%d%s", "dataset/train/bicycle/", i, ".jpg");
		cout << endl;
		sprintf(ch, "%s%d%s", "dataset/train/bicycle/", i, ".jpg");
		const char* imageName = ch;
		crntImg = cvLoadImage(imageName, 0);
		surfDetector.detect(crntImg, currentKeyPoint);
		bowDescriptorExtractor.compute(crntImg, currentKeyPoint, currentBowDescriptor);
		trainingData.push_back(currentBowDescriptor);
		labels.push_back((float)2);
	}

	for (i = 1; i <= 17; i++){
		printf(ch, "%s%d%s", "dataset/train/car/", i, ".jpg");
		cout << endl;
		sprintf(ch, "%s%d%s", "dataset/train/car/", i, ".jpg");
		const char* imageName = ch;
		crntImg = cvLoadImage(imageName, 0);
		surfDetector.detect(crntImg, currentKeyPoint);
		bowDescriptorExtractor.compute(crntImg, currentKeyPoint, currentBowDescriptor);
		trainingData.push_back(currentBowDescriptor);
		labels.push_back((float)3);
	}
	//}

	//Setting up SVM parameters
	CvSVMParams params;
	params.kernel_type = CvSVM::RBF;
	params.svm_type = CvSVM::C_SVC;
	params.gamma = 0.50625000000000009;
	params.C = 312.50000000000000;
	params.term_crit = cvTermCriteria(CV_TERMCRIT_ITER, 100, 0.000001);
	CvSVM svm;
	
	printf("%s\n", "Training SVM classifier");

	bool res = svm.train(trainingData, labels, cv::Mat(), cv::Mat(), params);

	cout << "Processing evaluation data..." << endl;


	Mat groundTruth(0, 1, CV_32FC1);
	Mat evalData(0, dictionarySize, CV_32FC1);
	k = 0;
	vector<KeyPoint> detectedKeyPoint;
	Mat detectedBowDescriptor;


	Mat results(0, 1, CV_32FC1);;

	for (i = 1; i <= 3; i++){
		printf(ch2, "%s%d%s", "dataset/test/aeroplane/test_", i, ".jpg");
		cout << endl;
		sprintf(ch2, "%s%d%s", "dataset/test/aeroplane/test_", i, ".jpg");
		const char* imageName = ch2;
		crntImg = cvLoadImage(imageName, 0);

		surfDetector.detect(crntImg, detectedKeyPoint);
		bowDescriptorExtractor.compute(crntImg, detectedKeyPoint, detectedBowDescriptor);

		evalData.push_back(detectedBowDescriptor);
		groundTruth.push_back((float)j);
		float response = svm.predict(detectedBowDescriptor);
		results.push_back(response);
		//printf("Result = %f", response);
		printResponse(response);
		cout << endl;
	}

	for (i = 1; i <= 3; i++){
		printf(ch2, "%s%d%s", "dataset/test/bicycle/test_", i, ".jpg");
		cout << endl;
		sprintf(ch2, "%s%d%s", "dataset/test/bicycle/test_", i, ".jpg");
		const char* imageName = ch2;
		crntImg = cvLoadImage(imageName, 0);

		surfDetector.detect(crntImg, detectedKeyPoint);
		bowDescriptorExtractor.compute(crntImg, detectedKeyPoint, detectedBowDescriptor);

		evalData.push_back(detectedBowDescriptor);
		//groundTruth.push_back((float)j);
		float response = svm.predict(detectedBowDescriptor);
		results.push_back(response);
		printResponse(response);
		cout << endl;
	}

	for (i = 1; i <= 3; i++){
		printf(ch2, "%s%d%s", "dataset/test/car/test_", i, ".jpg");
		cout << endl;
		sprintf(ch2, "%s%d%s", "dataset/test/car/test_", i, ".jpg");
		const char* imageName = ch2;
		crntImg = cvLoadImage(imageName, 0);

		surfDetector.detect(crntImg, detectedKeyPoint);
		bowDescriptorExtractor.compute(crntImg, detectedKeyPoint, detectedBowDescriptor);

		evalData.push_back(detectedBowDescriptor);
		groundTruth.push_back((float)j);
		float response = svm.predict(detectedBowDescriptor);
		results.push_back(response);
		printResponse(response);
		cout << endl;
	}
	
	printf(ch2, "%s%d%s", "dataset/test/car/test_", 3,".jpg");
	cout << endl;
	sprintf(ch2, "%s%d%s", "dataset/test/car/test_", 3, ".jpg");

	/**double errorRate = (double)countNonZero(groundTruth - results) / evalData.rows;
	printf("%s%f", "Error rate is ", errorRate);**/
	return 0;
}
