﻿// CannyDetection.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include "pch.h"
#include <iostream>
#include<opencv2/core/core.hpp>
#include<opencv2/highgui/highgui.hpp>
#include<opencv2/imgproc/imgproc.hpp>

#include<iostream>
#include<conio.h>         

using namespace cv;

int main() {
	cv::VideoCapture capWebcam(0);   // declare a VideoCapture object to associate webcam, 0 means use 1st (default) webcam

	if (capWebcam.isOpened() == false)  //  To check if object was associated to webcam successfully
	{
		std::cout << "error: Webcam connect unsuccessful\n"; // if not then print error message
		return(0);            // and exit program
	}

	cv::Mat imgOriginal;        // input image
	cv::Mat imgGrayscale;       // grayscale image
	cv::Mat imgBlurred;         // blured image
	cv::Mat imgCanny;           // Canny edge image

	char charCheckForEscKey = 0;
	int lowTh = 55;
	int highTh = 60;

	while (capWebcam.isOpened()) {            // until the Esc key is pressed or webcam connection is lost
		bool blnFrameReadSuccessfully = capWebcam.read(imgOriginal);   // get next frame

		if (!blnFrameReadSuccessfully || imgOriginal.empty()) {    // if frame read unsuccessfully
			std::cout << "error: frame can't read \n";      // print error message
			break;
		}

		cv::cvtColor(imgOriginal, imgGrayscale, COLOR_BGR2GRAY);                   // convert to grayscale

		//cv::GaussianBlur(imgGrayscale, imgBlurred, cv::Size(5, 5), 1.8);           // Blur Effect             

		cv::Canny(imgGrayscale, imgCanny, lowTh, highTh);       // Canny Edge Image

		//declare windows
		//cv::namedWindow("imgOriginal", WINDOW_NORMAL);
		cv::namedWindow("imgCanny", WINDOW_NORMAL);

		// show windows
		//cv::imshow("imgOriginal", imgOriginal);
		cv::imshow("imgCanny", imgCanny);

		charCheckForEscKey = cv::waitKey(1);        // delay and get key press
	}

	return(0);
}
