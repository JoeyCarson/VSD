/*
 * ViolenceModel.cpp
 *
 *  Created on: Mar 16, 2016
 *      Author: josephcarson
 */

#include <iostream>
#include <unistd.h>
#include <ejdb/ejdb.h>

#include "ViolenceModel.h"
#include "ImageUtil.h"

#include <boost/algorithm/string.hpp>
#include <boost/filesystem.hpp>

//#include <opencv/cv.h>
//#include <opencv2/videoio.hpp>
#include <opencv2/opencv.hpp>


#define VIOLENCE_MODEL_DB_NAME "violence_model.db"

ViolenceModel::ViolenceModel()
: ejdb(NULL)
{
	ejdbInit();
}

void ViolenceModel::ejdbInit()
{
	EJDB * tempEJDBPtr = NULL;

	if ( !ejdb ) {

		tempEJDBPtr = ejdbnew();

		if ( !tempEJDBPtr ) {
			std::cerr << "Failed Instantiating EJDB object.";
		} else if ( !ejdbopen(tempEJDBPtr, VIOLENCE_MODEL_DB_NAME, JBOWRITER | JBOCREAT) ) {
			std::cerr << "Failed opening EJDB database.";
			free(tempEJDBPtr); tempEJDBPtr = NULL;
		} else {
			// Success.
			std::cout << "Database opened.";
			this->ejdb = tempEJDBPtr;
		}
	}
}

void ViolenceModel::index(std::string resourcePath)
{
	// Create a VideoCapture instance bound to the path.
	cv::VideoCapture capture(resourcePath);

	// Load the prev frame with the first frame and current with the second
	// so that we can simply loop and compute.
	cv::Mat currentFrame, prevFrame;

	bool capPrevSuccess = false;
	bool capCurrSuccess = false;

	for ( uint i = 0, capPrevSuccess = capture.read(prevFrame), capCurrSuccess = capture.read(currentFrame);
		  capPrevSuccess && capCurrSuccess;
		  prevFrame = currentFrame, capCurrSuccess = capture.read(currentFrame), i++
	    )
	{
		std::cout<< "frame: " << i << "\n";

		// Convert to grayscale.
		if ( i == 0 ) {
			cv::Mat grayOut;
			// It's only necessary to gray scale filter the previous frame on the first iteration,
			// as each time the current frame will be equal to the prev frame, which was already filtered.
			cv::cvtColor(prevFrame, grayOut, CV_RGB2GRAY);
			prevFrame = grayOut;
		}

		// Filter the current frame.
		cv::Mat currentOut;
		cv::cvtColor(currentFrame, currentOut, CV_RGB2GRAY);
		currentFrame = currentOut;

		// Compute absolute binarized difference.
		cv::Mat absDiff, binAbsDiff;
		cv::absdiff(prevFrame, currentFrame, absDiff);
		cv::threshold ( absDiff, binAbsDiff, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU );

		//std::cout << "blah: " << binAbsDiff;

		// Output binAbsDiff for debug purposes.
		boost::filesystem::path bpath(resourcePath);
		std::stringstream frameName;
		frameName << bpath.stem().string() << "_" << i;

		ImageUtil::dumpDebugImage(binAbsDiff, frameName.str());
	}

}

ViolenceModel::~ViolenceModel() {
	// TODO Auto-generated destructor stub
}

