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
#include <boost/foreach.hpp>

//#include <opencv/cv.h>
//#include <opencv2/videoio.hpp>
#include <opencv2/opencv.hpp>


#define VIOLENCE_MODEL_DB_NAME "violence_model.db"
#define VIOLENCE_MODEL_TRAINING_SET "violence_model_train"

ViolenceModel::ViolenceModel(std::string trainingStorePath)
: ejdb(NULL),
  trainingStorePath(trainingStorePath)
{
	ejdbInit();
	trainingStoreInit();
}

void ViolenceModel::trainingStoreInit()
{
	cv::FileStorage file;
	bool trainingStoreOpenSuccess = file.open(trainingStorePath, cv::FileStorage::READ | cv::FileStorage::WRITE);
	if (!trainingStoreOpenSuccess) {
		std::cout << "Failed opening training store at " << this->trainingStorePath << "\n";
		return;
	}

	file[VIOLENCE_MODEL_TRAINING_SET] >> trainingStore;
	std::cout << "trainingStore is " << trainingStore << "\n";
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
	cv::Mat currentFrame, prevFrame;

	bool capPrevSuccess = false;
	bool capCurrSuccess = false;

	// Load the prev frame with the first frame and current with the second
	// so that we can simply loop and compute.
	for ( uint i = 0, capPrevSuccess = capture.read(prevFrame), capCurrSuccess = capture.read(currentFrame);
		  capPrevSuccess && capCurrSuccess;
		  prevFrame = currentFrame, capCurrSuccess = capture.read(currentFrame), i++ )
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

		// Output binAbsDiff for debug purposes.
		boost::filesystem::path bpath(resourcePath);
		std::stringstream frameName;
		frameName << "bin_abs_diff_" << bpath.stem().string() << "_" << i;
		ImageUtil::dumpDebugImage(binAbsDiff, frameName.str());

		// Find the contours (blobs) and use them to compute centroids, area, etc.
		// http://opencv.itseez.com/2.4/doc/tutorials/imgproc/shapedescriptors/moments/moments.html?highlight=moment#code
		std::vector<std::vector<cv::Point>> contours;
		std::vector<cv::Vec4i> hierarchy;
		cv::findContours(binAbsDiff, contours, hierarchy, CV_RETR_LIST, CV_CHAIN_APPROX_NONE);

		// Extract the centroid, area, and compactness.
		uint contourIndex = 0;
		std::vector<cv::Point2f> blobCentroids(contours.size());
		std::vector<double> blobAreas(contours.size());
		std::vector<double> blobCompactness(contours.size());

		// TODO: At the moment we're just reading these values into local variables.  The intention is to store them
		// in a class member that can store it in a format that is both efficient for learning model computation and
		// persistent store in the file system (so that the arduous process of extracting the features isn't necessary
		// every single time.
		// See http://www.boost.org/doc/libs/1_39_0/libs/bimap/doc/html/boost_bimap/one_minute_tutorial.html
		BOOST_FOREACH(std::vector<cv::Point> cont, contours)
		{
			cv::Moments mts = cv::moments(cont);

			// Contours that intersect one another may yield a zero area (m00) for butterfly-shaped contours.
			// All fields in Moments object end up being 0.  Does this mean that we shouldn't include them?
			// http://docs.opencv.org/2.4/modules/imgproc/doc/structural_analysis_and_shape_descriptors.html?highlight=moments#moments
			cv::Point2f centroid = mts.m00 > 0 ? cv::Point2f( mts.m10/mts.m00, mts.m01/mts.m00 ) : cv::Point2f(0,0);
			double area = cv::contourArea(cont, false);
			double compactness = area > 0 ? pow(cv::arcLength(cont, true), 2) / (4 * M_PI * area) : 0;

			std::stringstream contourName; contourName << contourIndex++;
			//ImageUtil::printContour(cont, contourName.str());
			std::cout << "area: " << area << "\n";
			std::cout << "centroid: x:" << centroid.x << " y: " << centroid.y << "\n";
			std::cout << "compactness: " << compactness << "\n";

			blobCentroids.push_back(centroid);
			blobAreas.push_back(area);
			blobCompactness.push_back(compactness);
		}
	}

}

ViolenceModel::~ViolenceModel() {
	// TODO Auto-generated destructor stub
}

