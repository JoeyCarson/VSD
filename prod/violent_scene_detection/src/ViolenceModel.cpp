/*
 * ViolenceModel.cpp
 *
 *  Created on: Mar 16, 2016
 *      Author: josephcarson
 */

#include <cassert>
#include <iostream>
#include <unistd.h>
#include <ejdb/ejdb.h>

#include "ImageBlob.h"
#include "ViolenceModel.h"
#include "ImageUtil.h"

#include <boost/heap/priority_queue.hpp>
#include <boost/algorithm/string.hpp>
#include <boost/filesystem.hpp>
#include <boost/foreach.hpp>

#include <opencv2/opencv.hpp>


#define VIOLENCE_MODEL_DB_NAME "violence_model.db"
#define VIOLENCE_MODEL_TRAINING_SET "violence_model_train"
#define VIOLENCE_MODEL_TRAINING_SET_CLASSES "violence_model_train_classes"

// This is just a suitable default for now.  Eventually, this should be made configurable.
const uint GRACIA_K = 8;

ViolenceModel::ViolenceModel(std::string trainingStorePath)
: trainingStorePath(trainingStorePath)
{
	trainingStoreInit();
}

void ViolenceModel::clear()
{
	trainingExampleStore.create(0, 0, CV_32F);
	trainingClassStore.create(0, 0, CV_32F);
	persistTrainingStore();
}

void ViolenceModel::trainingStoreInit()
{
	cv::FileStorage file;
	bool trainingStoreOpenSuccess = file.open(trainingStorePath, cv::FileStorage::READ);
	if (!trainingStoreOpenSuccess) {
		std::cout << "Failed opening training store at " << trainingStorePath << "\n";
		return;
	}

	// Read the data structures in from the training store.
	file[VIOLENCE_MODEL_TRAINING_SET] >> trainingExampleStore;
	std::cout << "trainingExampleStore loaded. size: " << trainingExampleStore.size() << "\n";

	file[VIOLENCE_MODEL_TRAINING_SET_CLASSES] >> trainingClassStore;
	std::cout << "trainingClassStore loaded. size: " << trainingClassStore.size() << "\n";

	// Ensure we go no further the heigt (rows) are not equivalent.
	assert(trainingClassStore.size().height == trainingExampleStore.size().height);
}

void ViolenceModel::index(std::string resourcePath, bool isViolent)
{
	std::vector<cv::Mat> trainingSample = extractFeatures(resourcePath);
	addTrainingSample(trainingSample, isViolent);
}

std::vector<cv::Mat> ViolenceModel::extractFeatures(std::string resourcePath)
{

	// TODO: Implement a cache lookup mechanism so that we don't retry to do this each time.
	// 		 Also so that we can avoid adding duplicate video files in the training set.

	// Create a VideoCapture instance bound to the path.
	cv::VideoCapture capture(resourcePath);
	cv::Mat currentFrame, prevFrame;

	// Create a max heap for keeping track of the largest blobs.
	boost::heap::priority_queue<ImageBlob> topBlobsHeap;
	topBlobsHeap.reserve(GRACIA_K);

	bool capPrevSuccess = false;
	bool capCurrSuccess = false;

	// Load the prev frame with the first frame and current with the second
	// so that we can simply loop and compute.
	for ( uint i = 0, capPrevSuccess = capture.read(prevFrame), capCurrSuccess = capture.read(currentFrame);
		  capPrevSuccess && capCurrSuccess;
		  prevFrame = currentFrame, capCurrSuccess = capture.read(currentFrame), i++ )
	{
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

		// TODO: At the moment we're just reading these values into local variables.  The intention is to store them
		// in a class member that can store it in a format that is both efficient for learning model computation and
		// persistent store in the file system (so that the arduous process of extracting the features isn't necessary
		// every single time.
		// See http://www.boost.org/doc/libs/1_39_0/libs/bimap/doc/html/boost_bimap/one_minute_tutorial.html
		BOOST_FOREACH(std::vector<cv::Point> cont, contours)
		{
			ImageBlob blob(cont);

			if ( topBlobsHeap.size() < GRACIA_K ) {
				// The heap isn't full yet, we can simply keep adding.
				topBlobsHeap.emplace(blob);
			} else if ( topBlobsHeap.top().area() < blob.area()) {
				// The new blob is larger and the heap is full, so bump out the top one.
				topBlobsHeap.pop();
				topBlobsHeap.emplace(blob);
			}
		}
	}

	// Read the ordered blobs back as an ordered list.
	std::vector<ImageBlob> blobs;
	while ( !topBlobsHeap.empty() ) {
		blobs.push_back( topBlobsHeap.top() );
		topBlobsHeap.pop();
	}

	// Build a single training sample for each algorithm.
	std::vector<cv::Mat> trainingSample = buildTrainingSample(blobs);
	return trainingSample;
}

void ViolenceModel::train()
{
	learningKernel.train(trainingExampleStore, cv::ml::ROW_SAMPLE, trainingClassStore);
}

std::vector<cv::Mat> ViolenceModel::buildTrainingSample(std::vector<ImageBlob> blobs)
{
	assert(blobs.size() == GRACIA_K);
	std::vector<cv::Mat> retVect;

	// Build v1 sample based on the given blobs as a vector.
	std::vector<float> v1ExampleVec;
	uint featureCount = 0;

	for ( uint i = 0; i < blobs.size(); i++ ) {

		ImageBlob bi = blobs[i];

		// Add the area.
		v1ExampleVec.push_back( (float)bi.area() ); featureCount++;

		// Centroid x and y.
		v1ExampleVec.push_back( bi.centroid().x ); featureCount++;
		v1ExampleVec.push_back( bi.centroid().y ); featureCount++;

		// Compute the distances of this blob from all other blobs.
		for ( uint j = 0; j < blobs.size(); j++ ) {
			if ( i != j ) {
				ImageBlob bj = blobs[j];
				v1ExampleVec.push_back( bi.distanceFrom(bj) ); featureCount++;
			}
		}
	}

	// Create a matrix based on this vector so that it can easily be added
	// to store as a row.  A cv::Mat created from an std::vector is effectively
	// a column vector.  We want to eventually store it as a row, so it must
	// also be transposed before we add it to the output std::vector.
	cv::Mat example1Mat(v1ExampleVec);
	retVect.push_back( example1Mat = example1Mat.t() );

	return retVect;
}

void ViolenceModel::addTrainingSample(std::vector<cv::Mat> trainingSample, bool isViolent)
{
	if ( trainingSample.size() >= 1 )
	{
		cv::Mat v1Sample = trainingSample[0];
		// OpenCV size is as follows.  [width (columns), height (rows)].
		// We effectively want to resize the matrix according to the
		// width (columns) of the training sample.
		if ( trainingExampleStore.size().width != v1Sample.size().width )
		{
			std::cout << "updating training store size. current: " << trainingExampleStore.size() <<"\n";
			// Create the training store with 0 rows of the training sample's width (column count).
			trainingExampleStore.create(0, v1Sample.size().width, CV_32F);
			trainingClassStore.create(0, 1, CV_32F);
			std::cout << "new trainingExampleStore size: " << v1Sample.size() << " trainingClassStore size:" << trainingClassStore.size() <<"\n";
		}

		// Add it to the training store.
		trainingExampleStore.push_back(v1Sample);

		// Add the class (true or false) to the training class store.
		cv::Mat classMat = (cv::Mat_<float>(1,1) << (float)isViolent);
		trainingClassStore.push_back(classMat);
		std::cout<<"training store size after add: " << trainingExampleStore.size() << " trainingClassStore size: " << trainingClassStore.size() <<"\n";

		// Save the training store.
		persistTrainingStore();
	}
}

void ViolenceModel::persistTrainingStore()
{
	cv::FileStorage file;

	// Open the training store file for write and write it.
	bool trainingStoreOpenSuccess = file.open(trainingStorePath, cv::FileStorage::WRITE);
	if (!trainingStoreOpenSuccess) {
		std::cout << "Failed opening training store at " << trainingStorePath << "\n";
		return;
	}

	std::cout << "persisting training store" << "\n";
	//std::cout << "training classes: \n" << trainingClassStore << "\n";
	file << VIOLENCE_MODEL_TRAINING_SET << trainingExampleStore;
	file << VIOLENCE_MODEL_TRAINING_SET_CLASSES << trainingClassStore;
}

ViolenceModel::~ViolenceModel() {
	// Be sure to save the training store when the model is destroyed.
	persistTrainingStore();
}
