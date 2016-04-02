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

// Names for accessing training set.
#define VIOLENCE_MODEL_TRAINING_SET "violence_model_train"
#define VIOLENCE_MODEL_TRAINING_SET_CLASSES "violence_model_train_classes"
#define VIOLENCE_MODEL_TRAINING_FILE_PATHS "violence_model_training_file_paths"

// Names for accessing cross validation set.
#define VIOLENCE_MODEL_XVAL_SET "violence_model_xval"
#define VIOLENCE_MODEL_XVAL_SET_CLASSES "violence_model_xval_classes"
#define VIOLENCE_MODEL_XVAL_FILE_PATHS "violence_model_xval_file_paths"

// Names for accessing test set.
#define VIOLENCE_MODEL_TEST_SET "violence_model_test"
#define VIOLENCE_MODEL_TEST_SET_CLASSES "violence_model_test_classes"
#define VIOLENCE_MODEL_TEST_FILE_PATHS "violence_model_test_file_paths"


#define VIOLENCE_MODEL_TRAINING_EXAMPLE_MOD_DATE "last_modified"
#define VIOLENCE_MODEL_TRAINING_EXAMPLE_PATH "path"

// This is just a suitable default for now.  Eventually, this should be made configurable.
const uint GRACIA_K = 8;

ViolenceModel::ViolenceModel(std::string trainingStorePath)
: trainingStorePath(trainingStorePath)
{
	storeInit();
}

void ViolenceModel::clear()
{
	std::cout << "Clearing the model index store.\n";
	cv::Mat *exampleStore;
	cv::Mat *classStore;
	std::map<std::string, time_t> *indexCache;

	// TODO: Can't we do this in a loop?
	resolveDataStructures(ViolenceModel::TRAINING, &exampleStore, &classStore, &indexCache);
	if ( exampleStore && classStore && indexCache) {
		exampleStore->create(0, 0, CV_32F);
		classStore->create(0, 0, CV_32F);
		indexCache->clear();
	}

	resolveDataStructures(ViolenceModel::X_VALIDATION, &exampleStore, &classStore, &indexCache);
	if ( exampleStore && classStore && indexCache) {
		exampleStore->create(0, 0, CV_32F);
		classStore->create(0, 0, CV_32F);
		indexCache->clear();
	}

	resolveDataStructures(ViolenceModel::TESTING, &exampleStore, &classStore, &indexCache);
	if ( exampleStore && classStore && indexCache) {
		exampleStore->create(0, 0, CV_32F);
		classStore->create(0, 0, CV_32F);
		indexCache->clear();
	}

	persistStore();
}

void ViolenceModel::storeInit()
{
	cv::FileStorage file;
	bool trainingStoreOpenSuccess = file.open(trainingStorePath, cv::FileStorage::READ);
	if (!trainingStoreOpenSuccess) {
		std::cout << "Failed opening training store at " << trainingStorePath << "\n";
		return;
	}

	cv::Mat *exampleStore;
	cv::Mat *classStore;
	std::map<std::string, time_t> *indexCache;

	// Initialize the training set from the file.
	resolveDataStructures(ViolenceModel::TRAINING, &exampleStore, &classStore, &indexCache);
	if ( exampleStore && classStore && indexCache ) {
		storeInit(file, VIOLENCE_MODEL_TRAINING_SET, *exampleStore,
						VIOLENCE_MODEL_TRAINING_SET_CLASSES, *classStore,
						VIOLENCE_MODEL_TRAINING_FILE_PATHS, *indexCache);
	}

	// Initialize the training set from the file.
	resolveDataStructures(ViolenceModel::X_VALIDATION, &exampleStore, &classStore, &indexCache);
	if ( exampleStore && classStore && indexCache ) {
		storeInit(file, VIOLENCE_MODEL_XVAL_SET, *exampleStore,
						VIOLENCE_MODEL_XVAL_SET_CLASSES, *classStore,
						VIOLENCE_MODEL_XVAL_FILE_PATHS, *indexCache);
	}

	// Initialize the training set from the file.
	resolveDataStructures(ViolenceModel::TESTING, &exampleStore, &classStore, &indexCache);
	if ( exampleStore && classStore && indexCache ) {
		storeInit(file, VIOLENCE_MODEL_TEST_SET, *exampleStore,
						VIOLENCE_MODEL_TEST_SET_CLASSES, *classStore,
						VIOLENCE_MODEL_TEST_FILE_PATHS, *indexCache);
	}

}

void ViolenceModel::storeInit(cv::FileStorage file, std::string exampleStoreName, cv::Mat &exampleStore,
													std::string classStoreName, cv::Mat &classStore,
													std::string indexCacheName, std::map<std::string, time_t> &indexCache)
{
	// Read the data structures in from the training store.
	file[exampleStoreName] >> exampleStore;
	std::cout << exampleStoreName << " loaded. size: " << exampleStore.size() << "\n";

	file[classStoreName] >> classStore;
	std::cout << classStoreName << " loaded. size: " << classStore.size() << "\n";

	cv::FileNode indexedFilePaths = file[indexCacheName];
	cv::FileNodeIterator iter = indexedFilePaths.begin(), end = indexedFilePaths.end();
	while ( iter != end )
	{
		std::string path = (*iter)[VIOLENCE_MODEL_TRAINING_EXAMPLE_PATH];
		int modTime = (int)(*iter)[VIOLENCE_MODEL_TRAINING_EXAMPLE_MOD_DATE];
		indexCache[path] = (time_t)modTime;
		iter++;
	}

	// Ensure we go no further the height (rows) are not equivalent.
	assert(classStore.size().height == exampleStore.size().height && classStore.size().height == indexCache.size());
}

void ViolenceModel::index(VideoSetTarget target, std::string resourcePath, bool isViolent)
{
	boost::filesystem::path path(resourcePath);
	if ( !isIndexed(target, path) ) {
		std::vector<cv::Mat> trainingSample = extractFeatures(resourcePath);
		addSample(target, path, trainingSample, isViolent);
	}
}

bool ViolenceModel::isIndexed(VideoSetTarget target, boost::filesystem::path resourcePath)
{
	std::map<std::string, time_t> *index = NULL;
	resolveDataStructures(target, NULL, NULL, &index);

	boost::filesystem::path absolutePath( boost::filesystem::absolute(resourcePath) );
	bool is = index ? index->find( absolutePath.generic_string() ) != index->end() : false;
	//std::cout<<"isIndexed: " << is << " path: " << absolutePath.generic_string() << "\n";
	return is;
}

std::vector<cv::Mat> ViolenceModel::extractFeatures(std::string resourcePath)
{

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
	std::vector<cv::Mat> trainingSample = buildSample(blobs);
	return trainingSample;
}

void ViolenceModel::train()
{
	learningKernel.train(trainingExampleStore, cv::ml::ROW_SAMPLE, trainingClassStore);
}

std::vector<cv::Mat> ViolenceModel::buildSample(std::vector<ImageBlob> blobs)
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

bool ViolenceModel::resolveDataStructures(VideoSetTarget target, cv::Mat **exampleStore, cv::Mat **classStore , std::map<std::string, time_t> **indexCache)
{
	bool successfullyResolved = true;
	// Training Set Data Structures.
	cv::Mat *examples = NULL;
	cv::Mat *classes = NULL;
	std::map<std::string, time_t> *index = NULL;

	switch ( target )
	{
		case ViolenceModel::TRAINING:
			examples = &trainingExampleStore;
			classes = &trainingClassStore;
			index = &trainingIndexCache;
			break;

		case ViolenceModel::TESTING:
			examples = &testExampleStore;
			classes = &testClassStore;
			index = &testIndexCache;
			break;

		case ViolenceModel::X_VALIDATION:
			examples = &xvalExampleStore;
			classes = &xvalClassStore;
			index = &xvalIndexCache;
			break;

		default: {
			std::cout << "VideoSetTarget " << target << " is invalid.";
			assert(false);
			successfullyResolved = false;
		}
	}

	// Write the addresses if output pointers are given.
	if ( exampleStore ) *exampleStore = examples;
	if ( classStore   ) *classStore   = classes;
	if ( indexCache   ) *indexCache   = index;

	return successfullyResolved;
}

void ViolenceModel::addSample(VideoSetTarget target, boost::filesystem::path path, std::vector<cv::Mat> sample, bool isViolent)
{
	boost::filesystem::path absolutePath = boost::filesystem::absolute(path);
	if ( !isIndexed( target, absolutePath ) )
	{

		cv::Mat *exampleStore, *classStore;
		std::map<std::string, time_t> *indexCache;
		resolveDataStructures(target, &exampleStore, &classStore, &indexCache);

		if ( exampleStore && classStore && indexCache ) {

			std::cout << "training sample at path " << absolutePath << "is not indexed. adding it.\n";

			if ( sample.size() >= 1)
			{
				cv::Mat v1Sample = sample[0];
				// OpenCV size is as follows.  [width (columns), height (rows)].
				// We effectively want to resize the matrix according to the
				// width (columns) of the training sample.
				if ( exampleStore->size().width != v1Sample.size().width )
				{
					std::cout << "updating training store size. current: " << exampleStore->size() <<"\n";
					// Create the training store with 0 rows of the training sample's width (column count).
					exampleStore->create(0, v1Sample.size().width, CV_32F);
					classStore->create(0, 1, CV_32F);
					std::cout << "new trainingExampleStore size: " << exampleStore->size() << " trainingClassStore size:" << classStore->size() <<"\n";
				}

				// Add it to the training store.
				exampleStore->push_back(v1Sample);

				// Add the class (true or false) to the training class store.
				cv::Mat classMat = (cv::Mat_<float>(1,1) << (float)isViolent);
				classStore->push_back(classMat);
				std::cout<<"training store size after add: " << classStore->size() << " trainingClassStore size: " << classStore->size() <<"\n";

				// Hash the modification date.
				time_t modDate = boost::filesystem::last_write_time(absolutePath);
				(*indexCache)[absolutePath.generic_string()] = modDate;
				//std::cout << "path: " << absolutePath.generic_string() << " " << modDate << "\n";
			}
		}
	}
}

void ViolenceModel::persistStore()
{
	cv::FileStorage file;

	// Open the training store file for write and write it.
	bool trainingStoreOpenSuccess = file.open(trainingStorePath, cv::FileStorage::WRITE);
	if (!trainingStoreOpenSuccess) {
		std::cout << "Failed opening training store at " << trainingStorePath << "\n";
		return;
	}

	cv::Mat *exampleStore, *classStore;
	std::map<std::string, time_t> *indexCache;

	// Persist the training set.
	resolveDataStructures(ViolenceModel::TRAINING, &exampleStore, &classStore, &indexCache);
	if ( exampleStore && classStore && indexCache ) {
		persistStore(file, VIOLENCE_MODEL_TRAINING_SET,*exampleStore,
						   VIOLENCE_MODEL_TRAINING_SET_CLASSES, *classStore,
						   VIOLENCE_MODEL_TRAINING_FILE_PATHS, *indexCache);
	}

	// Persist the cross-validation set.
	resolveDataStructures(ViolenceModel::X_VALIDATION, &exampleStore, &classStore, &indexCache);
	if ( exampleStore && classStore && indexCache ) {
		persistStore(file, VIOLENCE_MODEL_XVAL_SET, *exampleStore,
						   VIOLENCE_MODEL_XVAL_SET_CLASSES, *classStore,
						   VIOLENCE_MODEL_XVAL_FILE_PATHS, *indexCache);
	}


	// Persist the test set.
	resolveDataStructures(ViolenceModel::TESTING, &exampleStore, &classStore, &indexCache);
	if ( exampleStore && classStore && indexCache ) {
		persistStore(file, VIOLENCE_MODEL_TEST_SET, *exampleStore,
						   VIOLENCE_MODEL_TEST_SET_CLASSES, *classStore,
						   VIOLENCE_MODEL_TEST_FILE_PATHS, *indexCache);
	}
}

void ViolenceModel::persistStore(cv::FileStorage file, std::string exampleStoreName, const cv::Mat &exampleStore,
													   std::string classStoreName,   const cv::Mat &classStore,
													   std::string indexCacheName,   const std::map<std::string, time_t> &indexCache)
{


	std::cout << "persisting " << exampleStoreName << "\n";
	file << exampleStoreName << exampleStore;
	file << classStoreName << classStore;

	file << indexCacheName << "[";
	for ( std::pair<std::string, time_t> item : indexCache )
	{
		// OpenCV FileStorage doesn't seem to allow storage of longs, so we must cast to int.
		// Luckily int time doesn't overflow until 2038, and we're not really using the timestamp right now anyway.
		file << "{" << VIOLENCE_MODEL_TRAINING_EXAMPLE_PATH << item.first << VIOLENCE_MODEL_TRAINING_EXAMPLE_MOD_DATE << (int)item.second << "}";
	}
	file << "]";

}

ViolenceModel::~ViolenceModel() {
	// Be sure to save the training store when the model is destroyed.
	persistStore();
}
