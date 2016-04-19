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


const uint TARGET_COMMON_WIDTH = 320;
const uint TARGET_COMMON_HEIGHT = 240;

ViolenceModel::ViolenceModel(std::string trainingStorePath)
: trainingStorePath(trainingStorePath)
{
	//storeInit();
}

uint ViolenceModel::size(VideoSetTarget target)
{
	std::map<std::string, time_t> *targetIndex;
	resolveDataStructures(target, NULL, NULL, &targetIndex);
	return targetIndex ? targetIndex->size() : 0;
}

cv::Mat ViolenceModel::trueResults(VideoSetTarget target, bool positive)
{
	cv::Mat *exampleStore, *classStore, predictedClasses, ANDResult;

	if ( learningKernel.isTrained() && resolveDataStructures(target, &exampleStore, &classStore, NULL) )
	{
		// Evaluate the examples against the trained model.
		learningKernel.predict(*exampleStore, predictedClasses);

		if ( classStore->size() == predictedClasses.size() ) {

			cv::Mat classStoreCopy = classStore->clone();
			cv::Mat predictedClassesCopy = predictedClasses.clone();

			if ( !positive )
			{
				//std::cout << "classStoreCopy : " << classStoreCopy << "\n";
				cv::bitwise_xor(classStoreCopy, cv::Scalar(1), classStoreCopy);
				//std::cout << "classStoreCopy NOT : " << classStoreCopy << "\n";
				cv::bitwise_xor(predictedClasses, cv::Scalar(1), predictedClassesCopy);
			}

			predictedClassesCopy.convertTo(predictedClassesCopy, CV_32S);
			cv::bitwise_and(classStoreCopy, predictedClassesCopy, ANDResult);
		}
	}

	return ANDResult;
}

double ViolenceModel::computeError(VideoSetTarget target)
{
	// Compute the MSE for the target dataset when applying it to the trained model.
	cv::Mat *exampleStore, *classStore, predictedClasses, diff;
	resolveDataStructures(target, &exampleStore, &classStore, NULL);

	// Evaluate the examples against the trained model.
	learningKernel.predict(*exampleStore, predictedClasses);

	//std::cout << "learning kernel class predictions: \n" << predictedClasses << "\n";
	//uint totalTP = cv::sum(trueResults(target, true))[0];
	//uint totalTN = cv::sum(trueResults(target, false))[0];
	//std::cout << "totalTP: " << totalTP << " totalTN: " << totalTN << " sum: " << (totalTP + totalTN) << "\n";

	cv::Scalar mean;
	if ( classStore->size() == predictedClasses.size() ) {
		// Ensure that the matrices have compatible types.
		predictedClasses.convertTo(predictedClasses, CV_32S);
		//std::cout << "classStore type: " << classStore->type() << " predictedClasses type: " << predictedClasses.type() <<"\n";

		// Compute the absolute difference between the stored classes (ground truth) and the predicted classes.
		cv::absdiff(*classStore, predictedClasses, diff);
		diff.mul(diff);
		mean = cv::mean(diff);
	} else std::cout << "computeError -> class vectors are not compatible " << "predictedClasses: " << predictedClasses.size() << " classStore: " << classStore->size() << ".\n";

	std::cout << "computeError: " << targetToString(target) << ": " << mean << "\n";
	return mean[0];
}

std::string ViolenceModel::targetToString(VideoSetTarget target)
{
	std::string targetStr;

	switch (target)
	{
		case ViolenceModel::TRAINING: targetStr = "Training"; break;
		case ViolenceModel::X_VALIDATION: targetStr = "Cross Validation"; break;
		case ViolenceModel::TESTING: targetStr = "Testing"; break;
		default: targetStr = "Unknown";
	}

	return targetStr;
}

void ViolenceModel::clear()
{
	std::cout << "Clearing the model index store.\n";
	cv::Mat *exampleStore, *classStore;
	std::map<std::string, time_t> *indexCache;

	// TODO: Can't we do this in a loop?
	resolveDataStructures(ViolenceModel::TRAINING, &exampleStore, &classStore, &indexCache);
	if ( exampleStore && classStore && indexCache) {
		exampleStore->create(0, 0, CV_32F);
		classStore->create(0, 0, CV_32S);
		indexCache->clear();
	}

	resolveDataStructures(ViolenceModel::X_VALIDATION, &exampleStore, &classStore, &indexCache);
	if ( exampleStore && classStore && indexCache) {
		exampleStore->create(0, 0, CV_32F);
		classStore->create(0, 0, CV_32S);
		indexCache->clear();
	}

	resolveDataStructures(ViolenceModel::TESTING, &exampleStore, &classStore, &indexCache);
	if ( exampleStore && classStore && indexCache) {
		exampleStore->create(0, 0, CV_32F);
		classStore->create(0, 0, CV_32S);
		indexCache->clear();
	}

	persistStore();
}

void ViolenceModel::storeInit(cv::FileStorage &file, std::string exampleStoreName, cv::Mat &exampleStore,
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
		//std::cout << "found path: " << path << "\n";
		int modTime = (int)(*iter)[VIOLENCE_MODEL_TRAINING_EXAMPLE_MOD_DATE];
		indexCache[path] = (time_t)modTime;
		iter++;
	}

	// Ensure we go no further the height (rows) are not equivalent.
	std::cout << "classes: " << classStore.size().height << " examples: " << exampleStore.size().height << " indices: " << indexCache.size() << "\n";
	assert(classStore.size().height == exampleStore.size().height && classStore.size().height == indexCache.size());
}

void ViolenceModel::index(VideoSetTarget target, std::string resourcePath, bool isViolent)
{
	boost::filesystem::path path(resourcePath);
	if ( !isIndexed(target, path) ) {

		// Create a VideoCapture instance bound to the path.
		cv::VideoCapture capture(resourcePath);
		std::vector<cv::Mat> trainingSample = extractFeatures(capture, resourcePath);
		addSample(target, path, trainingSample, isViolent);

	} else {
		std::cout << "index -> skipping indexed path: " << resourcePath << "\n";
	}
}

bool ViolenceModel::isIndexed(VideoSetTarget target, boost::filesystem::path resourcePath)
{
	std::map<std::string, time_t> *index = NULL;
	resolveDataStructures(target, NULL, NULL, &index);

	boost::filesystem::path absolutePath = boost::filesystem::absolute(resourcePath);
	std::string indexKey = createIndexKey(absolutePath);

	bool is = index ? index->find( indexKey ) != index->end() : false;
	//std::cout<<"isIndexed: " << is << " path: " << absolutePath.generic_string() << "\n";
	return is;
}

std::string ViolenceModel::createIndexKey(boost::filesystem::path resourcePath)
{
	try {
		resourcePath = boost::filesystem::canonical(resourcePath);
	} catch ( boost::filesystem::filesystem_error &error ) {
		std::cout << "createIndexKey -> assuming path is OpenCV wildcard string.  resourcePath couldn't be canonicalized: " << error.what() <<"\n";
		// Oh well.  Resource path may not exist.
	}

	return resourcePath.generic_string();
}

std::vector<cv::Mat> ViolenceModel::extractFeatures(cv::VideoCapture capture, std::string sequenceName, const uint frameCount)
{

	cv::Mat currentFrame, prevFrame;

	// Create a max heap for keeping track of the largest blobs.
	boost::heap::priority_queue<ImageBlob> topBlobsHeap;
	topBlobsHeap.reserve(GRACIA_K);

	bool capPrevSuccess = false;
	bool capCurrSuccess = false;

	// TODO: Should we need to try to open the capture if it's not?

	// Load the prev frame with the first frame and current with the second
	// so that we can simply loop and compute.
	for ( uint frameIndex = 0, capPrevSuccess = capture.read(prevFrame), capCurrSuccess = capture.read(currentFrame);
		  capPrevSuccess && capCurrSuccess && (frameCount == 0 || frameIndex < ( frameCount - 1 ) );
		  prevFrame = currentFrame, capCurrSuccess = capture.read(currentFrame), frameIndex++ )
	{

		cv::Size targetSize(TARGET_COMMON_WIDTH, TARGET_COMMON_HEIGHT);

		// Convert to grayscale.
		if ( frameIndex == 0 ) {
			cv::Mat grayOut;
			// It's only necessary to gray scale filter the previous frame on the first iteration,
			// as each time the current frame will be equal to the prev frame, which was already filtered.
			cv::cvtColor(prevFrame, grayOut, CV_RGB2GRAY);
			prevFrame = grayOut;
			//prevFrame = ImageUtil::scaleImageIntoRect(prevFrame, targetSize);
			//cv::imshow("first", prevFrame);
			//cv::waitKey();

		}

		// Filter the current frame.
		cv::Mat currentOut;
		cv::cvtColor(currentFrame, currentOut, CV_RGB2GRAY);
		currentFrame = currentOut;

		// Scale to as close to target size as possible.
		//currentFrame = ImageUtil::scaleImageIntoRect(currentFrame, targetSize);
		//cv::imshow("current", currentFrame);
		//cv::waitKey();

		// Compute absolute binarized difference.
		cv::Mat absDiff, binAbsDiff;
		cv::absdiff(prevFrame, currentFrame, absDiff);
		cv::threshold ( absDiff, binAbsDiff, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU );

		// Output binAbsDiff for debug purposes.
		boost::filesystem::path bpath(sequenceName);
		std::stringstream frameName;
		frameName << "bin_abs_diff_" << bpath.stem().string() << "_" << frameIndex;
		//ImageUtil::dumpDebugImage(binAbsDiff, frameName.str());

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

	// If for whatever reason, enough blobs weren't found and added into the heap,
	// load up blank ImageBlobs just to fill out the vector.
	while ( topBlobsHeap.size() < GRACIA_K) {
		std::cout << "Adding blank ImageBlob to top blobs heap.\n";
		topBlobsHeap.emplace(ImageBlob());
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
	cv::Mat *trainEx, *trainCl;
	resolveDataStructures(ViolenceModel::TRAINING, &trainEx, &trainCl, NULL);
	if ( trainEx && trainCl ) {
		learningKernel.train(*trainEx, cv::ml::ROW_SAMPLE, *trainCl);
	} else {
		std::cout << "train -> data structures could not be resolved " << "\n";
	}
}

void ViolenceModel::predict(boost::filesystem::path filePath, float timeInterval)
{
	cv::VideoCapture cap;
	if ( cap.open( filePath.generic_string() ) ) {

		double frameRate = cap.get(CV_CAP_PROP_FPS);
		const uint framesPerExtraction = frameRate * timeInterval;
		// Try to predict across 20 frames each time until the capture is empty.
		for ( double totalFrames = cap.get(CV_CAP_PROP_FRAME_COUNT); totalFrames > 0; totalFrames -= framesPerExtraction )
		{
			cv::Mat output;
			std::vector<cv::Mat> featureRowVector = extractFeatures(cap, "prediction", framesPerExtraction);
			float resp = learningKernel.predict(featureRowVector[0], output);

			std::cout << "learning kernel -> predict returns: " << resp << " predicts: " << output << "\n";
		}

	} else {
		std::cout << "predict -> unable to open " << filePath << " for capture. \n";
	}
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

		// Add the area and compactness.
		v1ExampleVec.push_back( (float)bi.area() ); featureCount++;
		v1ExampleVec.push_back( bi.compactness() ); featureCount++;

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

bool ViolenceModel::resolveDataStructures(VideoSetTarget target, cv::Mat **exampleStore, cv::Mat **classStore , std::map<std::string, time_t> **indexCache, bool readFileIfEmpty)
{
	bool successfullyResolved = true;
	// Training Set Data Structures.
	cv::Mat *examples = NULL;
	cv::Mat *classes = NULL;
	std::map<std::string, time_t> *index = NULL;
	std::string exampleStoreName, classStoreName, indexStoreName;

	switch ( target )
	{
		case ViolenceModel::TRAINING:
			examples = &trainingExampleStore;
			classes = &trainingClassStore;
			index = &trainingIndexCache;
			exampleStoreName = VIOLENCE_MODEL_TRAINING_SET;
			classStoreName = VIOLENCE_MODEL_TRAINING_SET_CLASSES;
			indexStoreName = VIOLENCE_MODEL_TRAINING_FILE_PATHS;

			break;

		case ViolenceModel::TESTING:
			examples = &testExampleStore;
			classes = &testClassStore;
			index = &testIndexCache;
			exampleStoreName = VIOLENCE_MODEL_TEST_SET;
			classStoreName = VIOLENCE_MODEL_TEST_SET_CLASSES;
			indexStoreName = VIOLENCE_MODEL_TEST_FILE_PATHS;

			break;

		case ViolenceModel::X_VALIDATION:
			examples = &xvalExampleStore;
			classes = &xvalClassStore;
			index = &xvalIndexCache;
			exampleStoreName = VIOLENCE_MODEL_XVAL_SET;
			classStoreName = VIOLENCE_MODEL_XVAL_SET_CLASSES;
			indexStoreName = VIOLENCE_MODEL_XVAL_FILE_PATHS;
			break;

		default: {
			std::cout << "VideoSetTarget " << target << " is invalid.";
			assert(false);
			successfullyResolved = false;
		}
	}

	// If any of the structures are empty, initialize them.  Note that we will initialize all structures as they should always
	// be in sync with one another.
	// TODO: It would be great to hide all of this functionality in a base class so there is no temptation to grab a hold
	// 		 of any structures without properly pulling from this method.
	if ( successfullyResolved && readFileIfEmpty && (examples->empty() || classes->empty() || index->empty()) ) {

		cv::FileStorage file;
		std::cout << "initializing data structures for target: " << ViolenceModel::targetToString(target) << "\n";

		bool trainingStoreOpenSuccess = file.open(trainingStorePath, cv::FileStorage::READ);
		if (trainingStoreOpenSuccess) {
			storeInit(file, exampleStoreName, *examples,
							classStoreName, *classes,
							indexStoreName, *index);
		} else {
			std::cout << "Failed opening training store at " << trainingStorePath << "\n";
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
					classStore->create(0, 1, CV_32S);
					std::cout << "new exampleStore size: " << exampleStore->size() << " classStore size:" << classStore->size() <<"\n";
				}

				// Add it to the training store.
				exampleStore->push_back(v1Sample);

				// Add the class (true or false) to the training class store.
				cv::Mat classMat = (cv::Mat_<int>(1,1) << (int)isViolent);
				classStore->push_back(classMat);
				std::cout<<"exampleStore size after add: " << exampleStore->size() << " classStore size: " << classStore->size() <<"\n";

				// Hash the modification date.
				time_t modDate = boost::filesystem::last_write_time(absolutePath);
				(*indexCache)[ createIndexKey(absolutePath) ] = modDate;
				//std::cout << "path: " << absolutePath.generic_string() << " " << modDate << "\n";
				//persistStore();
			}
		}
	}
}

void ViolenceModel::persistStore()
{
	cv::Mat *trEx, *trCl;
	std::map<std::string, time_t> *trInd;
	resolveDataStructures(ViolenceModel::TRAINING, &trEx, &trCl, &trInd, false);

	cv::Mat *xvEx, *xvCl;
	std::map<std::string, time_t> *xvInd;
	resolveDataStructures(ViolenceModel::X_VALIDATION, &xvEx, &xvCl, &xvInd, false);

	cv::Mat *tstEx, *tstCl;
	std::map<std::string, time_t> *tstInd;
	resolveDataStructures(ViolenceModel::TESTING, &tstEx, &tstCl, &tstInd, false);

	cv::FileStorage file;
	// Open the training store file for write and write it.
	bool trainingStoreOpenSuccess = file.open(trainingStorePath, cv::FileStorage::WRITE);
	if (!trainingStoreOpenSuccess) {
		std::cout << "Failed opening training store at " << trainingStorePath << "\n";
		return;
	}

	// Persist the training set.
	if ( trEx && trCl && trInd ) {
		persistStore(file, VIOLENCE_MODEL_TRAINING_SET,*trEx,
						   VIOLENCE_MODEL_TRAINING_SET_CLASSES, *trCl,
						   VIOLENCE_MODEL_TRAINING_FILE_PATHS, *trInd);
	}

	// Persist the cross-validation set.
	if ( xvEx && xvCl && xvInd ) {
		persistStore(file, VIOLENCE_MODEL_XVAL_SET, *xvEx,
						   VIOLENCE_MODEL_XVAL_SET_CLASSES, *xvCl,
						   VIOLENCE_MODEL_XVAL_FILE_PATHS, *xvInd);
	}


	// Persist the test set.
	if ( tstEx && tstCl && tstInd ) {
		persistStore(file, VIOLENCE_MODEL_TEST_SET, *tstEx,
						   VIOLENCE_MODEL_TEST_SET_CLASSES, *tstCl,
						   VIOLENCE_MODEL_TEST_FILE_PATHS, *tstInd);
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
