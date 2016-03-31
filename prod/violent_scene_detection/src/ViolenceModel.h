/*
 * ViolenceModel.h
 *
 *  Created on: Mar 16, 2016
 *      Author: josephcarson
 */

#ifndef VIOLENCEMODEL_H_
#define VIOLENCEMODEL_H_

#include "LearningKernel.h"

#include <opencv2/opencv.hpp>
#include <boost/bimap.hpp>
#include <boost/filesystem.hpp>
#include <string>

class ImageBlob;

/**
 * The ViolenceModel class encapsulates the feature extraction operations of
 * Gracia's algorithm.  It should persist all relevant feature data in an EJDB database.
 */
class ViolenceModel {

public:

	enum VideoSampleType { TRAINING = 0,
					       CROSS_VALIDATION,
					       TESTING };

	/**
	 * Constructor.
	 * @param trainingStorePath - Path to where the training database file is stored.
	 * It will be opened and read in on construction and persisted upon update and destruction.
	 */
	ViolenceModel(std::string trainingStorePath = "./default_training_set.xml");

	/**
	 * Destructor.  Persists the training set database.
	 */
	virtual ~ViolenceModel();

	/**
	 * Extracts features and returns a list of training vectors, one for each version of the algorithm.
	 * This path must be compliant for constructing a cv::VideoCapture object (e.g. points to a
	 * video file that contains a supported codec video stream).
	 */
	std::vector<cv::Mat> extractFeatures(std::string resourcePath);

	/**
	 * Index the resource in the file system represented by resourcePath into the training store.
	 * This method extracts the feature vector and adds it to the training set.
	 * @param resourcePath - The path to the input file to index as a training sample.
	 * @param isViolent - true (default) if the given is a positive example of violence, false otherwise.
	 */
	void index(std::string resourcePath, bool isViolent = true);

	/**
	 * Returns true if the file at the given path is indexed, false otherwise.
	 */
	bool isIndexed(boost::filesystem::path resourcePath);

	/**
	 * Trains the learning model using the existing indexed training set.
	 */
	void train();

	/**
	 *
	 */
	void predict();

	/**
	 * Clear the data set objects and write empty data to store.
	 */
	void clear();

	/**
	 * Saves the training store to its file path.
	 */
	void persistStore();

private:

	// Path to the data set store.
	std::string trainingStorePath;

	// Training Set Data Structures.
	cv::Mat trainingExampleStore;
	cv::Mat trainingClassStore;
	std::map<std::string, time_t> trainingIndexCache;

	// Cross Validation Set Data Structures.
	cv::Mat xvalExampleStore;
	cv::Mat xvalClassStore;
	std::map<std::string, time_t> xvalIndexCache;

	// Test Set Data Structures.
	cv::Mat testExampleStore;
	cv::Mat testClassStore;
	std::map<std::string, time_t> testIndexCache;

	// ML Kernel.
	LearningKernel learningKernel;

	// Initialize all data sets from the file store.
	void storeInit();

	/**
	 * Initialize the given data structures from the file storage object.
	 */
	void storeInit(cv::FileStorage file, std::string exampleStoreName, cv::Mat &exampleStore,
										 std::string classStoreName,   cv::Mat &classStore,
										 std::string indexCacheName,   std::map<std::string, time_t> &indexCache);

	/**
	 * Builds a training sample based on the number of given image blobs.
	 * Returns a vector of training samples generated for each version of Gracia's
	 * algorithm, e.g. [0] is suitable for training v1, [1] is suitable for training v2.
	 */
	std::vector<cv::Mat> buildSample(std::vector<ImageBlob> blobs);

	/**
	 * Add the training samples for their respective algorithms to their respective
	 * training store.
	 */
	void addTrainingSample(boost::filesystem::path p, std::vector<cv::Mat> trainingSample, bool isViolent);

	/**
	 *
	 */
	void persistStore(cv::FileStorage file, std::string exampleStoreName, const cv::Mat &exampleStore,
										 	std::string classStoreName,   const cv::Mat &classStore,
											std::string indexCacheName,   const std::map<std::string, time_t> &indexCache);

};

#endif /* VIOLENCEMODEL_H_ */
