/*
 * ViolenceModel.h
 *
 *  Created on: Mar 16, 2016
 *      Author: josephcarson
 */

#ifndef VIOLENCEMODEL_H_
#define VIOLENCEMODEL_H_

#include <opencv2/opencv.hpp>
#include <boost/bimap.hpp>
#include <string>

class ImageBlob;

/**
 * The ViolenceModel class encapsulates the feature extraction operations of
 * Gracia's algorithm.  It should persist all relevant feature data in an EJDB database.
 */
class ViolenceModel {

public:

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

	// Index the resource in the file system represented by resourcePath.
	// This path should be compliant for constructing a cv::VideoCapture object.
	void index(std::string resourcePath);

private:

	std::string trainingStorePath;
	cv::Mat trainingStore;

	void trainingStoreInit();

	/**
	 * Builds a training sample based on the number of given image blobs.
	 * Returns a vector of training samples generated for each version of Gracia's
	 * algorithm, e.g. [0] is suitable for training v1, [1] is suitable for training v2.
	 */
	std::vector<cv::Mat> buildTrainingSample(std::vector<ImageBlob> blobs);

	/**
	 * Add the training samples for their respective algorithms to their respective
	 * training store.
	 */
	void addTrainingSample(std::vector<cv::Mat> trainingSample);


	/**
	 * Saves the training store to its file path.
	 */
	void persistTrainingStore();
};

#endif /* VIOLENCEMODEL_H_ */
