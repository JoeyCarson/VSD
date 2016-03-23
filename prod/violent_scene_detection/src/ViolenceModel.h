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

struct EJDB;

/**
 * The ViolenceModel class encapsulates the feature extraction operations of
 * Gracia's algorithm.  It should persist all relevant feature data in an EJDB database.
 */
class ViolenceModel {

public:
	ViolenceModel(std::string trainingStorePath = "./default_training_set.xml");
	virtual ~ViolenceModel();

	// Index the resource in the file system represented by resourcePath.
	// This path should be compliant for constructing a cv::VideoCapture object.
	void index(std::string resourcePath);

private:
	EJDB * ejdb;
	std::string trainingStorePath;
	cv::Mat trainingStore;

	void trainingStoreInit();
	void ejdbInit();

};

#endif /* VIOLENCEMODEL_H_ */
