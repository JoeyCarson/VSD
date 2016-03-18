/*
 * LearningKernel.h
 *
 *  Created on: Mar 18, 2016
 *      Author: josephcarson
 */

#ifndef LEARNINGKERNEL_H_
#define LEARNINGKERNEL_H_

#include <opencv2/opencv.hpp>

/**
 * Implements the Random Forests algorithm as proprosed by Gracia.
 * OpenCV supports this algorithm natively.
 * See http://docs.opencv.org/2.4/modules/ml/doc/random_trees.html
 */
class LearningKernel {

public:
	LearningKernel();
	virtual ~LearningKernel();

private:
	cv::RandomTrees randomForest;

};

#endif /* LEARNINGKERNEL_H_ */
