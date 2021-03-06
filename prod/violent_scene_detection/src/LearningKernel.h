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
 * Implements the Random Forests algorithm as proposed by Gracia.
 * OpenCV supports this algorithm natively.
 * http://docs.opencv.org/3.0-beta/modules/ml/doc/random_trees.html
 */
class LearningKernel {

public:

	/**
	 * Constructor.
	 * @param modelPath - Path to the learning kernel data structures storage.
	 */
	LearningKernel(std::string modelPath = "./default_stat_model.xml");
	virtual ~LearningKernel();

	/**
	 * Returns true if the underlying classifier is trained, false otherwise.
	 */
	bool isTrained() { return m_pTrees && m_pTrees->isTrained(); }

	/**
	 * Trains the learning model with the given training set.
	 * @param trainingSet - A matrix of training samples.
	 * @param layout - The layout of the matrix.  Must be either ROW_SAMPLE or COL_SAMPLE.
	 * @param response - Column vector of training class responses.
	 */
	void train(cv::Mat trainingSet, int layout, cv::Mat response);

	/**
	 * Runs the classifier.
	 * Make a prediction based on the given samples.
	 * @param samples - A matrix of samples (rows).
	 */
	float predict( cv::InputArray samples, cv::OutputArray out );

	/**
	 * Persist the learning kernel data structures.
	 */
	void persist();

private:
	std::string statModelPath;
	cv::Ptr<cv::ml::RTrees> m_pTrees;
	void initRandomTrees();

};

#endif /* LEARNINGKERNEL_H_ */
