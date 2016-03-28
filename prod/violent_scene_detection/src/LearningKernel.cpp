/*
 * LearningKernel.cpp
 *
 *  Created on: Mar 18, 2016
 *      Author: josephcarson
 */

#include "LearningKernel.h"

// http://docs.opencv.org/3.0-beta/modules/ml/doc/statistical_models.html

LearningKernel::LearningKernel()
: m_pTrees( cv::ml::RTrees::create( ) )
{
	initRandomTrees();
}

void LearningKernel::initRandomTrees()
{
	m_pTrees->setMaxDepth(1);
	m_pTrees->setMinSampleCount(1);

	//cv::ml::TermCriteria c;
	cv::TermCriteria criteria(cv::TermCriteria::EPS, 0, 0);
	m_pTrees->setTermCriteria(criteria);
	m_pTrees->setCalculateVarImportance(false);

	// This is a binary classifier (max of 2 classes).
	m_pTrees->setMaxCategories(2);
}

void LearningKernel::train(cv::Mat trainingSet, int layout, cv::Mat response)
{
	if ( layout != cv::ml::COL_SAMPLE && layout != cv::ml::ROW_SAMPLE ) {
		std::cout << "training set layout " << layout << " is invalid.\n";
		return;
	}

	m_pTrees->train(trainingSet, layout, response);
	//std::cout << "yay!! not crashing anymore!!" << "\n";
}

LearningKernel::~LearningKernel() {

}

