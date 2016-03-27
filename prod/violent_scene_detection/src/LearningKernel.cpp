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
	m_pTrees->setMaxDepth(200);
	m_pTrees->setMinSampleCount(400);
}

void LearningKernel::train(cv::Mat trainingSet, int layout)
{
	if ( layout != cv::ml::COL_SAMPLE && layout != cv::ml::ROW_SAMPLE ) {
		std::cout << "training set layout " << layout << " is invalid.\n";
		return;
	}

	cv::Mat response;
	m_pTrees->train(trainingSet, layout, response);
}

LearningKernel::~LearningKernel() {
	// TODO Auto-generated destructor stub
}

