/*
 * LearningKernel.cpp
 *
 *  Created on: Mar 18, 2016
 *      Author: josephcarson
 */

#include "LearningKernel.h"
//#include <opencv2/ml.hpp>
// http://docs.opencv.org/3.0-beta/modules/ml/doc/statistical_models.html

LearningKernel::LearningKernel(std::string modelPath)
: statModelPath(modelPath),
  m_pTrees( cv::ml::StatModel::load<cv::ml::RTrees>(statModelPath) )
{
	if ( !m_pTrees ) {
		m_pTrees = cv::ml::RTrees::create();
		initRandomTrees();
	}
}

void LearningKernel::initRandomTrees()
{
	m_pTrees->setMaxDepth(200);
	m_pTrees->setMinSampleCount(5);

	cv::TermCriteria criteria(cv::TermCriteria::EPS, 0, 0);
	m_pTrees->setTermCriteria(criteria);
	m_pTrees->setCalculateVarImportance(false);
	m_pTrees->setRegressionAccuracy(0);

	// This is a binary classifier (max of 2 classes).
	m_pTrees->setMaxCategories(2);
}

void LearningKernel::train(cv::Mat trainingSet, int layout, cv::Mat response)
{
	if ( layout != cv::ml::COL_SAMPLE && layout != cv::ml::ROW_SAMPLE ) {
		std::cout << "training set layout " << layout << " is invalid.\n";
		return;
	}

	std::cout << "train -> begin.\n";
	m_pTrees->train(trainingSet, layout, response);
	persist();
	std::cout << "train -> complete.\n";
	//std::cout << "yay!! not crashing anymore!!" << "\n";
}

float LearningKernel::predict( cv::InputArray samples, cv::OutputArray response )
{
	if ( m_pTrees ) {
		float resp = m_pTrees->predict(samples, response);
		return resp;
	}

	return 0;
}

void LearningKernel::persist()
{
	if ( m_pTrees && !m_pTrees->empty() && m_pTrees->isTrained() ) {
			m_pTrees->save(statModelPath);
	}
}

LearningKernel::~LearningKernel() {
	persist();
}

