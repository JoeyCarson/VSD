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
	}
}

void LearningKernel::initRandomTrees()
{
	std::cout << "init random trees\n";
	m_pTrees->setMaxDepth(500);
	m_pTrees->setMinSampleCount(4);

	cv::TermCriteria criteria(cv::TermCriteria::COUNT, 50, 0);
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

	initRandomTrees();

	std::cout << "train -> begin.\n";
	m_pTrees->clear();
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

		std::cout << "persisting trained model\n";
		m_pTrees->save(statModelPath);
	}
}

LearningKernel::~LearningKernel() {
	persist();
}

