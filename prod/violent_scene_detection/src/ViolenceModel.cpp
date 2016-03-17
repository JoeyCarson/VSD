/*
 * ViolenceModel.cpp
 *
 *  Created on: Mar 16, 2016
 *      Author: josephcarson
 */

#include "ViolenceModel.h"
#include "opencv2/opencv.hpp"

ViolenceModel::ViolenceModel(std::string filename)
:m_fileName(filename),
 m_videoCap(m_fileName)
{
	std::cout << "ViolenceModel: fileName:" << fileName() << " isOpened: " << m_videoCap.isOpened();
}

std::string ViolenceModel::fileName()
{
	return m_fileName;
}

ViolenceModel::~ViolenceModel() {
	// TODO Auto-generated destructor stub
}

