/*
 * ViolenceModel.h
 *
 *  Created on: Mar 16, 2016
 *      Author: josephcarson
 */

#ifndef VIOLENCEMODEL_H_
#define VIOLENCEMODEL_H_

#include <string>
#include <opencv2/opencv.hpp>

class ViolenceModel {

public:
	ViolenceModel(std::string filename);
	std::string fileName();
	virtual ~ViolenceModel();

private:
	std::string  m_fileName;
	cv::VideoCapture m_videoCap;


};

#endif /* VIOLENCEMODEL_H_ */
