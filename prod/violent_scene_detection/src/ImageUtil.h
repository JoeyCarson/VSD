/*
 * ImageUtil.h
 *
 *  Created on: Mar 20, 2016
 *      Author: josephcarson
 */

#ifndef IMAGEUTIL_H_
#define IMAGEUTIL_H_

#include <opencv2/opencv.hpp>

class ImageUtil {

public:

	// Writes the given matrix into the debug output directory.
	static void dumpDebugImage(cv::Mat image, std::string outputFileName);
	static void createDebugDirectory();
	static void printContour(std::vector<cv::Point> c, std::string name = "");
	//ImageUtil();
	//virtual ~ImageUtil();
};

#endif /* IMAGEUTIL_H_ */
