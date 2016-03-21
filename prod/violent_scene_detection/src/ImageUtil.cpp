/*
 * ImageUtil.cpp
 *
 *  Created on: Mar 20, 2016
 *      Author: josephcarson
 */
#include <errno.h>
#include <sys/stat.h>
#include <dirent.h>
#include <string>
#include <sstream>
#include "ImageUtil.h"

#define DEBUG_OUTPUT_DIR "./debug"

void ImageUtil::dumpDebugImage(cv::Mat image, std::string outputFileName)
{
	static bool firstCall = true;
	if ( firstCall ) {
		ImageUtil::createDebugDirectory();
		firstCall = false;
	}

	//std::cout << "asdasd " << frameName.str() << "\n";

	std::stringstream outFilePathStreamBase;
	outFilePathStreamBase << DEBUG_OUTPUT_DIR << "/" << outputFileName;

	// Write the OpenCV matrix itself.
	std::stringstream outMatFilePathStream;
	outMatFilePathStream << outFilePathStreamBase.str()  << ".xml";
	std::cout << "adasda: " << outMatFilePathStream.str() << "\n";
	cv::FileStorage file(outMatFilePathStream.str(), cv::FileStorage::WRITE);
	file << outputFileName.c_str() << image;
	file.release();

	std::stringstream outImageFileName;
	outImageFileName << outFilePathStreamBase.str() << ".png";
	std::cout << "out image name: " << outImageFileName.str() << "\n";
	if ( !cv::imwrite(outImageFileName.str().c_str(), image) ) {
		std::cout << "failed to write debug image file." << "\n";
	}
}

void ImageUtil::createDebugDirectory()
{
	DIR * dir = opendir(DEBUG_OUTPUT_DIR);
	if ( dir ) {
		std::stringstream removeCmd;
		removeCmd << "rm -rf " << DEBUG_OUTPUT_DIR ;
		system(removeCmd.str().c_str());
		free(dir);
		dir = NULL;
	} else {
		if ( errno != ENOENT ) {
			std::cout << "couldn't open output dir. " << strerror(errno) << "\n";
			return;
		}
	}

	if ( mkdir(DEBUG_OUTPUT_DIR, S_IRWXU | S_IRWXG | S_IRWXO) != 0 ) {
		std::cout << "can't output debug image.  failed to create debug output dir. " << strerror(errno);
		return;
	}
}
