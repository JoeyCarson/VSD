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

#include <boost/foreach.hpp>

#define DEBUG_OUTPUT_DIR "./debug"

void ImageUtil::dumpDebugImage(cv::Mat image, std::string outputFileName)
{
	static bool firstCall = true;
	if ( firstCall ) {
		ImageUtil::createDebugDirectory();
		firstCall = false;
	}

	std::stringstream outFilePathStreamBase;
	outFilePathStreamBase << DEBUG_OUTPUT_DIR << "/" << outputFileName;

	// Write the OpenCV matrix itself.
	std::stringstream outMatFilePathStream;
	outMatFilePathStream << outFilePathStreamBase.str()  << ".xml";
	//std::cout << "mat: " << outMatFilePathStream.str() << "\n";

	cv::FileStorage file(outMatFilePathStream.str(), cv::FileStorage::WRITE);
	file << outputFileName.c_str() << image;
	file.release();

	std::stringstream outImageFileName;
	outImageFileName << outFilePathStreamBase.str() << ".png";
	//std::cout << "out image name: " << outImageFileName.str() << "\n";

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

cv::Mat ImageUtil::scaleImageIntoRect(const cv::Mat img, cv::Size size)
{
	cv::Mat scaledImage; // Get a copy of the given image scaled to the scaled size.
	cv::Size scaledSize = ImageUtil::fitSizePreservingAspectRatio(img.size(), size);

	std::cout << "original image size: " << img.size() << " target fit size: " << size << " scaledImageSize: " << scaledSize << "\n";

	cv::resize(img, scaledImage, scaledSize);

	int centerX = size.width/2;
	int centerY = size.height/2;
	int quadrantWidth = scaledSize.width / 2;
	int quadrantHeight = scaledSize.height / 2;

	// Create a blank image of the output size.
	cv::Mat output(size, img.type());

	// Target region to copy the scaled image into.
	int x = centerX - quadrantWidth;
	int y = centerY - quadrantHeight;
	int w = centerX + quadrantWidth;
	int h = centerY + quadrantHeight;

	cv::Rect regionRect(x, y, w, h);
	std::cout << "qw: " << quadrantWidth << " qh: " << quadrantHeight << "\n";
	std::cout << "regionRect: " << regionRect << " x: " << x << " y: " << y << " w: " << w << " h: " << h << "\n";

	cv::Mat subRegion(output, regionRect);
	scaledImage.copyTo(subRegion);

	return output;
}

cv::Size ImageUtil::fitSizePreservingAspectRatio(cv::Size sourceSize, cv::Size targetSize)
{
	assert(sourceSize.width > 0 && sourceSize.height > 0 && targetSize.width > 0 && targetSize.height > 0);

	// ratio = new_width / old_width
	// new_height = old_height * ratio
	// then scaled size is (new_width, new height).

	//	float ratio = float(targetSize.width) / float(sourceSize.width);
	//	uint newHeight = sourceSize.height * ratio;
	cv::Size scaledSize;
//	float origAspect = float(sourceSize.width) / float(sourceSize.height);
//	float newAspect = float(targetSize.width) / float(targetSize.height);
//
//	// crop width to be origHeight * newAspect
//	if (origAspect > newAspect) {
//		int tw = (sourceSize.height * targetSize.width) / targetSize.height;
//		//r = cv::Rect( (sourceSize.width - tw)/2, 0, tw, sourceSize.height);
//		scaledSize = cv::Size(tw, sourceSize.height);
//	}
//	else {	// crop height to be origWidth / newAspect
//		int th = (sourceSize.width * targetSize.height) / targetSize.width;
//		//r = cv::Rect(0, (sourceSize.height - th)/2, sourceSize.width, th);
//		scaledSize = cv::Size(sourceSize.width, th);
//	}

	// Dominant dimensions.
	uint dominantTargetDim = targetSize.width > targetSize.height ? targetSize.width : targetSize.height;
	//uint dominantSourceDim = sourceSize.width > sourceSize.height ? sourceSize.width : sourceSize.height;

	// When scaling an image into a window, the new dimensions obviously must not exceed the target dimensions.
	// Whichever is the dominant dimension of the source image, should be the dimension that is respected in the target rectangle.
	float sourceAspectRatio = float(sourceSize.width) / float(sourceSize.height);
	//cv::Size scaledSize;

	// For instance...
	if ( dominantTargetDim == targetSize.height )
	{

		// If the target rectangle is portrait(ish), then the target height is dominant.
		// 		The source width should grow/shrink to fit the target width (the constrained dimension).
		//		The scaled height is the product of (1/aspect, a.k.a aspect^-1) and target width.
		// 		Consider 800/600: 600 ~= (1/1.33) * 800.
		scaledSize.width = targetSize.width;
		scaledSize.height = pow(sourceAspectRatio, -1) * scaledSize.width;

	}
	else if ( dominantTargetDim == targetSize.width )
	{

		// If the target rectangle is landscape(ish), then the target width is dominant.
		// 		The source height should grow/shrink to fit the target height (the constrained dimension).
		// 		The scaled width shall be the product of aspect ratio and the target height.
		// 		800:600 -> 1920:1080. dominant = w. aspect = 1.33.
		scaledSize.height = targetSize.height;
		scaledSize.width = sourceAspectRatio * scaledSize.height;
	}

	return scaledSize;
}

void ImageUtil::printContour(std::vector<cv::Point> c, std::string name)
{
	std::stringstream contourStr;
	BOOST_FOREACH(cv::Point pt, c) { contourStr << "( " << pt.x << ", " << pt.y << " ) "; }
	std::cout << "contour "  << name << ": " << contourStr.str() << "\n";
}
