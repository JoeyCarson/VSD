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

	// Returns the size that an image of sourceSize must be scaled to in order to fit into
	// targetSize, such that aspect ratio of the source size is preserved.
	static cv::Size fitSizePreservingAspectRatio(cv::Size sourceSize, cv::Size targetSize);

	// Returns a copy of the given image scaled to fit the given size while preserving the its aspect ratio.
	// The given image is copied into the dead center of a blank image of the given size.
	// This is useful for scaling different videos to be of a common size, while preserving aspect ratio
	// so that features are scaled to a common size without distortion.
	static cv::Mat scaleImageIntoRect(const cv::Mat img, cv::Size size);

	static void printContour(std::vector<cv::Point> c, std::string name = "");

	// Returns randomly shuffled copies of the given examples and classes of a dataset, such that their indices are shuffled equally.
	// Credit to: http://stackoverflow.com/questions/17348854/random-order-shuffle-cvmat-in-opencv
	static void shuffleDataset(const cv::Mat &examplesIn, const cv::Mat &classesIn, cv::Mat* shuffledExamplesOut, cv::Mat *shuffledClassesOut );

	/**
	 * Returns a binary matrix representing whether or not the example at each index was predicted correctly..
	 * @param positive - Whether to detect true positives or true negatives.
	 * @param predictions - Predictions made against a trained ML model.
	 * @param groundTruth - Ground truth of what the predictions should evaluate to.
	 */
	static cv::Mat trueResults(bool positive, const cv::Mat &predictions, const cv::Mat &groundTruth);

	//ImageUtil();
	//virtual ~ImageUtil();

private:
	static void createDebugDirectory();

};

#endif /* IMAGEUTIL_H_ */
