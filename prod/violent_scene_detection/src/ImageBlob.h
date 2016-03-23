/*
 * ImageBlob.h
 *
 *  Created on: Mar 23, 2016
 *      Author: josephcarson
 */

#ifndef IMAGEBLOB_H_
#define IMAGEBLOB_H_

#include <opencv2/opencv.hpp>
#include <vector>

/**
 * Represents a blob in an image and encapsulates operations
 * for computing properties about them.
 */
class ImageBlob {
public:
	/**
	 * Constructor.
	 * @param points - A vector of the pixel coordinates that describes the contour of the blob.
	 */
	ImageBlob(std::vector<cv::Point> points);
	virtual ~ImageBlob();

	// Implementation for ordering of blobs area.
	bool operator < (const ImageBlob &bRight) const;

	// Retrieves a copy of the points.
	std::vector<cv::Point> points() const;
	cv::Point2f centroid() const;
	double compactness();
	double area() const;


private:
	std::vector<cv::Point> m_points;

};

std::ostream& operator << (std::ostream &strm, const ImageBlob &a);

#endif /* IMAGEBLOB_H_ */
