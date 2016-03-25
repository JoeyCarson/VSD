/*
 * ImageBlob.cpp
 *
 *  Created on: Mar 23, 2016
 *      Author: josephcarson
 */

#include "ImageBlob.h"


//ImageBlob::ImageBlob(const ImageBlob &ref)
//{
//	this->m_points = ref.points();
//}

ImageBlob::ImageBlob(std::vector<cv::Point> points)
: m_points(points)
{

}

ImageBlob::~ImageBlob() {
	// TODO Auto-generated destructor stub
}

std::vector<cv::Point> ImageBlob::points() const
{
	return m_points;
}

double ImageBlob::area() const
{
	return cv::contourArea(m_points);
}

cv::Point2f ImageBlob::centroid() const
{
	cv::Moments mts = cv::moments(m_points);
	return mts.m00 > 0 ? cv::Point2f( mts.m10/mts.m00, mts.m01/mts.m00 ) : cv::Point2f(0,0);
}

double ImageBlob::compactness()
{
	double area = this->area();
	return area > 0 ? pow(cv::arcLength(m_points, true), 2) / (4 * M_PI * area) : 0;
}

bool ImageBlob::operator < (const ImageBlob &bRight) const
{
	double thisArea = this->area();
	double rightArea = bRight.area();
	return thisArea < rightArea;
}

float ImageBlob::distanceFrom(const ImageBlob &other)
{
	cv::Point2f centroidOther = other.centroid();
	cv::Point2f centroidThis = this->centroid();

	float dX = centroidOther.x - centroidThis.x;
	float dY = centroidOther.y - centroidThis.y;
	float d  = sqrt(pow(dX,2) + pow(dY,2));

	//std::cout << "distance: " << d << "\n";

	return d;
}

std::ostream& operator<< (std::ostream &strm, const ImageBlob &a)
{
	cv::Point2f centroid = a.centroid();
	return strm << "area: " << a.area() << " Cx:" << centroid.x << " Cy:" << centroid.y;
}
