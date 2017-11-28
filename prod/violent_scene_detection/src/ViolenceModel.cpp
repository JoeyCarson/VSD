/*
 * ViolenceModel.cpp
 *
 *  Created on: Mar 16, 2016
 *      Author: josephcarson
 */

#include <cassert>
#include <iostream>
#include <unistd.h>
#include <sys/time.h>
#include <stdlib.h>

#include "ImageBlob.h"
#include "ViolenceModel.h"
#include "ImageUtil.h"

#include <boost/heap/priority_queue.hpp>
#include <boost/algorithm/string.hpp>
#include <boost/filesystem.hpp>
#include <boost/foreach.hpp>

#include <opencv2/opencv.hpp>

// Names for accessing training set.
#define VIOLENCE_MODEL_TRAINING_SET "violence_model_train"
#define VIOLENCE_MODEL_TRAINING_SET_CLASSES "violence_model_train_classes"
#define VIOLENCE_MODEL_TRAINING_FILE_PATHS "violence_model_training_file_paths"


#define VIOLENCE_MODEL_TRAINING_EXAMPLE_MOD_DATE "last_modified"
#define VIOLENCE_MODEL_TRAINING_EXAMPLE_PATH "path"

// This is just a suitable default for now.  Eventually, this should be made configurable.
const uint GRACIA_K = 8;


const uint TARGET_COMMON_WIDTH = 320;
const uint TARGET_COMMON_HEIGHT = 240;

ViolenceModel::ViolenceModel(std::string trainingStorePath)
: trainingStorePath(trainingStorePath),
  trainingExampleStore(NULL),
  trainingClassStore(NULL),
  trainingIndexCache(NULL)
{

}

uint ViolenceModel::size()
{
	std::map<std::string, time_t> *targetIndex = NULL;
	resolveDataStructures(NULL, NULL, &targetIndex);
	return targetIndex ? targetIndex->size() : 0;
}

//double ViolenceModel::computeError(/*VideoSetTarget target*/)
//{
//	// Compute the MSE for the target dataset when applying it to the trained model.
//	cv::Mat *exampleStore, *classStore, predictedClasses, diff;
//	resolveDataStructures(/*target,*/ &exampleStore, &classStore, NULL);
//
//	// Evaluate the examples against the trained model.
//	learningKernel.predict(*exampleStore, predictedClasses);
//
//	//std::cout << "learning kernel class predictions: \n" << predictedClasses << "\n";
//	uint totalTP = cv::sum(trueResults(/*target,*/ true))[0];
//	uint totalTN = cv::sum(trueResults(/*target,*/ false))[0];
//	std::cout << "totalTP: " << totalTP << " totalTN: " << totalTN << " sum: " << (totalTP + totalTN) << "\n";
//
//	cv::Scalar mean;
//	if ( classStore->size() == predictedClasses.size() ) {
//		// Ensure that the matrices have compatible types.
//		predictedClasses.convertTo(predictedClasses, CV_32S);
//		//std::cout << "classStore type: " << classStore->type() << " predictedClasses type: " << predictedClasses.type() <<"\n";
//
//		// Compute the absolute difference between the stored classes (ground truth) and the predicted classes.
//		cv::absdiff(*classStore, predictedClasses, diff);
//		diff.mul(diff);
//		mean = cv::mean(diff);
//	} else std::cout << "computeError -> class vectors are not compatible " << "predictedClasses: " << predictedClasses.size() << " classStore: " << classStore->size() << ".\n";
//
//	std::cout << "computeError: " << targetToString(target) << ": " << mean << "\n";
//	return mean[0];
//}
//


void ViolenceModel::graciaCrossValidate(uint k )
{

	cv::Mat *examples, *classes;
	if ( resolveDataStructures(&examples, &classes, NULL) ) {

		// TODO: Figure out why those values that are 1 in classes come out as 255 when == 1 expression is used.
		//		 In the meantime, just divide by 255 to yield 1.
		cv::Mat positives = ((*classes) == 1)/255;
		cv::Mat negatives = ((*classes) == 0)/255;

		//std::cout << "pos: " << positives << "\n";
		//std::cout << "neg: " << negatives << "\n";

		std::cout << "graciaCrossValidate\n";

		uint positiveCount = cv::sum(positives)[0];
		uint negativeCount = cv::sum(negatives)[0];
		uint equalizedCount = std::min(positiveCount, negativeCount);
		std::cout << "negatives: " << negativeCount << " positives: " << positiveCount << " min: " << equalizedCount << "\n";
		equalizedCount /= 2;

		cv::Mat shuffledExamples, shuffledClasses, randomPositiveEx, randomPositiveCl, randomNegativeEx, randomNegativeCl;
		ImageUtil::shuffleDataset(*examples, *classes, &shuffledExamples, &shuffledClasses);
		std::cout << "shEx size: " << shuffledExamples.size() << " shCl size: " << shuffledClasses.size() << "\n";

		uint dsi = 0; // This is horribly inefficient!! Shame!!
		cv::Mat validationEx, validationCl;
		for ( dsi = 0; dsi < shuffledExamples.size().height; dsi++ ) {

			if ( shuffledClasses.at<int>(dsi, 0) == 1 && randomPositiveCl.size().height != equalizedCount ) {
				//std::cout << "found positive at dsi: " << dsi << "\n";
				randomPositiveCl.push_back(shuffledClasses.row(dsi));
				randomPositiveEx.push_back(shuffledExamples.row(dsi));
			} else if ( shuffledClasses.at<int>(dsi, 0) == 0 && randomNegativeCl.size().height != equalizedCount ) {
				//std::cout << "found negative at dsi: " << dsi << "\n";
				randomNegativeCl.push_back(shuffledClasses.row(dsi));
				randomNegativeEx.push_back(shuffledExamples.row(dsi));
			} else  {
				// If the sample can't go into one of the specific training sets, put it into validation set.
				validationEx.push_back( shuffledExamples.row(dsi) );
				validationCl.push_back( shuffledClasses.row(dsi) );
			}

		}

		std::cout << "randNegClSize: " << randomNegativeCl.size() << " randPosClSize: " << randomPositiveCl.size() << "\n";
		std::cout << "randNegExSize: " << randomNegativeEx.size() << " randPosExSize: " << randomPositiveEx.size() << "\n";
		std::cout << "the rest of randomized data set will be used for validation from row " << dsi << " to " << shuffledClasses.size().height << "\n";
		assert(randomNegativeCl.size() == randomPositiveCl.size() && randomNegativeEx.size() == randomPositiveEx.size());
		cv::Mat randomizedTrainingEx, randomizedTrainingCl;

		// Add the random positive and negative training examples and classes together.
		randomizedTrainingEx.push_back(randomPositiveEx);
		randomizedTrainingEx.push_back(randomNegativeEx);

		randomizedTrainingCl.push_back(randomPositiveCl);
		randomizedTrainingCl.push_back(randomNegativeCl);

		std::cout << "randTrainEx: " << randomizedTrainingEx.size() << " randomizedTrainingCl: " << randomizedTrainingCl.size() << "\n";
		cv::Mat randTrainExCopy, randTrainClCopy;
		learningKernel.train(randomizedTrainingEx, cv::ml::ROW_SAMPLE, randomizedTrainingCl);

		cv::Mat predictedCl;
		std::cout << "valEx " << validationEx.size() << " valCl " << validationCl.size() << "\n";

		cv::Mat pos = ((validationCl == 1) &= 1);
		cv::Mat neg = ((validationCl == 0) &= 1);
		uint posCount = cv::sum(pos)[0];
		uint negCount = cv::sum(neg)[0];
		std::cout << "validations positives : " << posCount << " validation negatives: " << negCount << "\n";

		learningKernel.predict(validationEx, predictedCl);
		int TP = cv::sum(ImageUtil::trueResults(true, predictedCl, validationCl))[0];
		int TN = cv::sum(ImageUtil::trueResults(false, predictedCl, validationCl))[0];
		int totalAcc = TP + TN;
		std::cout << "true positives: " << TP << " true negatives: " << TN << " acc: " << (float(totalAcc) / predictedCl.size().height) << "\n";

	}

}

void ViolenceModel::crossValidate(uint k)
{
	cv::Mat *examples, *classes;
	if ( resolveDataStructures(&examples, &classes, NULL) ) {

		std::cout << "cross validate -> k:" << k << "\n";

		// First get a randomly ordered copy of the training set.
		// Classes and examples are randomized on the same indices, meaning that they're still aligned example to class correctly.
		cv::Mat randomExamples, randomClasses;
		ImageUtil::shuffleDataset(*examples, *classes, &randomExamples, &randomClasses);
		//std::cout /*<< "randomExamples: " << randomExamples*/ << " randomClasses: " << randomClasses << "\n";

		uint ki_size = examples->size().height / k;
		std::cout << "\ncross validate ki_size: " << ki_size << "\n\n";
		cv::Mat randomSubExamples;
		cv::Mat randomSubClasses;

		// The previously used examples and classes.  This allows us to step through the randomized samples efficiently,
		// while remembering the ones we used in the past.
		std::vector<cv::Mat> prevSubExamples, prevSubClasses;
		uint runningTP = 0, runningTN = 0;

		for ( uint ki = 0; ki < k; ki++ )
		{
			// The offset into the training data that represents the beginning of region ki.
			uint k_offset_begin = ki * ki_size;
			uint k_offset_end   = k_offset_begin + ki_size - 1;
			std::cout << "k_offset_begin: " << k_offset_begin << " k_offset_end: " << k_offset_end << "\n";

			// The range of this sub region in the data.
			cv::Range k_range(k_offset_begin, k_offset_end);
			//std::cout << "k_range: " << k_range << "\n";
			randomSubExamples = randomExamples.rowRange(k_range).clone();
			randomSubClasses = randomClasses.rowRange(k_range).clone();

			// The range from the first element after
			cv::Range kplus1_to_end(k_offset_end + 1, examples->size().height - 1);

			cv::Mat currentTrainingExamples = randomExamples.rowRange(kplus1_to_end);
			cv::Mat currentTrainingClasses = randomClasses.rowRange(kplus1_to_end);

			// Add any previous sub regions that we used as validation example/class data.
			for ( uint i = 0; i < prevSubExamples.size(); i++ ) {
				currentTrainingExamples.push_back( prevSubExamples[i] );
				currentTrainingClasses.push_back( prevSubClasses[i] );
			}

			std::cout << "currentTrainingExamples " << currentTrainingExamples.size() << "\n";
			// Add the currently used sub regions so that the can be added to the training set on the next iteration.
			prevSubExamples.push_back( randomSubExamples.clone() );
			prevSubClasses.push_back( randomSubClasses.clone() );

			cv::Mat predictions; // Train this bitch and predict.
			learningKernel.train(currentTrainingExamples, cv::ml::ROW_SAMPLE, currentTrainingClasses);
			learningKernel.predict(randomSubExamples, predictions);

			cv::Mat positives = ((randomSubClasses == 1) &= 1);
			cv::Mat negatives = ((randomSubClasses == 0) &= 1);
			uint positiveCount = cv::sum(positives)[0];
			uint negativeCount = cv::sum(negatives)[0];
			std::cout << "validations positives : " << positiveCount << " validation negatives: " << negativeCount << "\n";

			int TP = cv::sum(ImageUtil::trueResults(true, predictions, randomSubClasses))[0];
			int TN = cv::sum(ImageUtil::trueResults(false, predictions, randomSubClasses))[0];

			std::cout << "true positives: " << TP << " true negatives: " << TN << " size: " << predictions.size().height << "\n\n";
			runningTP += TP; runningTN += TN;
		}

		uint totalAccurate = runningTP + runningTN;
		std::cout << "totalTP: " << runningTP << " totalTN: " << runningTN << " totalAccurate: " << totalAccurate << " mean accurate: " << float(totalAccurate) / examples->size().height << "\n\n";

	} else {
		std::cout << "cross validation failed: couldn't resolve data structures.\n";
	}

}

void ViolenceModel::clear()
{
	std::cout << "Clearing the model index store.\n";
	cv::Mat *exampleStore, *classStore;
	std::map<std::string, time_t> *indexCache;

    // Could we just delete the files?
	resolveDataStructures(&exampleStore, &classStore, &indexCache);
	if ( exampleStore && classStore && indexCache) {
		exampleStore->create(0, 0, CV_32F);
		classStore->create(0, 0, CV_32S);
		indexCache->clear();
	}

	persistStore();
}

void ViolenceModel::storeInit(cv::FileStorage &file, std::string exampleStoreName, cv::Mat &exampleStore,
													 std::string classStoreName, cv::Mat &classStore,
													 std::string indexCacheName, std::map<std::string, time_t> &indexCache)
{
	// Read the data structures in from the training store.
	file[exampleStoreName] >> exampleStore;
	std::cout << exampleStoreName << " loaded. size: " << exampleStore.size() << "\n";

	file[classStoreName] >> classStore;
	std::cout << classStoreName << " loaded. size: " << classStore.size() << "\n";

	cv::FileNode indexedFilePaths = file[indexCacheName];
	cv::FileNodeIterator iter = indexedFilePaths.begin(), end = indexedFilePaths.end();
	while ( iter != end )
	{
		std::string path = (*iter)[VIOLENCE_MODEL_TRAINING_EXAMPLE_PATH];
		//std::cout << "found path: " << path << "\n";
		int modTime = (int)(*iter)[VIOLENCE_MODEL_TRAINING_EXAMPLE_MOD_DATE];
		indexCache[path] = (time_t)modTime;
		iter++;
	}

	// Ensure we go no further the height (rows) are not equivalent.
	std::cout << "classes: " << classStore.size().height << " examples: " << exampleStore.size().height << " indices: " << indexCache.size() << "\n";
	assert(classStore.size().height == exampleStore.size().height && classStore.size().height == indexCache.size());
}

void ViolenceModel::index(std::string resourcePath, bool isViolent)
{
	boost::filesystem::path path(resourcePath);
	if ( !isIndexed( path ) ) {

		// Create a VideoCapture instance bound to the path.
		cv::VideoCapture capture(resourcePath);
		cv::Mat trainingSample = extractFeatureVector(capture, resourcePath);
		addSample( path, trainingSample, isViolent);

	} else {
		std::cout << "index -> skipping indexed path: " << resourcePath << "\n";
	}
}

bool ViolenceModel::isIndexed(boost::filesystem::path resourcePath)
{
	std::map<std::string, time_t> *index = NULL;
	resolveDataStructures(NULL, NULL, &index);

	boost::filesystem::path absolutePath = boost::filesystem::absolute(resourcePath);
	std::string indexKey = createIndexKey(absolutePath);

	bool is = index ? index->find( indexKey ) != index->end() : false;
	//std::cout<<"isIndexed: " << is << " path: " << absolutePath.generic_string() << "\n";
	return is;
}

std::string ViolenceModel::createIndexKey(boost::filesystem::path resourcePath)
{
	try {
		resourcePath = boost::filesystem::canonical(resourcePath);
	} catch ( boost::filesystem::filesystem_error &error ) {
		std::cout << "createIndexKey -> assuming path is OpenCV wildcard string.  resourcePath couldn't be canonicalized: " << error.what() <<"\n";
		// Oh well.  Resource path may not exist.
	}

	return resourcePath.generic_string();
}

cv::Mat ViolenceModel::extractFeatureVector(cv::VideoCapture capture, std::string sequenceName, const uint frameCount)
{
	bool capPrevSuccess = false;
	bool capCurrSuccess = false;

	cv::Mat interframeSamples;

	// TODO: Should we need to try to open the capture if it's not?
	timeval begin, end;
	gettimeofday(&begin, NULL);

	// Load the prev frame with the first frame and current with the second
	// so that we can simply loop and compute.
	uint frameIndex = 0;
	cv::Mat currentFrame, prevFrame;

	for ( frameIndex = 0, capPrevSuccess = capture.read(prevFrame), capCurrSuccess = capture.read(currentFrame); // initialize
		  capPrevSuccess && capCurrSuccess && (frameCount == 0 || frameIndex < ( frameCount - 1 ) ); // continue?
		  prevFrame = currentFrame.clone(), capCurrSuccess = capture.read(currentFrame), frameIndex++ ) // update
	{
		// TODO: Eventually we should get this common scaling working.
		cv::Size targetSize(TARGET_COMMON_WIDTH, TARGET_COMMON_HEIGHT);

		// Convert to grayscale.
		if ( frameIndex == 0 )
		{
			// It's only necessary to pre-process the previous frame on the first iteration,
			// as subsequent previous frames will be equal to the current frame in an earlier iteration.
			ViolenceModel::preprocess(prevFrame);
		}

		// Always preprocess the current frame.
		ViolenceModel::preprocess(currentFrame);

		// Compute absolute difference between previous and current frames.
		cv::Mat absDiff, binAbsDiff;
		cv::absdiff(prevFrame, currentFrame, absDiff);

		// Binarize the absolute difference.
		double thresh = 255 * 0.2;
		cv::threshold ( absDiff, binAbsDiff, thresh, 255, cv::THRESH_BINARY );

		// Output binAbsDiff for debug purposes.
		boost::filesystem::path bpath(sequenceName);
		std::stringstream frameName;
		frameName << "bin_abs_diff_" << bpath.stem().string() << "_" << frameIndex;
		//ImageUtil::dumpDebugImage(binAbsDiff, frameName.str());

		// Find the contours (blobs) and use them to compute centroids, area, etc.
		// http://opencv.itseez.com/2.4/doc/tutorials/imgproc/shapedescriptors/moments/moments.html?highlight=moment#code
		std::vector<cv::Vec4i> hierarchy;
		std::vector<std::vector<cv::Point>> contours;
		cv::findContours(binAbsDiff, contours, hierarchy, CV_RETR_LIST, CV_CHAIN_APPROX_NONE);

		// Grab the interframe sample and it add to the running set of interframe samples.
		cv::Mat trainingSample = extractInterframeSampleFromContours(contours, GRACIA_K);
		interframeSamples.push_back(trainingSample);
	}
    
    // We've gathered a matrix of interframe samples each row looks like [k_areas][k_compactness][k!_distances].
    cv::Mat histoFeature = buildHistogramFeature(interframeSamples, 8); // same as GRACIA_K?
    
	gettimeofday(&end, NULL);
	std::cout << "extractFeatureVector takes: " << (end.tv_sec)  - begin.tv_sec << " s\n";

	return histoFeature;
}

cv::Mat ViolenceModel::extractInterframeSampleFromContours(std::vector<std::vector<cv::Point>> contours, uint k)
{
	// Create a max heap for keeping track of the largest blobs.
	std::priority_queue<ImageBlob, std::vector<ImageBlob>, std::greater<ImageBlob>> topBlobsHeap;
	uint blobOrdinal = 0;
	cv::Mat output;

	// 1. Collect the top K largest blobs by area.
	// Each contour is used to create an ImageBlob object that defines its own natural order by area.
	// A heap is used to order them in descending order of area.
	// See http://www.boost.org/doc/libs/1_39_0/libs/bimap/doc/html/boost_bimap/one_minute_tutorial.html
	BOOST_FOREACH(std::vector<cv::Point> cont, contours)
	{
		ImageBlob blob(cont, blobOrdinal++);

		if ( topBlobsHeap.size() < k ) {
			// The heap isn't full yet, we can simply keep adding.
			//std::cout << "adding blob to heap " << blob.ordinal() << " area: " << blob.area() << "\n";
			topBlobsHeap.push(blob);
		} else if ( topBlobsHeap.top() < blob) {
			// The new blob is larger than the largest and the heap is full, so bump out the top one to make room for the talent.
			ImageBlob old = topBlobsHeap.top();
			//std::cout << "popping and overwriting the largest blob" << " old area: " << old.area() <<  " new area: " << blob.area() << "\n";
			topBlobsHeap.pop();
			topBlobsHeap.push(blob);
		}
	}
    
    // 1a. If there weren't k blobs found, create some empty ones to supplement those that are missing.
    // There is probably a more efficient way of handling this.  Consider it in the future.
    for ( size_t i = topBlobsHeap.size(); i < k; i++ )
    {
        std::vector<cv::Point> empty;
        topBlobsHeap.push(empty);
    }
    

	// 2. Now that we've got the k largest blobs,
	if ( topBlobsHeap.size() == k )
	{
		// Read the ordered blobs back as an ordered list.
		std::vector<ImageBlob> blobs;
		while ( !topBlobsHeap.empty() )
		{
			ImageBlob b = topBlobsHeap.top();
			topBlobsHeap.pop();

			//std::cout << "writing blob to vector: ordinal: " << b.ordinal() << " area: " << b.area() << "\n";
			blobs.push_back( b );
		}

		output = buildInterframeSample(blobs);
	}
	else
	{
		// If we weren't able to find enough motion blobs in the sequence, we shouldn't be committing it to the training store.
		// Primarily because if we add too many zeros, it makes the sample ambiguous and contributes to what the learning kernel expects
		// a positive or negative sample to be.  If we don't have enough motion blobs, print an error message and simply continue.
		std::cout << "Sample produced only " << topBlobsHeap.size() << " motion blobs (" << k << " required) and is not being added to the index." << "\n";
	}

	return output;
}

std::vector<cv::Rect> ViolenceModel::preprocess(cv::Mat &frame, cv::Mat *personMask)
{

	std::vector<cv::Rect> rectangles;
	if ( personMask )
	{
		// TODO: Scale to as close to target size as possible, using ImageUtil::scaleImageIntoRect.
		//       After it works of course.
		rectangles = ImageUtil::detectPersonRectangles(frame, personMask);

		// If we've been given an output pointer for the person mask, it must be converted to grayscale,
		// so that it's compatible with the given image after we convert it to grayscale.
		// We intentionally detect person rectangles using the source image, as HOG performance is known
		// to be slightly degraded when computed on grayscale images.
		cv::cvtColor(*personMask, *personMask, CV_RGB2GRAY);
		//ImageUtil::dumpDebugImage(*personMask, "mask");
	}

	cv::Mat grayOut;
	cv::cvtColor(frame, grayOut, CV_RGB2GRAY);
	frame = grayOut;

	return rectangles;
}

void ViolenceModel::train()
{
	// TODO: Pull an equal number of positive and negative examples.  Build a training data matrix on these.
	cv::Mat *trainEx, *trainCl;
	resolveDataStructures(&trainEx, &trainCl, NULL);

	if ( trainEx && trainCl ) {
		learningKernel.train(*trainEx, cv::ml::ROW_SAMPLE, *trainCl);
	} else {
		std::cout << "train -> data structures could not be resolved " << "\n";
	}
}

void ViolenceModel::predict(boost::filesystem::path filePath, float timeInterval)
{
	cv::VideoCapture cap;
	if ( cap.open( filePath.generic_string() ) ) {

		double frameRate = cap.get(CV_CAP_PROP_FPS);
		const uint framesPerExtraction = frameRate * timeInterval;
		int continuousViolentCount = 0;
		float time = 0;

		// Try to predict across 20 frames each time until the capture is empty.
		for ( double totalFrames = cap.get(CV_CAP_PROP_FRAME_COUNT); totalFrames > 0; totalFrames -= framesPerExtraction )
		{
			cv::Mat output;
			cv::Mat featureRowVector = extractFeatureVector(cap, "prediction", framesPerExtraction);
			time += timeInterval;
			learningKernel.predict(featureRowVector, output);

			int isViolent = output.at<int>(0, 0);
			//std::cout << "learning kernel -> predict returns: " << resp << " predicts: " << isViolent << "\n";

			if ( isViolent ) {
				continuousViolentCount++;
			} else {
				continuousViolentCount = 0;
			}

			if ( continuousViolentCount >= 1 ) {
				std::cout << "aggression detected at " << time/60 << "s. \a\n";
			}

		}

	} else {
		std::cout << "predict -> unable to open " << filePath << " for capture. \n";
	}
}


cv::Mat ViolenceModel::buildInterframeSample(std::vector<ImageBlob> blobs)
{
	assert(blobs.size() == GRACIA_K);
	std::vector<cv::Mat> retVect;

	// Build v1 sample based on the given blobs as a vector.
	std::vector<float> v1ExampleVec;
	uint featureCount = 0;

    // 1. Top k areas.
	for ( uint i = 0; i < blobs.size(); i++ ) {
		// Add the area and compactness.
		ImageBlob bi = blobs[i];
		v1ExampleVec.push_back( (float)bi.area() ); featureCount++;
	}
    
    // 2. Top k compactnesses.
    for ( uint i = 0; i < blobs.size(); i++ ) {
        // Add the area and compactness.
        ImageBlob bi = blobs[i];
        v1ExampleVec.push_back( (float)bi.compactness() ); featureCount++;
    }

    // 3. Top k centroid distances.
	for ( uint i = 0; i < blobs.size(); i++ ) {
		ImageBlob bi = blobs[i];
        
        // Compute the distances of this blob from all other blobs.
        for ( uint j = i + 1; j < blobs.size(); j++) {
            ImageBlob bj = blobs[j];
            v1ExampleVec.push_back( bi.distanceFrom( bj ) ); featureCount++;
        }
	}

	// Create a matrix based on this vector so that it can easily be added
	// to store as a row.  A cv::Mat created from an std::vector is effectively
	// a column vector.  We want to eventually store it as a row, so it must
	// also be transposed before we add it to the output std::vector.
	cv::Mat example1Mat(v1ExampleVec);
    example1Mat = example1Mat.t();
    
    // Now that all properties are packed into a cv::Mat, vectorize normalization of each feature.
    uint colBegin = 0, colEnd = (uint)blobs.size();
    cv::Mat k_areas = example1Mat(cv::Range(0, 1), cv::Range(colBegin, colEnd));
    float areaSum = cv::sum(k_areas)[0];
    // std::cout << "sum: " << areaSum << "\nmat: \n" << k_areas << "\n";
    if ( areaSum > 0 ) k_areas /= areaSum;
    
    colBegin = colEnd; colEnd += blobs.size();
    cv::Mat k_compactness = example1Mat(cv::Range(0, 1), cv::Range(colBegin, colEnd));
    float compactnessSum = cv::sum(k_areas)[0];
    if ( compactnessSum > 0 ) k_compactness /= compactnessSum;

    colBegin = colEnd; colEnd = example1Mat.cols;
    cv::Mat distances = example1Mat(cv::Range(0, 1), cv::Range(colBegin, colEnd));
    float sumDistances = cv::sum(distances)[0];
    if ( sumDistances > 0 ) distances /= sumDistances;
    
    //std::cout << "norm: \n" << example1Mat << "\n";
	return example1Mat;
}

cv::Mat ViolenceModel::buildHistogramFeature(cv::Mat interframeSamples, unsigned short binCount)
{
    cv::Mat out;
    
    int channels[] = {0};
    int histSize[] = {binCount};
    
    cv::Mat tempHisto;
    cv::Range allRows = cv::Range::all();
    // From left to right, each (8 uint) histogram intends to convey the likelihood of the (i % 8)'th
    // largest blob's area or compactness or proximity (depending on the value of i) to fall in each bin for the given class.
    // This makes the feature vector easy to reason with using a decision tree, which we ultimately use in LearningKernel.
    for ( uint i = 0; i < interframeSamples.cols; i++ )
    {
        cv::Mat col_i = interframeSamples(allRows, cv::Range(i, i + 1));
        //std::cout << "col_i: " << col_i << "\n";
        
        // Compute the boundaries of the the data, such that we can evenly distribute the energy.
        // Each of the areas is normalized according to the sum of all areas.  This yields the
        // percentage of the motion area that each of the k blobs are.
        double lowBound = 0, highBound = 0;
        cv::minMaxLoc(col_i, &lowBound, &highBound);
        const float valueRange[] = { (float)lowBound, (float)highBound };
        const float* ranges[] = { valueRange };

        cv::calcHist(&col_i,   // Input matrix.
                     1,        // number of source images.                 X
                     channels, // which channels to compute histo of.      X (ours is 1)
                     cv::Mat(),// do not use mask                          X (just a constructor call)
                     tempHisto,// the output histogram                     X (easy)
                     1,        // Surely should be 1 for us                X (pass 1)
                     histSize, // size (bins) of each histo in the output ( in both cases, this should be 8).  X
                     ranges,   // valid ranges to be broken up across bins X {0, 1.0}
                     true,     // the histogram is uniform : toy with this value!
                     false     // never accumulate.
                     );
        //std::cout << "tempHisto: " << tempHisto << "\n";
        out.push_back(tempHisto);
    }

    // Normalize all histograms by the total number of interframes.
    uint interFrameCount = interframeSamples.rows;
    out /= interFrameCount;
    
    return out = out.t();
}

bool ViolenceModel::resolveDataStructures( cv::Mat **exampleStore, cv::Mat **classStore , std::map<std::string, time_t> **indexCache, bool readFileIfEmpty)
{
	// If any of the structures are empty, initialize them.  Note that we will initialize all structures as they should always
	// be in sync with one another.
	// TODO: It would be great to hide all of this functionality in a base class so there is no temptation to grab a hold
	// 		 of any structures without properly pulling from this method.
	if ( readFileIfEmpty && (!trainingExampleStore || !trainingClassStore || !trainingIndexCache) ) {

		trainingExampleStore = new cv::Mat();
		trainingClassStore = new cv::Mat();
		trainingIndexCache = new std::map<std::string, time_t>();

		cv::FileStorage file;
		std::cout << "initializing data structures for target: " << "\n";

		bool trainingStoreOpenSuccess = file.open(trainingStorePath, cv::FileStorage::READ);

		std::string exampleStoreName, classStoreName, indexStoreName;
		exampleStoreName = VIOLENCE_MODEL_TRAINING_SET;
		classStoreName   = VIOLENCE_MODEL_TRAINING_SET_CLASSES;
		indexStoreName   = VIOLENCE_MODEL_TRAINING_FILE_PATHS;

		if (trainingStoreOpenSuccess) {
			storeInit(file, exampleStoreName, *trainingExampleStore,
							classStoreName, *trainingClassStore,
							indexStoreName, *trainingIndexCache);
		} else {
			std::cout << "Failed opening training store at " << trainingStorePath << "\n";
		}
	}

	// Write the addresses if output pointers are given.
	if ( exampleStore ) *exampleStore = trainingExampleStore;
	if ( classStore   ) *classStore   = trainingClassStore;
	if ( indexCache   ) *indexCache   = trainingIndexCache;

	// Consider the function successful if all 3 objects are instantiated.
	return trainingExampleStore && trainingClassStore && trainingIndexCache;
}

void ViolenceModel::addSample( boost::filesystem::path path, cv::Mat sample, bool isViolent)
{
	boost::filesystem::path absolutePath = boost::filesystem::absolute(path);

	if ( !isIndexed( absolutePath ) )
	{

		cv::Mat *exampleStore, *classStore;
		std::map<std::string, time_t> *indexCache;
		resolveDataStructures(&exampleStore, &classStore, &indexCache);

		if ( exampleStore && classStore && indexCache ) {

			std::cout << "training sample at path " << absolutePath << "is not indexed. adding it.\n";

			// OpenCV size is as follows.  [width (columns), height (rows)].
			// We effectively want to resize the matrix according to the
			// width (columns) of the training sample.
			if ( exampleStore->size().width != sample.size().width )
			{
				std::cout << "updating training store size. current: " << exampleStore->size() <<"\n";
				// Create the training store with 0 rows of the training sample's width (column count).
				exampleStore->create(0, sample.size().width, CV_32F);
				classStore->create(0, 1, CV_32S);
				std::cout << "new exampleStore size: " << exampleStore->size() << " classStore size:" << classStore->size() <<"\n";
			}

			// Add it to the training store.
			exampleStore->push_back(sample);

			// Add the class (true or false) to the training class store.
			cv::Mat classMat = (cv::Mat_<int>(1,1) << (int)isViolent);
			classStore->push_back(classMat);
			std::cout<<"exampleStore size after add: " << exampleStore->size() << " classStore size: " << classStore->size() <<"\n";

			// Hash the modification date.
			time_t modDate = boost::filesystem::last_write_time(absolutePath);
			(*indexCache)[ createIndexKey(absolutePath) ] = modDate;
			//std::cout << "path: " << absolutePath.generic_string() << " " << modDate << "\n";
			//persistStore();
		}
	}
}

void ViolenceModel::persistStore()
{
	cv::Mat *trEx, *trCl;
	std::map<std::string, time_t> *trInd;
	resolveDataStructures(&trEx, &trCl, &trInd, false);

	// Persist the training set.
	if ( trEx && trCl && trInd ) {

		cv::FileStorage file;
		// Open the training store file for write and write it.
		bool trainingStoreOpenSuccess = file.open(trainingStorePath, cv::FileStorage::WRITE);
		if (!trainingStoreOpenSuccess) {
			std::cout << "Failed opening training store at " << trainingStorePath << "\n";
			return;
		}

		persistStore(file, VIOLENCE_MODEL_TRAINING_SET,*trEx,
						   VIOLENCE_MODEL_TRAINING_SET_CLASSES, *trCl,
						   VIOLENCE_MODEL_TRAINING_FILE_PATHS, *trInd);
	}

}

void ViolenceModel::persistStore(cv::FileStorage file, std::string exampleStoreName, const cv::Mat &exampleStore,
													   std::string classStoreName,   const cv::Mat &classStore,
													   std::string indexCacheName,   const std::map<std::string, time_t> &indexCache)
{


	std::cout << "persisting " << exampleStoreName << "\n";
	file << exampleStoreName << exampleStore;
	file << classStoreName << classStore;

	file << indexCacheName << "[";
	for ( std::pair<std::string, time_t> item : indexCache )
	{
		// OpenCV FileStorage doesn't seem to allow storage of longs, so we must cast to int.
		// Luckily int time doesn't overflow until 2038, and we're not really using the timestamp right now anyway.
		file << "{" << VIOLENCE_MODEL_TRAINING_EXAMPLE_PATH << item.first << VIOLENCE_MODEL_TRAINING_EXAMPLE_MOD_DATE << (int)item.second << "}";
	}
	file << "]";

}

ViolenceModel::~ViolenceModel() {
	// Be sure to save the training store when the model is destroyed.
	persistStore();
}
