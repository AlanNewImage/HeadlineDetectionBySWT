/*
    Copyright 2012 Andrew Perrault and Saurav Kumar.

    This file is part of DetectText.

    DetectText is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    DetectText is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with DetectText.  If not, see <http://www.gnu.org/licenses/>.
*/

#include <cassert>
#include <cmath>
#include <iostream>
#include <opencv/cv.h>
#include <opencv/highgui.h>
#include <opencv/cxcore.h>
#include <math.h>
#include <time.h>
#include <utility>
#include <algorithm>
#include <vector>
#include <TextDetection.h>
#include <queue>

using namespace std;

#define PI 3.14159265

std::vector<std::pair<CvPoint,CvPoint> > findBoundingBoxes( std::vector<std::vector<Point2d> > & components,
                                                           std::vector<Chain> & chains,
                                                           std::vector<std::pair<Point2d,Point2d> > & compBB,
                                                           IplImage * output) {
    std::vector<std::pair<CvPoint,CvPoint> > bb;
    bb.reserve(chains.size());
    for (std::vector<Chain>::iterator chainit = chains.begin(); chainit != chains.end(); chainit++) {
        int minx = output->width;
        int miny = output->height;
        int maxx = 0;
        int maxy = 0;
        for (std::vector<int>::const_iterator cit = chainit->components.begin(); cit != chainit->components.end(); cit++) {
                miny = std::min(miny,compBB[*cit].first.y);
                minx = std::min(minx,compBB[*cit].first.x);
                maxy = std::max(maxy,compBB[*cit].second.y);
                maxx = std::max(maxx,compBB[*cit].second.x);
        }
        CvPoint p0 = cvPoint(minx,miny);
        CvPoint p1 = cvPoint(maxx,maxy);
        std::pair<CvPoint,CvPoint> pair(p0,p1);
        bb.push_back(pair);
    }
    return bb;
}

std::vector<std::pair<CvPoint,CvPoint> > findBoundingBoxes( std::vector<std::vector<Point2d> > & components,
                                                           IplImage * output) {
    std::vector<std::pair<CvPoint,CvPoint> > bb;
    bb.reserve(components.size());
    for (std::vector<std::vector<Point2d> >::iterator compit = components.begin(); compit != components.end(); compit++) {
        int minx = output->width;
        int miny = output->height;
        int maxx = 0;
        int maxy = 0;
        for (std::vector<Point2d>::iterator it = compit->begin(); it != compit->end(); it++) {
                miny = std::min(miny,it->y);
                minx = std::min(minx,it->x);
                maxy = std::max(maxy,it->y);
                maxx = std::max(maxx,it->x);
        }
        CvPoint p0 = cvPoint(minx,miny);
        CvPoint p1 = cvPoint(maxx,maxy);
        std::pair<CvPoint,CvPoint> pair(p0,p1);
        bb.push_back(pair);
    }
    return bb;
}

void normalizeImage (IplImage * input, IplImage * output) {
    assert ( input->depth == IPL_DEPTH_32F );
    assert ( input->nChannels == 1 );
    assert ( output->depth == IPL_DEPTH_32F );
    assert ( output->nChannels == 1 );
    float maxVal = 0;
    float minVal = 1e100;
    for( int row = 0; row < input->height; row++ ){
        const float* ptr = (const float*)(input->imageData + row * input->widthStep);
        for ( int col = 0; col < input->width; col++ ){
            if (*ptr < 0) { }
            else {
                maxVal = std::max(*ptr, maxVal);
                minVal = std::min(*ptr, minVal);
            }
            ptr++;
        }
    }

    float difference = maxVal - minVal;
    for( int row = 0; row < input->height; row++ ){
        const float* ptrin = (const float*)(input->imageData + row * input->widthStep);\
        float* ptrout = (float*)(output->imageData + row * output->widthStep);\
        for ( int col = 0; col < input->width; col++ ){
            if (*ptrin < 0) {
                *ptrout = 1;
            } else {
                *ptrout = ((*ptrin) - minVal)/difference;
            }
            ptrout++;
            ptrin++;
        }
    }
}

void renderComponents (IplImage * SWTImage, std::vector<std::vector<Point2d> > & components, IplImage * output) {
    cvZero(output);
	for (std::vector<std::vector<Point2d> >::iterator it = components.begin(); it != components.end();it++) {
        for (std::vector<Point2d>::iterator pit = it->begin(); pit != it->end(); pit++) {
            CV_IMAGE_ELEM(output, float, pit->y, pit->x) = CV_IMAGE_ELEM(SWTImage, float, pit->y, pit->x);
        }
    }
    for( int row = 0; row < output->height; row++ ){
        float* ptr = (float*)(output->imageData + row * output->widthStep);
        for ( int col = 0; col < output->width; col++ ){
            if (*ptr == 0) {
                *ptr = -1;
            }
            ptr++;
        }
    }
    float maxVal = 0;
    float minVal = 1e100;
    for( int row = 0; row < output->height; row++ ){
        const float* ptr = (const float*)(output->imageData + row * output->widthStep);
        for ( int col = 0; col < output->width; col++ ){
            if (*ptr == 0) { }
            else {
                maxVal = std::max(*ptr, maxVal);
                minVal = std::min(*ptr, minVal);
            }
            ptr++;
        }
    }
    float difference = maxVal - minVal;
    for( int row = 0; row < output->height; row++ ){
        float* ptr = (float*)(output->imageData + row * output->widthStep);\
        for ( int col = 0; col < output->width; col++ ){
            if (*ptr < 1) {
                *ptr = 1;
            } else {
                *ptr = ((*ptr) - minVal)/difference;
            }
            ptr++;
        }
    }

}

void renderComponentsWithBoxes (IplImage * SWTImage, std::vector<std::vector<Point2d> > & components,
                                std::vector<std::pair<Point2d,Point2d> > & compBB, IplImage * output) {
    IplImage * outTemp =
            cvCreateImage ( cvGetSize ( output ), IPL_DEPTH_32F, 1 );

    renderComponents(SWTImage,components,outTemp);
    std::vector<std::pair<CvPoint,CvPoint> > bb;
    bb.reserve(compBB.size());
    for (std::vector<std::pair<Point2d,Point2d> >::iterator it=compBB.begin(); it != compBB.end(); it++ ) {
        CvPoint p0 = cvPoint(it->first.x,it->first.y);
        CvPoint p1 = cvPoint(it->second.x,it->second.y);
        std::pair<CvPoint,CvPoint> pair(p0,p1);
        bb.push_back(pair);
    }

    IplImage * out =
            cvCreateImage ( cvGetSize ( output ), IPL_DEPTH_8U, 1 );
    cvConvertScale(outTemp, out, 255, 0);
    cvCvtColor (out, output, CV_GRAY2RGB);
    //cvReleaseImage ( &outTemp );
    //cvReleaseImage ( &out );

    int count = 0;
    for (std::vector<std::pair<CvPoint,CvPoint> >::iterator it= bb.begin(); it != bb.end(); it++) {
        CvScalar c;
        if (count % 3 == 0) c=cvScalar(255,0,0);
        else if (count % 3 == 1) c=cvScalar(0,255,0);
        else c=cvScalar(0,0,255);
        count++;
        cvRectangle(output,it->first,it->second,c,2);
    }
}

void renderComponentsWithBoxes(IplImage *SWTImage, vector<Component> &components, IplImage* output)
{
	vector<vector<Point2d> > compPoints;
	vector<pair<Point2d, Point2d> > compBB;
	for (int i = 0; i < components.size(); i++)
	{
		compPoints.push_back(components[i].componentPoints);
		compBB.push_back(components[i].compBB);
	}
	renderComponentsWithBoxes (SWTImage, compPoints, compBB, output);
}

void renderChainsWithBoxes (IplImage * SWTImage,
                   std::vector<std::vector<Point2d> > & components,
                   std::vector<Chain> & chains,
                   std::vector<std::pair<Point2d,Point2d> > & compBB,
                   IplImage * output) {
    // keep track of included components
    std::vector<bool> included;
    included.reserve(components.size());
    for (unsigned int i = 0; i != components.size(); i++) {
        included.push_back(false);
    }
    for (std::vector<Chain>::iterator it = chains.begin(); it != chains.end();it++) {
        for (std::vector<int>::iterator cit = it->components.begin(); cit != it->components.end(); cit++) {
            included[*cit] = true;
        }
    }
    std::vector<std::vector<Point2d> > componentsRed;
    for (unsigned int i = 0; i != components.size(); i++ ) {
        if (included[i]) {
            componentsRed.push_back(components[i]);
        }
    }
    IplImage * outTemp =
            cvCreateImage ( cvGetSize ( output ), IPL_DEPTH_32F, 1 );

    std::cout << componentsRed.size() << " components after chaining" << std::endl;
    renderComponents(SWTImage,componentsRed,outTemp);
    std::vector<std::pair<CvPoint,CvPoint> > bb;
    bb = findBoundingBoxes(components, chains, compBB, outTemp);

    IplImage * out =
            cvCreateImage ( cvGetSize ( output ), IPL_DEPTH_8U, 1 );
    cvConvertScale(outTemp, out, 255, 0);
    cvCvtColor (out, output, CV_GRAY2RGB);
    cvReleaseImage ( &out );
    cvReleaseImage ( &outTemp);

    int count = 0;
    for (std::vector<std::pair<CvPoint,CvPoint> >::iterator it= bb.begin(); it != bb.end(); it++) {
        CvScalar c;
        if (count % 3 == 0) c=cvScalar(255,0,0);
        else if (count % 3 == 1) c=cvScalar(0,255,0);
        else c=cvScalar(0,0,255);
        count++;
        cvRectangle(output,it->first,it->second,c,2);
    }
}

void renderChainsWithBoxes(IplImage *SWTImage, vector<Component> &components, vector<Chain> &chains, IplImage *output)
{
	std::vector<std::vector<Point2d> > componentPoints;
	vector<pair<Point2d, Point2d> > compBB;
	for (int i = 0; i < components.size(); i++)
	{
		componentPoints.push_back(components[i].componentPoints);
		compBB.push_back(components[i].compBB);
	}
	renderChainsWithBoxes(SWTImage, componentPoints, chains, compBB, output);
}

void renderChains (IplImage * SWTImage,
                   std::vector<std::vector<Point2d> > & components,
                   std::vector<Chain> & chains,
                   IplImage * output) {
    // keep track of included components
    std::vector<bool> included;
    included.reserve(components.size());
    for (unsigned int i = 0; i != components.size(); i++) {
        included.push_back(false);
    }
    for (std::vector<Chain>::iterator it = chains.begin(); it != chains.end();it++) {
        for (std::vector<int>::iterator cit = it->components.begin(); cit != it->components.end(); cit++) {
            included[*cit] = true;
        }
    }
    std::vector<std::vector<Point2d> > componentsRed;
    for (unsigned int i = 0; i != components.size(); i++ ) {
        if (included[i]) {
            componentsRed.push_back(components[i]);
        }
    }
    std::cout << componentsRed.size() << " components after chaining" << std::endl;
    IplImage * outTemp =
            cvCreateImage ( cvGetSize ( output ), IPL_DEPTH_32F, 1 );
    renderComponents(SWTImage,componentsRed,outTemp);
    cvConvertScale(outTemp, output, 255, 0);
	cvReleaseImage(&outTemp);
}

void renderChains (IplImage * SWTImage,
				   std::vector<Component> &components,
                   std::vector<Chain> & chains,
                   IplImage * output)
{
	std::vector<std::vector<Point2d> > componentPoints;
	for (int i = 0; i < components.size(); i++)
	{
		componentPoints.push_back(components[i].componentPoints);
	}
	renderChains(SWTImage, componentPoints, chains, output);
}

float computeChainsTotalSWT(IplImage* SWTImage, vector<Point2d> &points)
{
	float totalSWT = 0;
	for (int i = 0; i < points.size(); i++)
	{
		totalSWT += CV_IMAGE_ELEM(SWTImage, float, points[i].y, points[i].x);
	}
	return totalSWT;
}

float computeChainsTotalArea(vector<Point2d> &points, cv::RotatedRect &rRect)
{
	vector<cv::Point> pointsForOpenCV(points.size());
	for (int i = 0; i < points.size(); i++)
	{
		pointsForOpenCV[i] = cv::Point(points[i].x, points[i].y);
	}
	rRect = cv::minAreaRect(pointsForOpenCV);
	int width = rRect.size.height;
	int length = rRect.size.width;
	return width*length;
}


cv::RotatedRect findHeadlineLocation(IplImage* SWTImage, 
							 std::vector<Component> &components, 
							 std::vector<Chain> &chains)
{
	int maxScore = 0;
	cv::RotatedRect headLocation;
	for (std::vector<Chain>::iterator it = chains.begin(); it != chains.end();it++) {
		vector<Point2d> points;
        for (std::vector<int>::iterator cit = it->components.begin(); cit != it->components.end(); cit++) {
			points.insert(points.end(), components[*cit].componentPoints.begin(), components[*cit].componentPoints.end());
        }
		float SWTScore = computeChainsTotalSWT(SWTImage, points);
		cv::RotatedRect tmpLoc;
		float AreaScore = computeChainsTotalArea(points, tmpLoc);
		if (SWTScore+AreaScore > maxScore)
		{
			maxScore = SWTScore + AreaScore;
			headLocation = tmpLoc;
		}
    }
	return headLocation;
}

void drawRotatedRect(cv::Mat img, cv::RotatedRect &rRect, cv::Scalar color, int thickness)
{
	assert((rRect.size.height != 0) &&(rRect.size.width != 0));

	cv::Point2f vertices[4];
	rRect.points(vertices);
	for (int i = 0; i < 4; i++)
	{
		line(img, vertices[i], vertices[(i+1)%4], color, thickness);
	}
}

IplImage * textDetection (IplImage * input, bool dark_on_light)
{
    assert ( input->depth == IPL_DEPTH_8U );
    assert ( input->nChannels == 3 );
    std::cout << "Running textDetection with dark_on_light " << dark_on_light << std::endl;
    // Convert to grayscale
    IplImage * grayImage =
            cvCreateImage ( cvGetSize ( input ), IPL_DEPTH_8U, 1 );
    cvCvtColor ( input, grayImage, CV_RGB2GRAY );
    // Create Canny Image
    double threshold_low = 100;
    double threshold_high = 200;

    IplImage * edgeImage =
            cvCreateImage( cvGetSize (input),IPL_DEPTH_8U, 1 );
    cvCanny(grayImage, edgeImage, threshold_low, threshold_high, 3) ;
    cvSaveImage ( "canny.png", edgeImage);

    // Create gradient X, gradient Y
    IplImage * gaussianImage =
            cvCreateImage ( cvGetSize(input), IPL_DEPTH_32F, 1);
    cvConvertScale (grayImage, gaussianImage, 1./255., 0);
    cvSmooth( gaussianImage, gaussianImage, CV_GAUSSIAN, 5, 5);
    IplImage * gradientX =
            cvCreateImage ( cvGetSize ( input ), IPL_DEPTH_32F, 1 );
    IplImage * gradientY =
            cvCreateImage ( cvGetSize ( input ), IPL_DEPTH_32F, 1 );
    cvSobel(gaussianImage, gradientX , 1, 0, CV_SCHARR);
    cvSobel(gaussianImage, gradientY , 0, 1, CV_SCHARR);
    cvSmooth(gradientX, gradientX, 3, 3);
    cvSmooth(gradientY, gradientY, 3, 3);
    cvReleaseImage ( &gaussianImage );

    // Calculate SWT and return ray vectors
    std::vector<Ray> rays;
    IplImage * SWTImage =
            cvCreateImage ( cvGetSize ( input ), IPL_DEPTH_32F, 1 );
    for( int row = 0; row < input->height; row++ ){
        float* ptr = (float*)(SWTImage->imageData + row * SWTImage->widthStep);
        for ( int col = 0; col < input->width; col++ ){
            *ptr++ = -1;
        }
    }
    strokeWidthTransform ( edgeImage, gradientX, gradientY, dark_on_light, SWTImage, rays );
    SWTMedianFilter ( SWTImage, rays );

    IplImage * output2 =
            cvCreateImage ( cvGetSize ( input ), IPL_DEPTH_32F, 1 );
    normalizeImage (SWTImage, output2);
    IplImage * saveSWT =
            cvCreateImage ( cvGetSize ( input ), IPL_DEPTH_8U, 1 );
    cvConvertScale(output2, saveSWT, 255, 0);
    cvSaveImage ( "SWT.png", saveSWT);
    cvReleaseImage ( &output2 );
    cvReleaseImage( &saveSWT );

    // Calculate legally connect components from SWT and gradient image.
    // return type is a vector of vectors, where each outer vector is a component and
    // the inner vector contains the (y,x) of each pixel in that component.
    std::vector<std::vector<Point2d> > componentPoints;

	cv::Mat SWTImageMat(SWTImage);
	findLegallyCC(SWTImageMat, componentPoints);

    // Filter the components
   
    vector<Component> components = filterComponents(SWTImage, componentPoints);
	computeAverageComponentColor(input, components); 
    IplImage * output3 =
            cvCreateImage ( cvGetSize ( input ), 8U, 3 );
	renderComponentsWithBoxes(SWTImage, components, output3);
    cvSaveImage ( "components.png",output3);

	// Cluster Chinese Word
	clusterChineseWord(input, components);
    IplImage * output4 =
            cvCreateImage ( cvGetSize ( input ), 8U, 3 );
	
	renderComponentsWithBoxes (SWTImage, components, output4);
    cvSaveImage ( "words.png",output4);
    cvReleaseImage ( &output4 );

    // Make chains of components
    std::vector<Chain> chains;
	chains = makeChains(components);

    IplImage * output5 =
            cvCreateImage ( cvGetSize ( input ), IPL_DEPTH_8U, 1 );
	renderChains ( SWTImage, components, chains, output5 );
    //cvSaveImage ( "text.png", output4);

    IplImage * output6 =
            cvCreateImage ( cvGetSize ( input ), IPL_DEPTH_8U, 3 );
    cvCvtColor (output5, output6, CV_GRAY2RGB);
    //cvReleaseImage ( &output4 );

    IplImage * output7 =
            cvCreateImage ( cvGetSize ( input ), IPL_DEPTH_8U, 3 );
	renderChainsWithBoxes ( SWTImage, components, chains, output7);
    cvSaveImage ( "chainWithBox.png",output7);

    cv::RotatedRect headLoc = findHeadlineLocation(SWTImage, components, chains);
	cv::Mat output8(input);
	drawRotatedRect(output8, headLoc, cv::Scalar(0,255,0), 4);
	cv::cvtColor(output8, output8, CV_BGR2RGB);
	cv::imwrite("headline.png", output8);

    cvReleaseImage ( &gradientX );
    cvReleaseImage ( &gradientY );
    cvReleaseImage ( &SWTImage );
    cvReleaseImage ( &edgeImage );
    cvReleaseImage ( &grayImage );

    cvReleaseImage ( &output5 );
    cvReleaseImage ( &output6 );
    cvReleaseImage ( &output7 );

    return output6;
}

void strokeWidthTransform (IplImage * edgeImage,
                           IplImage * gradientX,
                           IplImage * gradientY,
                           bool dark_on_light,
                           IplImage * SWTImage,
                           std::vector<Ray> & rays) {
    // First pass
    float prec = .05;
    for( int row = 0; row < edgeImage->height; row++ ){
        const uchar* ptr = (const uchar*)(edgeImage->imageData + row * edgeImage->widthStep);
        for ( int col = 0; col < edgeImage->width; col++ ){
            if (*ptr > 0) {
                Ray r;

                Point2d p;
                p.x = col;
                p.y = row;
                r.p = p;
                std::vector<Point2d> points;
                points.push_back(p);

                float curX = (float)col + 0.5;
                float curY = (float)row + 0.5;
                int curPixX = col;
                int curPixY = row;
                float G_x = CV_IMAGE_ELEM ( gradientX, float, row, col);
                float G_y = CV_IMAGE_ELEM ( gradientY, float, row, col);
                // normalize gradient
                float mag = sqrt( (G_x * G_x) + (G_y * G_y) );
                if (dark_on_light){
                    G_x = -G_x/mag;
                    G_y = -G_y/mag;
                } else {
                    G_x = G_x/mag;
                    G_y = G_y/mag;

                }
                while (true) {
                    curX += G_x*prec;
                    curY += G_y*prec;
                    if ((int)(floor(curX)) != curPixX || (int)(floor(curY)) != curPixY) {
                        curPixX = (int)(floor(curX));
                        curPixY = (int)(floor(curY));
                        // check if pixel is outside boundary of image
                        if (curPixX < 0 || (curPixX >= SWTImage->width) || curPixY < 0 || (curPixY >= SWTImage->height)) {
                            break;
                        }
                        Point2d pnew;
                        pnew.x = curPixX;
                        pnew.y = curPixY;
                        points.push_back(pnew);

                        if (CV_IMAGE_ELEM ( edgeImage, uchar, curPixY, curPixX) > 0) {
                            r.q = pnew;
                            // dot product
                            float G_xt = CV_IMAGE_ELEM(gradientX,float,curPixY,curPixX);
                            float G_yt = CV_IMAGE_ELEM(gradientY,float,curPixY,curPixX);
                            mag = sqrt( (G_xt * G_xt) + (G_yt * G_yt) );
                            if (dark_on_light){
                                G_xt = -G_xt/mag;
                                G_yt = -G_yt/mag;
                            } else {
                                G_xt = G_xt/mag;
                                G_yt = G_yt/mag;

                            }

							// 如果p和q的夹角小于90度
                            if (acos(G_x * -G_xt + G_y * -G_yt) < PI/2.0 ) {
                                float length = sqrt( ((float)r.q.x - (float)r.p.x)*((float)r.q.x - (float)r.p.x) + ((float)r.q.y - (float)r.p.y)*((float)r.q.y - (float)r.p.y));
                                for (std::vector<Point2d>::iterator pit = points.begin(); pit != points.end(); pit++) {
                                    if (CV_IMAGE_ELEM(SWTImage, float, pit->y, pit->x) < 0) {
                                        CV_IMAGE_ELEM(SWTImage, float, pit->y, pit->x) = length;
                                    } else {
                                        CV_IMAGE_ELEM(SWTImage, float, pit->y, pit->x) = std::min(length, CV_IMAGE_ELEM(SWTImage, float, pit->y, pit->x));
                                    }
                                }
                                r.points = points;
                                rays.push_back(r);
                            }
                            break;
                        }
                    }
                }
            }
            ptr++;
        }
    }

}

void SWTMedianFilter (IplImage * SWTImage,
                     std::vector<Ray> & rays) {
    for (std::vector<Ray>::iterator rit = rays.begin(); rit != rays.end(); rit++) {
        for (std::vector<Point2d>::iterator pit = rit->points.begin(); pit != rit->points.end(); pit++) {
            pit->SWT = CV_IMAGE_ELEM(SWTImage, float, pit->y, pit->x);
        }
        std::sort(rit->points.begin(), rit->points.end(), &Point2dSort);
        float median = (rit->points[rit->points.size()/2]).SWT;
        for (std::vector<Point2d>::iterator pit = rit->points.begin(); pit != rit->points.end(); pit++) {
            CV_IMAGE_ELEM(SWTImage, float, pit->y, pit->x) = std::min(pit->SWT, median);
        }
    }

}

bool Point2dSort (const Point2d &lhs, const Point2d &rhs) {
    return lhs.SWT < rhs.SWT;
}

void findLegallyCC(cv::Mat SWTImage, 
				   std::vector<std::vector<Point2d> > &components)
{
	cv::Mat labelMap(SWTImage.size(), CV_32FC1, cv::Scalar(0));
	cv::Point eightNeibor[8]	= {cv::Point(1, -1), 
		cv::Point(1, 0),
		cv::Point(1, 1),
		cv::Point(0, 1),
		cv::Point(-1, 1),
		cv::Point(-1, 0),
		cv::Point(-1, -1),
		cv::Point(0, -1)
	};
	queue<cv::Point> pointQueue;
	
	int label=0;

	for (int i = 0; i < SWTImage.rows; i++)
	{
		for (int j = 0; j < SWTImage.cols; j++)
		{
			if (SWTImage.at<float>(i, j) > 0 && labelMap.at<float>(i, j) == 0)
			{
				pointQueue.push(cv::Point(j, i));	
				label++;
				labelMap.at<float>(i, j) = label;
				components.push_back(vector<Point2d>());
				while (pointQueue.empty() == false)
				{
					cv::Point curPoint = pointQueue.front();	
					float curPointSW = SWTImage.at<float>(curPoint.y, curPoint.x);
					Point2d tmpPoint;
					tmpPoint.x = curPoint.x;
					tmpPoint.y = curPoint.y;
					tmpPoint.SWT= curPointSW;
					components[label-1].push_back(tmpPoint);

					for (int k = 0; k < 8; k++)
					{
						cv::Point neibor = curPoint + eightNeibor[k];
						if (neibor.x>=0 && neibor.x < SWTImage.cols 
							&& neibor.y>=0 && neibor.y<SWTImage.rows )
						{
							float neiborSWT = SWTImage.at<float>(neibor.y, neibor.x);
							if (labelMap.at<float>(neibor.y, neibor.x)==0
								&& neiborSWT > 0 
								&& (neiborSWT/curPointSW <3.0 && curPointSW/neiborSWT <3.0))
							{
								pointQueue.push(neibor);	
								labelMap.at<float>(neibor.y, neibor.x) = label;
							}
						}
					}	
					pointQueue.pop();	
				}
			}	
		}
	}
	int num_vertices=0;
	for (auto vecIt = components.begin(); vecIt != components.end(); vecIt++)
	{
		num_vertices+=vecIt->size();
	}
	std::cout << "Our implementation Before filtering, " << components.size()<< " components and " << num_vertices << " vertices" << std::endl;
}

void componentStats(IplImage * SWTImage,
                                        const std::vector<Point2d> & component,
                                        float & mean, float & variance, float & median,
                                        int & minx, int & miny, int & maxx, int & maxy)
{
        std::vector<float> temp;
        temp.reserve(component.size());
        mean = 0;
        variance = 0;
        minx = 1000000;
        miny = 1000000;
        maxx = 0;
        maxy = 0;
        for (std::vector<Point2d>::const_iterator it = component.begin(); it != component.end(); it++) {
                float t = CV_IMAGE_ELEM(SWTImage, float, it->y, it->x);
                mean += t;
                temp.push_back(t);
                miny = std::min(miny,it->y);
                minx = std::min(minx,it->x);
                maxy = std::max(maxy,it->y);
                maxx = std::max(maxx,it->x);
        }
        mean = mean / ((float)component.size());
        for (std::vector<float>::const_iterator it = temp.begin(); it != temp.end(); it++) {
            variance += (*it - mean) * (*it - mean);
        }
        variance = variance / ((float)component.size());
        std::sort(temp.begin(),temp.end());
        median = temp[temp.size()/2];
}

void filterComponents(IplImage * SWTImage,
                      std::vector<std::vector<Point2d> > & components,
                      std::vector<std::vector<Point2d> > & validComponents,
                      std::vector<Point2dFloat> & compCenters,
                      std::vector<float> & compMedians,
                      std::vector<Point2dFloat> & compDimensions,
                      std::vector<std::pair<Point2d,Point2d> > & compBB )
{
        validComponents.reserve(components.size());
        compCenters.reserve(components.size());
        compMedians.reserve(components.size());
        compDimensions.reserve(components.size());
        // bounding boxes
        compBB.reserve(components.size());
        for (std::vector<std::vector<Point2d> >::iterator it = components.begin(); it != components.end();it++) 
		{
			if (it->size() < 10)
			{
				continue;
			}

            // compute the stroke width mean, variance, median
            float mean, variance, median;
            int minx, miny, maxx, maxy;
            componentStats(SWTImage, (*it), mean, variance, median, minx, miny, maxx, maxy);

            // check if variance is less than half the mean
            if (variance > 1 * mean) {
                 continue;
            }

            float length = (float)(maxx-minx+1);
            float width = (float)(maxy-miny+1);
            // check font height
            if (width > 300 || width < 10) {
                continue;
            }
            float area = length * width;
            float rminx = (float)minx;
            float rmaxx = (float)maxx;
            float rminy = (float)miny;
            float rmaxy = (float)maxy;
            // compute the rotated bounding box

			std::vector<cv::Point> points(it->size());
			for (int i = 0; i < it->size(); i++)
			{
				points[i] = cv::Point((*it)[i].x, (*it)[i].y);	
			}
			cv::RotatedRect bb = cv::minAreaRect(points);
			width = bb.size.height;
			length = bb.size.width;

            // check if the aspect ratio is between 1/5 and 5 
            if (length/width < 1./5. || length/width > 5.) {
                continue;
            }

            Point2dFloat center;
            center.x = ((float)(maxx+minx))/2.0;
            center.y = ((float)(maxy+miny))/2.0;

            Point2dFloat dimensions;
            dimensions.x = maxx - minx + 1;
            dimensions.y = maxy - miny + 1;

            Point2d bb1;
            bb1.x = minx;
            bb1.y = miny;

            Point2d bb2;
            bb2.x = maxx;
            bb2.y = maxy;
            std::pair<Point2d, Point2d> pair(bb1,bb2);

            compBB.push_back(pair);
            compDimensions.push_back(dimensions);
            compMedians.push_back(median);
            compCenters.push_back(center);
            validComponents.push_back(*it);
		}
       std::vector<std::vector<Point2d > > tempComp;
       std::vector<Point2dFloat > tempDim;
       std::vector<float > tempMed;
       std::vector<Point2dFloat > tempCenters;
       std::vector<std::pair<Point2d,Point2d> > tempBB;
       tempComp.reserve(validComponents.size());
       tempCenters.reserve(validComponents.size());
       tempDim.reserve(validComponents.size());
       tempMed.reserve(validComponents.size());
       tempBB.reserve(validComponents.size());
       for (unsigned int i = 0; i < validComponents.size(); i++) {
            int count = 0;
            for (unsigned int j = 0; j < validComponents.size(); j++) {
                if (i != j) {
                    if (compBB[i].first.x <= compCenters[j].x && compBB[i].second.x >= compCenters[j].x &&
                        compBB[i].first.y <= compCenters[j].y && compBB[i].second.y >= compCenters[j].y) {
                        count++;
                    }
                }
            }
            if (count < 2) {
                tempComp.push_back(validComponents[i]);
                tempCenters.push_back(compCenters[i]);
                tempMed.push_back(compMedians[i]);
                tempDim.push_back(compDimensions[i]);
                tempBB.push_back(compBB[i]);
            }
        }
        validComponents = tempComp;
        compDimensions = tempDim;
        compMedians = tempMed;
        compCenters = tempCenters;
        compBB = tempBB;

        compDimensions.reserve(tempComp.size());
        compMedians.reserve(tempComp.size());
        compCenters.reserve(tempComp.size());
        validComponents.reserve(tempComp.size());
        compBB.reserve(tempComp.size());
        std::cout << "After filtering " << validComponents.size() << " components" << std::endl;
}

vector<Component> filterComponents(IplImage *SWTImage, 
	std::vector<std::vector<Point2d> > & componentsPoints)
{
	std::vector<std::vector<Point2d>> validComPoints;
    std::vector<std::pair<Point2d,Point2d> > compBB;
    std::vector<Point2dFloat> compCenters;
    std::vector<float> compMedians;
    std::vector<Point2dFloat> compDimensions;

    filterComponents(SWTImage, componentsPoints, validComPoints, compCenters, compMedians, compDimensions, compBB );
	vector<Component> validComponents(validComPoints.size());
	for (int i = 0; i < validComPoints.size(); i++)
	{
		Component com(validComPoints[i], validComponents.size(), compCenters[i], compMedians[i], compDimensions[i], compBB[i]);
		validComponents[i] = com;
	}
	return validComponents;
}

bool sharesOneEnd( Chain c0, Chain c1) {
    if (c0.p == c1.p || c0.p == c1.q || c0.q == c1.q || c0.q == c1.p) {
        return true;
    }
    else {
        return false;
    }
}

bool chainSortDist (const Chain &lhs, const Chain &rhs) {
    return lhs.dist < rhs.dist;
}

bool chainSortLength (const Chain &lhs, const Chain &rhs) {
    return lhs.components.size() > rhs.components.size();
}


void computeAverageComponentColor(IplImage *colorImage, 
								  vector<Component> &components)
{
    for (std::vector<Component>::iterator it = components.begin(); it != components.end();it++) {
        Point3dFloat mean;
        mean.x = 0;
        mean.y = 0;
        mean.z = 0;
        int num_points = 0;
		for (std::vector<Point2d>::iterator pit = (it->componentPoints).begin(); pit != (it->componentPoints).end(); pit++) {
            mean.x += (float) CV_IMAGE_ELEM (colorImage, unsigned char, pit->y, (pit->x)*3 );
            mean.y += (float) CV_IMAGE_ELEM (colorImage, unsigned char, pit->y, (pit->x)*3+1 );
            mean.z += (float) CV_IMAGE_ELEM (colorImage, unsigned char, pit->y, (pit->x)*3+2 );
            num_points++;
        }
        mean.x = mean.x / ((float)num_points);
        mean.y = mean.y / ((float)num_points);
        mean.z = mean.z / ((float)num_points);
		it->color = mean;
    }
}

vector<Chain> computeSortedChainCandidate(vector<Component> &components)
{
	vector<Chain> chains;
    for ( unsigned int i = 0; i < components.size(); i++ ) {
        for ( unsigned int j = i + 1; j < components.size(); j++ ) {
			// 笔画宽度的中值之比小于2
			if ( (components[i].medianSWT/components[j].medianSWT <= 2.0 && 
				components[j].medianSWT/components[i].medianSWT <= 2.0))
			{
				float dist = (components[i].center.x - components[j].center.x) * (components[i].center.x - components[j].center.x) +
							 (components[i].center.y - components[j].center.y) * (components[i].center.y - components[j].center.y);
				dist = sqrt(dist);
				float colorDist = (components[i].color.x - components[j].color.x) * (components[i].color.x - components[j].color.x) +
									(components[i].color.y - components[j].color.y) * (components[i].color.y - components[j].color.y) +
									(components[i].color.z - components[j].color.z) * (components[i].color.z - components[j].color.z);
				//float threshold = (std::min(components[i].demension.x,components[i].demension.y)+
				//	std::min(components[j].demension.x,components[j].demension.y))/2;
				float threshold = (components[i].demension.x+components[i].demension.y+
					components[j].demension.x+components[j].demension.y)/4;
				if (dist < threshold && colorDist < 1600) {
						Chain c;
						c.p = i;
						c.q = j;
						std::vector<int> comps;
						comps.push_back(c.p);
						comps.push_back(c.q);
						c.components = comps;
						c.dist = dist;
						float d_x = (components[i].center.x - components[j].center.x);
						float d_y = (components[i].center.y - components[j].center.y);
						/*
						float d_x = (compBB[i].first.x - compBB[j].second.x);
						float d_y = (compBB[i].second.y - compBB[j].second.y);
						*/
						float mag = sqrt(d_x*d_x + d_y*d_y);
						d_x = d_x / mag;
						d_y = d_y / mag;
						Point2dFloat dir;
						dir.x = d_x;
						dir.y = d_y;
						c.direction = dir;
						chains.push_back(c);
                }
            }
        }
    }
	std::stable_sort(chains.begin(), chains.end(), &chainSortDist);
	return chains;
}

Component mergeTwoComponent(Component &a, Component &b)
{
	a.componentPoints.insert(a.componentPoints.end(), b.componentPoints.begin(), b.componentPoints.end());
	a.medianSWT = (a.medianSWT*a.componentPointsNum +b.medianSWT*b.componentPointsNum)/(a.componentPointsNum+b.componentPointsNum);

	int minx = 1000000;
	int miny = 1000000;
	int maxx = 0;
	int maxy = 0;
	for (std::vector<Point2d>::const_iterator it = a.componentPoints.begin(); it != a.componentPoints.end(); it++) {
		miny = std::min(miny,it->y);
		minx = std::min(minx,it->x);
		maxy = std::max(maxy,it->y);
		maxx = std::max(maxx,it->x);
	}
	a.demension.x =(float)(maxx - minx + 1);
	a.demension.y =(float)(maxy - miny + 1);

	a.center.x = ((float)(maxx+minx))/2.0;
	a.center.y = ((float)(maxy+miny))/2.0;

	Point2dFloat dimensions;
	dimensions.x = maxx - minx + 1;
	dimensions.y = maxy - miny + 1;

	Point2d bb1;
	bb1.x = minx;
	bb1.y = miny;

	Point2d bb2;
	bb2.x = maxx;
	bb2.y = maxy;

	a.compBB.first.x = minx;
	a.compBB.first.y = miny;
	a.compBB.second.x = maxx;
	a.compBB.second.y = maxy;
	
	a.color.x = (a.color.x*a.componentPointsNum +b.color.x*b.componentPointsNum)/(a.componentPointsNum+b.componentPointsNum);
	a.color.y = (a.color.y*a.componentPointsNum +b.color.y*b.componentPointsNum)/(a.componentPointsNum+b.componentPointsNum);
	a.color.z = (a.color.z*a.componentPointsNum +b.color.z*b.componentPointsNum)/(a.componentPointsNum+b.componentPointsNum);

	a.componentPointsNum += b.componentPointsNum;

	return a;
}

void clusterChineseWord( IplImage * colorImage, vector<Component> &components)
{
	vector<Chain> candidateChains = computeSortedChainCandidate(components);

	while (!candidateChains.empty())
	{
		Chain closestChain = candidateChains[0];
		mergeTwoComponent(components[closestChain.p], components[closestChain.q]);
		components.erase(components.begin()+closestChain.q);
		candidateChains = computeSortedChainCandidate(components);
	}

	std::cout << components.size() << " Chinese word" << std::endl;
}

std::vector<Chain> makeChains(vector<Component> &components) {
	vector<Chain> chains;
	for ( unsigned int i = 0; i < components.size(); i++ ) {
		for ( unsigned int j = i + 1; j < components.size(); j++ ) {
			// 笔画宽度的中值之比小于2, 高度之比小于2
			if ( (components[i].medianSWT/components[j].medianSWT <= 2.0 && 
				components[j].medianSWT/components[i].medianSWT <= 2.0)
				&& (components[i].demension.y/components[j].demension.y <= 2.0 && 
				components[j].demension.y/components[i].demension.y <= 2.0)) 
			{
				float dist = (components[i].center.x - components[j].center.x) * (components[i].center.x - components[j].center.x) +
					(components[i].center.y - components[j].center.y) * (components[i].center.y - components[j].center.y);
				dist = sqrt(dist);
				float colorDist = (components[i].color.x - components[j].color.x) * (components[i].color.x - components[j].color.x) +
					(components[i].color.y - components[j].color.y) * (components[i].color.y - components[j].color.y) +
					(components[i].color.z - components[j].color.z) * (components[i].color.z - components[j].color.z);

				float threshold = (components[i].demension.x+components[i].demension.y+
					components[j].demension.x+components[j].demension.y)/2;
				//if (dist < 3*(float)(std::max(std::min(components[i].demension.x,components[i].demension.y),
				//	std::min(components[j].demension.x,components[j].demension.y)))
				//	&& colorDist < 1600) {
				if (dist < threshold && colorDist < 1600)
				{
						Chain c;
						c.p = i;
						c.q = j;
						std::vector<int> comps;
						comps.push_back(c.p);
						comps.push_back(c.q);
						c.components = comps;
						c.dist = dist;
						float d_x = (components[i].center.x - components[j].center.x);
						float d_y = (components[i].center.y - components[j].center.y);
						/*
						float d_x = (compBB[i].first.x - compBB[j].second.x);
						float d_y = (compBB[i].second.y - compBB[j].second.y);
						*/
						float mag = sqrt(d_x*d_x + d_y*d_y);
						d_x = d_x / mag;
						d_y = d_y / mag;
						Point2dFloat dir;
						dir.x = d_x;
						dir.y = d_y;
						c.direction = dir;
						chains.push_back(c);
				}
			}
		}
	}

    std::cout << chains.size() << " eligible pairs" << std::endl;
    std::sort(chains.begin(), chains.end(), &chainSortDist);

    std::cerr << std::endl;
	const float strictness = PI/6.0;
    //merge chains
    int merges = 1;
    while (merges > 0) {
        for (unsigned int i = 0; i < chains.size(); i++) {
            chains[i].merged = false;
        }
        merges = 0;
        std::vector<Chain> newchains;
        for (unsigned int i = 0; i < chains.size(); i++) {
            for (unsigned int j = 0; j < chains.size(); j++) {
                if (i != j) {
                    if (!chains[i].merged && !chains[j].merged && sharesOneEnd(chains[i],chains[j])) {
                        if (chains[i].p == chains[j].p) {
                            if (acos(chains[i].direction.x * -chains[j].direction.x + chains[i].direction.y * -chains[j].direction.y) < strictness) {
                                  /*      if (chains[i].p == chains[i].q || chains[j].p == chains[j].q) {
                                            std::cout << "CRAZY ERROR" << std::endl;
                                        } else if (chains[i].p == chains[j].p && chains[i].q == chains[j].q) {
                                            std::cout << "CRAZY ERROR" << std::endl;
                                        } else if (chains[i].p == chains[j].q && chains[i].q == chains[j].p) {
                                            std::cout << "CRAZY ERROR" << std::endl;
                                        }
                                        std::cerr << 1 <<std::endl;

                                        std::cerr << chains[i].p << " " << chains[i].q << std::endl;
                                        std::cerr << chains[j].p << " " << chains[j].q << std::endl;
                                std::cerr << compCenters[chains[i].q].x << " " << compCenters[chains[i].q].y << std::endl;
                                std::cerr << compCenters[chains[i].p].x << " " << compCenters[chains[i].p].y << std::endl;
                                std::cerr << compCenters[chains[j].q].x << " " << compCenters[chains[j].q].y << std::endl;
                                std::cerr << std::endl; */

                                chains[i].p = chains[j].q;
                                for (std::vector<int>::iterator it = chains[j].components.begin(); it != chains[j].components.end(); it++) {
                                    chains[i].components.push_back(*it);
                                }
								float d_x = (components[chains[i].p].center.x - components[chains[i].q].center.x);
								float d_y = (components[chains[i].p].center.y - components[chains[i].q].center.y);
                                chains[i].dist = d_x * d_x + d_y * d_y;

                                float mag = sqrt(d_x*d_x + d_y*d_y);
                                d_x = d_x / mag;
                                d_y = d_y / mag;
                                Point2dFloat dir;
                                dir.x = d_x;
                                dir.y = d_y;
                                chains[i].direction = dir;
                                chains[j].merged = true;
                                merges++;
                                /*j=-1;
                                i=0;
                                if (i == chains.size() - 1) i=-1;
                                std::stable_sort(chains.begin(), chains.end(), &chainSortLength);*/
                            }
                        } else if (chains[i].p == chains[j].q) {
                            if (acos(chains[i].direction.x * chains[j].direction.x + chains[i].direction.y * chains[j].direction.y) < strictness) {
/*
                                if (chains[i].p == chains[i].q || chains[j].p == chains[j].q) {
                                    std::cout << "CRAZY ERROR" << std::endl;
                                } else if (chains[i].p == chains[j].p && chains[i].q == chains[j].q) {
                                    std::cout << "CRAZY ERROR" << std::endl;
                                } else if (chains[i].p == chains[j].q && chains[i].q == chains[j].p) {
                                    std::cout << "CRAZY ERROR" << std::endl;
                                }
                                std::cerr << 2 <<std::endl;

                                std::cerr << chains[i].p << " " << chains[i].q << std::endl;
                                std::cerr << chains[j].p << " " << chains[j].q << std::endl;
                                std::cerr << chains[i].direction.x << " " << chains[i].direction.y << std::endl;
                                std::cerr << chains[j].direction.x << " " << chains[j].direction.y << std::endl;
                                std::cerr << compCenters[chains[i].q].x << " " << compCenters[chains[i].q].y << std::endl;
                                std::cerr << compCenters[chains[i].p].x << " " << compCenters[chains[i].p].y << std::endl;
                                std::cerr << compCenters[chains[j].p].x << " " << compCenters[chains[j].p].y << std::endl;
                                std::cerr << std::endl; */

                                chains[i].p = chains[j].p;
                                for (std::vector<int>::iterator it = chains[j].components.begin(); it != chains[j].components.end(); it++) {
                                    chains[i].components.push_back(*it);
                                }
								float d_x = (components[chains[i].p].center.x - components[chains[i].q].center.x);
								float d_y = (components[chains[i].p].center.y - components[chains[i].q].center.y);
                                float mag = sqrt(d_x*d_x + d_y*d_y);
                                chains[i].dist = d_x * d_x + d_y * d_y;

                                d_x = d_x / mag;
                                d_y = d_y / mag;

                                Point2dFloat dir;
                                dir.x = d_x;
                                dir.y = d_y;
                                chains[i].direction = dir;
                                chains[j].merged = true;
                                merges++;
                                /*j=-1;
                                i=0;
                                if (i == chains.size() - 1) i=-1;
                                std::stable_sort(chains.begin(), chains.end(), &chainSortLength); */
                            }
                        } else if (chains[i].q == chains[j].p) {
                            if (acos(chains[i].direction.x * chains[j].direction.x + chains[i].direction.y * chains[j].direction.y) < strictness) {
     /*                           if (chains[i].p == chains[i].q || chains[j].p == chains[j].q) {
                                    std::cout << "CRAZY ERROR" << std::endl;
                                } else if (chains[i].p == chains[j].p && chains[i].q == chains[j].q) {
                                    std::cout << "CRAZY ERROR" << std::endl;
                                } else if (chains[i].p == chains[j].q && chains[i].q == chains[j].p) {
                                    std::cout << "CRAZY ERROR" << std::endl;
                                }
                                std::cerr << 3 <<std::endl;

                                std::cerr << chains[i].p << " " << chains[i].q << std::endl;
                                std::cerr << chains[j].p << " " << chains[j].q << std::endl;

                                std::cerr << compCenters[chains[i].p].x << " " << compCenters[chains[i].p].y << std::endl;
                                std::cerr << compCenters[chains[i].q].x << " " << compCenters[chains[i].q].y << std::endl;
                                std::cerr << compCenters[chains[j].q].x << " " << compCenters[chains[j].q].y << std::endl;
                                std::cerr << std::endl; */
                                chains[i].q = chains[j].q;
                                for (std::vector<int>::iterator it = chains[j].components.begin(); it != chains[j].components.end(); it++) {
                                    chains[i].components.push_back(*it);
                                }
								float d_x = (components[chains[i].p].center.x - components[chains[i].q].center.x);
								float d_y = (components[chains[i].p].center.y - components[chains[i].q].center.y);
                                float mag = sqrt(d_x*d_x + d_y*d_y);
                                chains[i].dist = d_x * d_x + d_y * d_y;


                                d_x = d_x / mag;
                                d_y = d_y / mag;
                                Point2dFloat dir;
                                dir.x = d_x;
                                dir.y = d_y;

                                chains[i].direction = dir;
                                chains[j].merged = true;
                                merges++;
                                /*j=-1;
                                i=0;
                                if (i == chains.size() - 1) i=-1;
                                std::stable_sort(chains.begin(), chains.end(), &chainSortLength); */
                            }
                        } else if (chains[i].q == chains[j].q) {
                            if (acos(chains[i].direction.x * -chains[j].direction.x + chains[i].direction.y * -chains[j].direction.y) < strictness) {
                     /*           if (chains[i].p == chains[i].q || chains[j].p == chains[j].q) {
                                    std::cout << "CRAZY ERROR" << std::endl;
                                } else if (chains[i].p == chains[j].p && chains[i].q == chains[j].q) {
                                    std::cout << "CRAZY ERROR" << std::endl;
                                } else if (chains[i].p == chains[j].q && chains[i].q == chains[j].p) {
                                    std::cout << "CRAZY ERROR" << std::endl;
                                }
                                std::cerr << 4 <<std::endl;
                                std::cerr << chains[i].p << " " << chains[i].q << std::endl;
                                std::cerr << chains[j].p << " " << chains[j].q << std::endl;
                                std::cerr << compCenters[chains[i].p].x << " " << compCenters[chains[i].p].y << std::endl;
                                std::cerr << compCenters[chains[i].q].x << " " << compCenters[chains[i].q].y << std::endl;
                                std::cerr << compCenters[chains[j].p].x << " " << compCenters[chains[j].p].y << std::endl;
                                std::cerr << std::endl; */
                                chains[i].q = chains[j].p;
                                for (std::vector<int>::iterator it = chains[j].components.begin(); it != chains[j].components.end(); it++) {
                                    chains[i].components.push_back(*it);
                                }
								float d_x = (components[chains[i].p].center.x - components[chains[i].q].center.x);
								float d_y = (components[chains[i].p].center.y - components[chains[i].q].center.y);
                                float mag = sqrt(d_x*d_x + d_y*d_y);
                                chains[i].dist = d_x * d_x + d_y * d_y;

                                d_x = d_x / mag;
                                d_y = d_y / mag;
                                Point2dFloat dir;
                                dir.x = d_x;
                                dir.y = d_y;
                                chains[i].direction = dir;
                                chains[j].merged = true;
                                merges++;
                                /*j=-1;
                                i=0;
                                if (i == chains.size() - 1) i=-1;
                                std::stable_sort(chains.begin(), chains.end(), &chainSortLength);*/
                            }
                        }
                    }
                }
            }
        }
        for (unsigned int i = 0; i < chains.size(); i++) {
            if (!chains[i].merged) {
                newchains.push_back(chains[i]);
            }
        }
        chains = newchains;
        std::stable_sort(chains.begin(), chains.end(), &chainSortLength);
    }

    std::vector<Chain> newchains;
    newchains.reserve(chains.size());
    for (std::vector<Chain>::iterator cit = chains.begin(); cit != chains.end(); cit++) {
        if (cit->components.size() >= 3) {
            newchains.push_back(*cit);
        }
    }
    chains = newchains;
    std::cout << chains.size() << " chains after merging" << std::endl;
    return chains;
}
