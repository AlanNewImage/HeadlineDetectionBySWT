
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
#include <fstream>
#include "TextDetection.h"
#include <opencv/highgui.h>
#include <exception>
#include <map>
#include <set>
#include <queue>

using namespace std;

void convertToFloatImage ( IplImage * byteImage, IplImage * floatImage )
{
  cvConvertScale ( byteImage, floatImage, 1 / 255., 0 );
}

class FeatureError : public std::exception
{
std::string message;
public:
FeatureError ( const std::string & msg, const std::string & file )
{
  std::stringstream ss;

  ss << msg << " " << file;
  message = msg.c_str ();
}
~FeatureError () throw ( )
{
}
};

IplImage * loadByteImage ( const char * name )
{
  IplImage * image = cvLoadImage ( name );

  if ( !image )
  {
    return 0;
  }
  cvCvtColor ( image, image, CV_BGR2RGB );
  return image;
}

IplImage * loadFloatImage ( const char * name )
{
  IplImage * image = cvLoadImage ( name );

  if ( !image )
  {
    return 0;
  }
  cvCvtColor ( image, image, CV_BGR2RGB );
  IplImage * floatingImage = cvCreateImage ( cvGetSize ( image ),
                                             IPL_DEPTH_32F, 3 );
  cvConvertScale ( image, floatingImage, 1 / 255., 0 );
  cvReleaseImage ( &image );
  return floatingImage;
}

int mainTextDetection ( int argc, char * * argv )
{
  IplImage * byteQueryImage = loadByteImage ( argv[1] );
  if ( !byteQueryImage )
  {
    printf ( "couldn't load query image\n" );
    return -1;
  }

  // Detect text in the image
  IplImage * output = textDetection ( byteQueryImage, atoi(argv[3]) );
  cvReleaseImage ( &byteQueryImage );
  cvSaveImage ( argv[2], output );
  cvReleaseImage ( &output );
  return 0;
}


int find_root(vector<int> patents, int index)
{
	int x = index;
	while (patents[x] != 0)
	{
		x = patents[x];
	}
	return x;
}

void unionTwoSet(vector<int> &patents, int first, int second)
{
	int root1 = find_root(patents, first);
	int root2 = find_root(patents, second);
	if (root1 != root2)
	{
		patents[max(root1, root2)] = min(root1, root2);
	}
}

void CCATest(cv::Mat image)
{
	int labelNum = 0;
	std::set<std::pair<int, int> > labelPair;
	std::vector<int> labels;
	labels.push_back(0);

	cv::Mat labelImage(image.size(), CV_8U, cv::Scalar::all(0));

	//First pass
	for (int i = 0; i < image.rows; i++)
	{
		uchar *p = (uchar*)image.ptr<uchar>(i);
		for (int j = 0; j < image.cols; j++)
		{
			if (*p == 0)
			{
				if (i >= 1 && j >= 1)
				{
					uchar left = labelImage.at<uchar>(i-1, j);
					uchar up = labelImage.at<uchar>(i, j-1);
					if (left != 0 && up != 0)
					{
						labelImage.at<uchar>(i, j) = std::min(left, up);
						if (left != up)
						{
							labelPair.insert(std::make_pair(std::max(left, up), std::min(left, up)));
						}
					}
					else if (left != 0)
					{
						labelImage.at<uchar>(i, j) = left;
					}
					else if (up != 0)
					{
						labelImage.at<uchar>(i, j) = up;
					}
					else
					{
						labelNum++;
						labelImage.at<uchar>(i, j) = labelNum;
						labels.push_back(labelNum);
					}
				}
			}
			p++;
		}
	}

	std::vector<int> patents(labels.size());
	for (auto mit = labelPair.begin(); mit!=labelPair.end(); mit++)
	{
		unionTwoSet(patents, mit->first, mit->second);
	}

	//Secend pass
	for (int i = 0; i < labelImage.rows; i++)
	{
		uchar *p = (uchar*)labelImage.ptr<uchar>(i);
		for (int j = 0; j < labelImage.cols; j++)
		{
			if (*p != 0)
			{
				labelImage.at<uchar>(i, j) = find_root(patents, labelImage.at<uchar>(i, j));
			}
			p++;
		}
	}
/*
	//Render
	std::vector<cv::Scalar> colors(labels.size());
	for (int i = 1; i < colors.size(); i++)
	{
		int root = find_root(patents, i);
		if (root%3 == 1)
		{
			colors[i] = cv::Scalar(0,0,255);
		}		
		else if (root%3 == 2)
		{
			colors[i] = cv::Scalar(0,255,0);
		}		
		else
		{
			colors[i] = cv::Scalar(255,0,0);
		}
	}

	cv::Mat renderImage(image.size(), CV_8UC3, cv::Scalar(0,0,0));
	for (int i = 0; i < renderImage.rows; i++)
	{
		for (int j = 0; j < renderImage.cols; j++)
		{
			uchar label = labelImage.at<uchar>(i, j);
			if (label != 0)
			{
				renderImage.at<cv::Vec3b>(i, j)[0] = colors[label][0];
				renderImage.at<cv::Vec3b>(i, j)[1] = colors[label][1];
				renderImage.at<cv::Vec3b>(i, j)[2] = colors[label][2];
			}
		}
	}

	cv::imshow("CCA result", renderImage);
	cv::waitKey();
	*/
}


void regionGrowing(cv::Mat image)
{
	cv::Mat labelMap(image.size(), CV_8UC1, cv::Scalar(0));
	cv::Point eightL[8]	= {cv::Point(1, -1), 
		cv::Point(1, 0),
		cv::Point(1, 1),
		cv::Point(0, 1),
		cv::Point(-1, 1),
		cv::Point(-1, 0),
		cv::Point(-1, -1),
		cv::Point(0, -1)
	};
	std::vector<cv::Point> eightNeibor(eightL, eightL+8);	
	queue<cv::Point> pointQueue;
	
	int label=0;
	for (int i = 0; i < image.rows; i++)
	{
		for (int j = 0; j < image.cols; j++)
		{
			if (image.at<uchar>(i, j) == 0 && labelMap.at<uchar>(i, j) == 0)
			{
				pointQueue.push(cv::Point(j, i));	
				label++;
				labelMap.at<uchar>(i, j) = label;
				while (pointQueue.empty() == false)
				{
					cv::Point curPoint = pointQueue.front();	
					for (int k = 0; k < 8; k++)
					{
						cv::Point neibor = curPoint + eightNeibor[k];
						if (neibor.x>=0 && neibor.x < image.cols && neibor.y>=0 && neibor.y<image.rows 
							&& labelMap.at<uchar>(neibor.y, neibor.x)==0
							&& image.at<uchar>(neibor.y, neibor.x) ==0)
						{
							pointQueue.push(neibor);	
							labelMap.at<uchar>(neibor.y, neibor.x) = label;
						}
					}	
					pointQueue.pop();	
				}
			}	
		}
	}
	/*
	//Render
	std::vector<cv::Scalar> colors(label+1);
	for (int i = 1; i < colors.size(); i++)
	{
		if (i%3 == 1)
		{
			colors[i] = cv::Scalar(0,0,255);
		}		
		else if (i%3 == 2)
		{
			colors[i] = cv::Scalar(0,255,0);
		}		
		else
		{
			colors[i] = cv::Scalar(255,0,0);
		}
	}

	cv::Mat renderImage(image.size(), CV_8UC3, cv::Scalar(0,0,0));
	for (int i = 0; i < renderImage.rows; i++)
	{
		for (int j = 0; j < renderImage.cols; j++)
		{
			uchar label = labelMap.at<uchar>(i, j);
			if (label != 0)
			{
				renderImage.at<cv::Vec3b>(i, j)[0] = colors[label][0];
				renderImage.at<cv::Vec3b>(i, j)[1] = colors[label][1];
				renderImage.at<cv::Vec3b>(i, j)[2] = colors[label][2];
			}
		}
	}

	cv::imshow("Region Growing result", renderImage);
	cv::waitKey();
	*/
}



int main ( int argc, char * * argv )
{	
	cv::Mat image = cv::imread("E:\\code\\DetectText\\data\\icvpr.jpg", cv::IMREAD_GRAYSCALE);
	cv::Mat binaryImage;
	cv::threshold(image, binaryImage, 20, 255, cv::THRESH_BINARY); 
	cv::dilate(binaryImage, binaryImage, cv::Mat());

	double start = cv::getTickCount();
	CCATest(binaryImage);
	double during = (cv::getTickCount()-start)/cv::getTickFrequency();
	cout << "Two pass during is " << during << endl;

	start = cv::getTickCount();
	regionGrowing(binaryImage);
	during = (cv::getTickCount()-start)/cv::getTickFrequency();
	cout << "Region growing during is " << during << endl;

  if ( ( argc != 4 ) )
  {
    printf ( "usage: %s imagefile resultImage darkText\n",
             argv[0] );

    return -1;
  }
  mainTextDetection ( argc, argv );
  system("PAUSE");
  return 0;
}
