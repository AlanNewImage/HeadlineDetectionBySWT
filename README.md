---
layout: post
title: 使用笔画宽度变化做商品包装的标题检测
categories: 技术博文
author: PangBo
tags: [OCR]
---

> **导言：**  最近在项目有一个需求：对商品包装（书、食品、报纸等等）的标题进行检测定位。包装的标题都是字符，无疑要使用OCR的相关技术。仔细考虑之后，发现标题有两个特点：一是相比其他文本串（比如生产厂商、生产日期等），**标题本身所占面积很大，二来标题中的每一字都比较大，使用的字体都比较粗。**考虑这两点，我们选择了笔划宽度变换（Stroke Width Transform，以下简称SWT）。SWT本身是用来做文本定位的方法，其定位的连续文本区域和中间产品笔划宽度联合起来，正好可以完美地衡量之前所述标题的两个特点。我在Github上找到了一个SWT的开源实现，但针对我们的需求做了不少改进，比如对于中文标题的检测、联通域分析速度慢等。改进之后的算法对于大多数标题都可准确定位，但对已笔划宽度过大的字体标题仍然存在不鲁棒的问题。

<!-- more -->

##SWT简介
SWT[^foot1]是微软提出的在自然场景下检测文本的方法。思想非常简单：绝大多数情况下，连续文本字体相同、大小相同，其笔划（比如中文的一横或者一撇等等）的宽度应该是变化不大的。在笔划宽度图上的联通区域就应该是一个属于一个字。联通部件根据一定的规则筛选之后，满足条件的部件如果属于一个文本行，则它与文本行的方向应该与文本行的走向相同。通过这个条件，将联通部件组合聚成文本行，即完成了文本检测。

[^foot1]: Epshtein B, Ofek E, Wexler Y. Detecting text in natural scenes with stroke width transform. (CVPR2010)

SWT的流程可以参考下图：

<img src="https://farm1.staticflickr.com/361/19798602839_d789cce611_z.jpg" alt="overflow of SWT" />

<p dir="ltr" align="center" style="font-size:xx-small">SWT流程图</p>

网上有一些SWT的开源实现，比如[CCV库](https://github.com/liuliu/ccv)就有SWT的实现。我使用了另外一个实现([链接](https://github.com/aperrau/DetectText))。这个实现对原来的论文做了一些改进，并且在Nokia N900手机上进行了移植和实验。

最主要的改动是：原始论文中计算笔划宽度时，笔划的起点和终点的方向相差不能超过30度，其实现中将其放松到了90度。原因是作者发现30度会导致字体的笔划连接处不连续。我自己实验时也的确发现将90度比30度效果好，中文的转折部分在30度时是断开的，90度可以正确连接，而其他性能也没什么影响。这个实现的另外一些改进也可以参考论文[^foot2]

[^foot2]: Text Detection on Nokia N900 Using Stroke Width Transform

## 针对连通域连接的改进
开源实现中，将物理上连接并且具有相同笔划宽度的部分连接起来组合成连通区域，使用的是boost库中的无向图的连通域分析。作者把每一个像素当成图中的一个节点，将与像素相邻并且笔划宽度差距范围在三分之一以内的节点当成与像素节点连接的边。然后通过boost库中的 connected_components(g, &c[0])获得联通区域。

但自己调试的时候，发现boost库实现的连通域速度特别慢，一幅大概500*300的图，生成连通域都要两三分钟，而且更为诡异的是图的连通域运算不慢，时间全都耗费在了运算完之后的析构函数上。自己在stackoverflow上查别人和发现别人讨论类似的问题，似乎是boost库本身实现的问题。

我尝试了两种不同的方法，two pass的方法和one pass的区域生长，发现one pass的算法比two pass算法快至少一个数量级。boost两分钟的运行时间，使用区域生长只需要不到一百毫秒。

{% highlight c++ %}
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
}
{% endhighlight %}

生成联通区域的结果如下：

<img src="https://farm1.staticflickr.com/307/19797318530_b77d3d396d_z.jpg" alt="501" title="原图" />

<p dir="ltr" align="center" style="font-size:xx-small">原图</p>

<img src="https://farm1.staticflickr.com/459/19990515011_fa4aa39191_z.jpg" alt="components" title="生成的联通区域"/>

<p dir="ltr" align="center" style="font-size:xx-small">生成的连通区域</p>

## 针对中文的改进
传统的SWT是针对英文设计的，英文文本的一个特点是其：绝大多数字母都是连在一起(**i**和**j**除外，但上面的点很小，略去不会对检测有太大影响)，而汉字是由不同的部分组成，部分与部分之间可能不连接。更为复杂的是：汉字的结构决定了不同部分如何分布，左右结构还好说，将一个左右结构的字分成两个字处理，并不会影响文本串的检测，但其他结构的字，其两个部分的相对方向与文本行的生长方向并不一致。比如上下结构的字，其上半部分与下半部分的方向与文本行的方向刚好垂直，这样使用原始算法将部件聚合成文本串，就无法正常工作。

这里采用的方法是**先将不同结构的汉字组合成一个字，再通过字与字之间的关系组合成文本串**。一个汉字无论是何种结构，不同部件之间都是距离还是要远远小于部件与其他汉字部件之间的距离，所以相比英文的检测，中文的检测多了一个步骤：先将联通部件组合成汉字。

具体实现来说，将部件与部件组成的pair，如果pair满足笔划宽度、颜色相似的条件（同原论文，笔画宽度的中值之比小于2，RGB的差值小于40），并且满足物理距离的条件：同一个汉字不同部件的距离，应该不大于两个部件之间长宽的平均值，即：

{% highlight c++ %}
float threshold = (components[i].demension.x+components[i].demension.y+	components[j].demension.x+components[j].demension.y)/4;
// components[i]和components[j]是pair的两个部件
{% endhighlight %}

将pair融合起来组成一个汉字时，从距离最相近的pari开始融合，即满足条件的pair按照距离排序，从小到大开始融合。融合时，将部件中的点组合在一起，并且更新中心位置、笔划宽度、最高最低点、颜色等一系列属性。融合持续到无法找到满足条件的组件才结束。将部件组合成汉字的代码如下：

{% highlight c++ %}
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
{% endhighlight %}

将得到的汉字，再按照经典SWT论文中组合成chain的方法就可以得到图像中的文本串。我们在这里没有做什么修改，**三个汉字及三个汉字以上**，才会被判定为一个文本串。

聚集成汉字的效果如下：

<img src="https://farm1.staticflickr.com/376/19364313553_5276b99184_z.jpg" alt="中文" title="聚集的汉字" />

<p dir="ltr" align="center" style="font-size:xx-middle">聚集成的汉字</p>

## 根据文本检测结果得到标题

得到所有文本串之后，标题文本串一定是其中的一条。在之前的导言里以及提到过：标题有两个特点：一是相比其他文本串（比如生产厂商、生产日期等），**标题本身所占面积很大，二来标题中的每一字都比较大，使用的字体都比较粗，即SWT的值都比较大。**
所以，我们计算每一个文本串(chain)的占据面积和chain上所有点的SWT值，选择两者相加最大的chain作为标题的文本串。因为两个衡量标准的单位都是像素，所以不需要对它们进行归一化，直接求和即可。这部分代码如下：

{% highlight c++ %}

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

{% endhighlight %}

标题检测结果：

<img src="https://farm1.staticflickr.com/294/19797315470_6b87859a83_z.jpg" alt="headline" title="最终检测结果"/>

<p dir="ltr" align="center" style="font-size:xx-small">检测到的标题</p>

##后记##
虽然算法针对中文做了一些优化，但并不影响英文的标题检测效果。比如下图中左边所示的英文报纸，中间一栏显示的是转换中文聚集步骤之后的结果，可以看到其效果是将普通的英文字母转换成了单词，但不影响整个文本行的检测。最右边的图为检测效果，报纸标题被准确地定位出来。

<img src="https://farm1.staticflickr.com/337/19960992926_e5d090a265_z.jpg" alt="英文报纸效果" />

<p dir="ltr" align="middle" style="font-size:xx-small">英文标题检测</p>

算法对于某些笔画宽度变化大的字体没有办法处理，比如下图，因为算法中需要滤去笔划宽度方差大的连通区域。下图为一个失败案例。

<img src="https://farm1.staticflickr.com/295/19960993956_902274ed75_z.jpg" alt="失败的" />

<p dir="ltr" align="middle" style="font-size:xx-small"> 因为文字笔画宽度过大而无法检测 </p>

整个代码已上传到Github，欢迎[下载](https://github.com/ploverpang/DetectText)。


