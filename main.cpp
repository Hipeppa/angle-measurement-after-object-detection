#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include <sys/timeb.h>
#include <time.h>

using namespace std;
using namespace cv;
Mat OtsuAlgThreshold(Mat &image)
{
    if (image.channels() != 1)
    {
        cout << "Please input Gray-image!" << endl;
    }
    int T = 0; //Otsu算法阈值
    double varValue = 0; //类间方差中间值保存
    double w0 = 0; //前景像素点数所占比例
    double w1 = 0; //背景像素点数所占比例
    double u0 = 0; //前景平均灰度
    double u1 = 0; //背景平均灰度
    double Histogram[256] = { 0 }; //灰度直方图，下标是灰度值，保存内容是灰度值对应的像素点总数
    uchar *data = image.data;

    double totalNum = image.rows*image.cols; //像素总数

    for (int i = 0; i < image.rows; i++)
    {
        for (int j = 0; j < image.cols; j++)
        {
            if (image.at<uchar>(i, j) != 0) Histogram[data[i*image.step + j]]++;
        }
    }
    int minpos, maxpos;
    for (int i = 0; i < 255; i++)
    {
        if (Histogram[i] != 0)
        {
            minpos = i;
            break;
        }
    }
    for (int i = 255; i > 0; i--)
    {
        if (Histogram[i] != 0)
        {
            maxpos = i;
            break;
        }
    }

    for (int i = minpos; i <= maxpos; i++)
    {
        //每次遍历之前初始化各变量
        w1 = 0;       u1 = 0;       w0 = 0;       u0 = 0;
        //***********背景各分量值计算**************************
        for (int j = 0; j <= i; j++) //背景部分各值计算
        {
            w1 += Histogram[j];   //背景部分像素点总数
            u1 += j*Histogram[j]; //背景部分像素总灰度和
        }
        if (w1 == 0) //背景部分像素点数为0时退出
        {
            break;
        }
        u1 = u1 / w1; //背景像素平均灰度
        w1 = w1 / totalNum; // 背景部分像素点数所占比例
        //***********背景各分量值计算**************************

        //***********前景各分量值计算**************************
        for (int k = i + 1; k < 255; k++)
        {
            w0 += Histogram[k];  //前景部分像素点总数
            u0 += k*Histogram[k]; //前景部分像素总灰度和
        }
        if (w0 == 0) //前景部分像素点数为0时退出
        {
            break;
        }
        u0 = u0 / w0; //前景像素平均灰度
        w0 = w0 / totalNum; // 前景部分像素点数所占比例
        //***********前景各分量值计算**************************

        //***********类间方差计算******************************
        double varValueI = w0*w1*(u1 - u0)*(u1 - u0); //当前类间方差计算
        if (varValue < varValueI)
        {
            varValue = varValueI;
            T = i;
        }
    }
    Mat dst;
    threshold(image, dst, T, 255, CV_THRESH_OTSU);
    return dst;
}
//迭代阈值分割

Mat IterationThreshold(Mat src)
{
    int width = src.cols;
    int height = src.rows;
    int hisData[256] = { 0 };
    for (int j = 0; j < height; j++)
    {
        uchar* data = src.ptr<uchar>(j);
        for (int i = 0; i < width; i++)
            hisData[data[i]]++;
    }

    int T0 = 0;
    for (int i = 0; i < 256; i++)
    {
        T0 += i*hisData[i];
    }
    T0 /= width*height;

    int T1 = 0, T2 = 0;
    int num1 = 0, num2 = 0;
    int T = 0;
    while (1)
    {
        for (int i = 0; i < T0 + 1; i++)
        {
            T1 += i*hisData[i];
            num1 += hisData[i];
        }
        if (num1 == 0)
            continue;
        for (int i = T0 + 1; i < 256; i++)
        {
            T2 += i*hisData[i];
            num2 += hisData[i];
        }
        if (num2 == 0)
            continue;

        T = (T1 / num1 + T2 / num2) / 2;

        if (T == T0)
            break;
        else
            T0 = T;
    }

    Mat dst;
    threshold(src, dst, T, 255, 0);
    return dst;
}
//将相对角度转化为绝对角度
static double calcLineDegree(const Point2f& firstPt, const Point2f& secondPt)
{
    double curLineAngle = 0.0f;
    if (secondPt.x - firstPt.x != 0)
    {
        curLineAngle = atan(static_cast<double>(firstPt.y - secondPt.y) / static_cast<double>(secondPt.x - firstPt.x));
        if (curLineAngle < 0)
        {
            curLineAngle += CV_PI;
        }
    }
    else
    {
        curLineAngle = CV_PI / 2.0f; //90度
    }
    return curLineAngle*180.0f/CV_PI;
}
static double getRcDegree(const RotatedRect box)
{
    double degree = 0.0f;
    Point2f vertVect[4];
    box.points(vertVect);
    //line 1
    const double firstLineLen = (vertVect[1].x - vertVect[0].x)*(vertVect[1].x - vertVect[0].x) +
                                (vertVect[1].y - vertVect[0].y)*(vertVect[1].y - vertVect[0].y);
    //line 2
    const double secondLineLen = (vertVect[2].x - vertVect[1].x)*(vertVect[2].x - vertVect[1].x) +
                                 (vertVect[2].y - vertVect[1].y)*(vertVect[2].y - vertVect[1].y);
    if (firstLineLen > secondLineLen)
    {
        degree = calcLineDegree(vertVect[0], vertVect[1]);
    }
    else
    {
        degree = calcLineDegree(vertVect[2], vertVect[1]);
    }
    return degree;
}

int main()
{
    //A是目标检测后bbox的xmin，ymin，xmax，ymax，最后一个参数是图片名称
    vector<int> A{351,84,571,209,416};
    string file1="D:\\learn\\opencv_set\\"+to_string(A[4])+".jpg";
    Mat src1 = imread(file1,1);
    imshow("原图",src1);
    //从模拟检测后的产生的bbox区域
    Mat src = src1(Rect(A[0], A[1]-10, A[2]-A[0], A[3]-A[1]+20));
    imshow("bbox",src);
    //将原图分成BGR三个通道，其中mv[0]是B通道，mv[1]是R通道，mv[2]是G通道
    //将BR通道加权求和后灰度化，使用otsu阈值分割
    struct timeb startTime, endTime;
    ftime(&startTime);
    Mat mv[3];
    split(src,mv);
    Mat gray;
    //如果二值图效果不好，可以通过改这里参数，尝试R和B的组合或者B和G的组合
    //这里是4*R-1*B，
    addWeighted(mv[1],4,mv[0],-1,0,gray);
    Mat img;
    //使用otsu阈值分割，二值图
    img = OtsuAlgThreshold(gray);
    //膨胀
    Mat morphologyDst;
    cv::morphologyEx(img, morphologyDst, cv::MORPH_DILATE,
                     cv::getStructuringElement(cv::MORPH_RECT, cv::Size(7, 1)));
    //检测可能出现的轮廓
    //找出最小外接矩形，同时选出最大的外接矩形（排除小黑点的轮廓导致的外接矩形）
    vector<vector<Point>> contours;
    vector<Vec4i> hierarchy;
    findContours(morphologyDst, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0) );
    vector<RotatedRect> minRect(contours.size());
    float area=0;
    for( int i = 0; i < contours.size(); i++ )
    {
        minRect[i] = minAreaRect( Mat(contours[i]) );
        area=max(area,minRect[i].size.area());
    }

    // 绘出轮廓并测出角度
    Mat drawing=src;
    for( int i = 0; i< contours.size(); i++ )
    {
        if(minRect[i].size.area()==area){
            Point2f rect_points[4];
            minRect[i].points(rect_points);
            //寻找长
            for( int j = 0; j < 4; j++ )line( drawing, rect_points[j], rect_points[(j+1)%4], Scalar(0,255,0), 2, 8 );
            double degree1=getRcDegree(minRect[i]);
            cout<<"绝对角度"<<degree1<<endl;
            cout << "X:" << minRect[i].center.x << "Y:" << minRect[i].center.y <<endl<< "Angle:" << minRect[i].angle << endl;
        }
    }

    ftime(&endTime);
    cout<<"time:"<<(endTime.time - startTime.time) * 1000 + (endTime.millitm - startTime.millitm)<<"ms"<<endl;
    imshow("灰度图",gray);
    imshow("二值图", img);
    imshow("膨胀", morphologyDst);
    imshow( "角度框", drawing );
    waitKey(0);

}
