# angle-measurement-after-object-detection
用于目标检测后，进行单个物体的角度测量
可以接在像yolo这些目标检测程序之后，对bbox中的物体的角度进行一个大概的测量
思路：
1、将原图片分为BRG三个通道
2、将其中两个通道进行相加或者相减，得到灰度图
3、利用otsu阈值分割或者迭代阈值分割，得到二值图
4、进行膨胀处理
5、利用findContours和minAreaRect得到bbox中的最大外接矩形，因为bbox中只有一个物体，而膨胀处理后，图片中还有小黑点，所以选取一个bbox中所有外接矩形中最大的矩形
6、利用minAreaRect.angle函数得到相对角度，再进行转换，得到绝对角度
