# retinaface-facenet

## 相关项目

- 前端模块  [meeting-face](https://github.com/saiGou-14H/meeting-face)
- 后端模块  [FaceSystem](https://github.com/saiGou-14H/FaceSystem)

## 项目简介

系统采用静默活体检测，样本数据集采用开源的活体检测数据集CASIA-FASD，人脸检测采用Wider Face数据集进行训练，包含32,203张图片，总共标注393,703个人脸数据。在模型预训练阶段，采用mobilefacenet网络与ArcFace损失函数相结合的方式在数据集CISIA-WebFace（包含上万个人共计约44万张图像）上进行初训练120代，后采用OpenCV图像处理技术对数据集进行清洗再训练30代。人脸识别采用CISIA-WebFace数据集进行训练，包含上10575个人共计约452960张图像。最终模型在LFW数据集上进行测试得到了99.8%的准确率。前后端通过WebSocket协议通信进行数据实时交互，实现了线上线下参会人员管理和人脸识别会议签到。

**数据集**

- 人脸检测数据集 [Wider Face](http://mmlab.ie.cuhk.edu.hk/projects/WIDERFace/WiderFace_Results.html)
- 人脸识别数据集 [CISIA-WebFace](https://pan.baidu.com/s/1SV-4eS74i7oGlpZ1C-ozIw?pwd=8888)

**人脸算法**

- 检测算法 [retinaface](https://github.com/serengil/retinaface)
- 识别算法 [facenet](https://github.com/davidsandberg/facenet)

## 项目实现

人脸图片可通过URL链接或者base64图片编码格式进行传输，将解析出来的人脸图片传进人脸检测模型进行人脸检测，再进人脸对齐截取，之后吧截取后的人脸传入facenet网络进行预测，得到人脸的128位编码再与face_encodings_default.npy中的人脸进行比较并且从中获取人脸用户信息，返回人脸距离最小并且大于识别阈值的人脸用户名字与用户ID。

人脸识别端采用Python编写，使用Flask封装成人脸识别对应功能接口，对请求进行数据处理，然后返回给Java后端。

## 基于OpenCV图像预处理技术

基于深度学习的人脸识别方法很大程度上克服了传统方法对光照以及面部表情敏感的问题，但现实应用场景中由于光线影响造成的面部图像过暗或者面部光照分布不均很大程度上影响了人脸识别的准确率，因此在摄像头采集的人脸图像送入人脸检测网络前需要对图像进行预处理，这样对后续的人脸检测以及识别有很大的帮助。

图像直方图常用来描述图像中的亮度分布情况，其中横坐标表示单通道中的像素值，纵坐标表示图像中对应像素值的个数，直方图均衡化的目的是通过调节图像直方图的均衡分布来增强图像的对比度。 现实图像采集场景中，由于光照的影响，经常会发生直方图不均匀造成的面部区域过亮或过暗的情况，对后期的特征提取以及特征比对造成影响。一般情况直方图针对的是单通道灰度图像，由于本系统通过摄像头网络采集到的图片为RGB三通道图像，因此图像预处理中将对三个通道分别进行直方图均衡化。首先对采集图像的平均亮度进行统计，与设定的阈值进行对比，对未达到阈值的图像进行直方图均衡化处理，最后将处理后的图像送入人脸检测网络。 直方图表示数字图像中每一个像素值出现频率的统计关系，直方图均衡化实质上是对图像像素进行的非线性拉伸，重新分配每个单元的像素值，使的一定灰度范围内每个像素值的数目大致相等。图2-3，图2-4展示了图像直方图均衡化前后的对比，可以明显看出，处理后的面部图像特征更加的清晰。

### 原始图片

![原始图片1](https://github.com/saiGou-14H/save-image/blob/main/%E4%BA%BA%E8%84%B8%E8%AF%86%E5%88%AB%E4%BC%9A%E8%AE%AE%E7%AD%BE%E5%88%B0%E7%B3%BB%E7%BB%9F/%E4%BA%BA%E8%84%B8%E8%AF%86%E5%88%AB%E6%A8%A1%E5%9D%97/%E5%8E%9F%E5%A7%8B%E6%9A%97%E5%9B%BE.png)

![原始图片2](https://github.com/saiGou-14H/save-image/blob/main/%E4%BA%BA%E8%84%B8%E8%AF%86%E5%88%AB%E4%BC%9A%E8%AE%AE%E7%AD%BE%E5%88%B0%E7%B3%BB%E7%BB%9F/%E4%BA%BA%E8%84%B8%E8%AF%86%E5%88%AB%E6%A8%A1%E5%9D%97/%E5%8E%9F%E5%A7%8B%E5%9B%BE%E7%89%87%E8%BF%87%E6%9B%9D.png)

### 直方图均衡化

![直方图均衡化后1](https://github.com/saiGou-14H/save-image/blob/main/%E4%BA%BA%E8%84%B8%E8%AF%86%E5%88%AB%E4%BC%9A%E8%AE%AE%E7%AD%BE%E5%88%B0%E7%B3%BB%E7%BB%9F/%E4%BA%BA%E8%84%B8%E8%AF%86%E5%88%AB%E6%A8%A1%E5%9D%97/%E5%8E%9F%E5%A7%8B%E6%9A%97%E5%9B%BE%E7%9B%B4%E6%96%B9%E5%9B%BE%E5%9D%87%E8%A1%A1%E5%8C%96%E5%90%8E.png)

![直方图均衡化后2](https://github.com/saiGou-14H/save-image/blob/main/%E4%BA%BA%E8%84%B8%E8%AF%86%E5%88%AB%E4%BC%9A%E8%AE%AE%E7%AD%BE%E5%88%B0%E7%B3%BB%E7%BB%9F/%E4%BA%BA%E8%84%B8%E8%AF%86%E5%88%AB%E6%A8%A1%E5%9D%97/%E5%8E%9F%E5%A7%8B%E5%9B%BE%E7%89%87%E8%BF%87%E6%9B%9D%E7%9B%B4%E6%96%B9%E5%9B%BE%E5%9D%87%E8%A1%A1%E5%8C%96%E5%A4%84%E7%90%86%E5%90%8E.png)

