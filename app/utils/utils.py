import json
import time
from io import BytesIO

import cv2
import numpy as np
import base64

import requests
from PIL import Image


def bytes_2cv(im):
    '''二进制图片转cv2
    :param im: 二进制图片数据，bytes
    :return: cv2图像，numpy.ndarray
    '''
    return cv2.imdecode(np.array(bytearray(im), dtype='uint8'), cv2.IMREAD_UNCHANGED)  # 从二进制图片数据中读取


def cv2_bytes(im):
    '''cv2转二进制图片
    :param im: cv2图像，numpy.ndarray
    :return: 二进制图片数据，bytes
    '''
    return np.array(cv2.imencode('.jpg', im)[1]).tobytes()


def image_base64(image_np):
    """
    将np图片(imread后的图片）转码为base64格式
    image_np: cv2图像，numpy.ndarray
    Returns: base64编码后数据
    """
    image = cv2.imencode('.jpg', image_np)[1]
    image_code = str(base64.b64encode(image))[2:-1]
    return image_code


def base64_image(base64_code):
    """
    将base64编码解析成opencv可用图片
    base64_code: base64编码后数据
    Returns: cv2图像，numpy.ndarray
    """
    # base64解码
    base64_code = base64_code.replace('data:image/png;base64,', '')
    base64_code = base64_code.replace('data:image/jpeg;base64,', '')
    base64_code = base64_code.replace(' ', '+')
    base64_code = base64_code.replace(r'\\\\', '+')
    img_data = base64.b64decode(base64_code)
    # 转换为np数组
    img_array = np.fromstring(img_data, np.uint8)
    # 转换成opencv可用格式
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    return img


def cv_imread(file_path):
    cv_img = cv2.imdecode(np.fromfile(file_path, dtype=np.uint8), -1)
    return cv_img
def cv_imwrite(file_path,img):
    cv2.imencode('.jpg', img)[1].tofile(file_path)

#获取请求数据
def getPostData(request):
    request.encoding = 'utf-8'
    if(request.form.to_dict().__contains__("data")):
        return json.loads(request.form.to_dict()["data"])
    return None

def getGetData(request):
    request.encoding = 'utf-8'
    if (request.form.to_dict().__contains__("data")):
        return json.loads(request.form.to_dict()["data"])
    return None

#解析数据
def getImg(data):
    if data["Type"] == "URL":
        url = data["url"]
        #start_time=time.time()
        response = requests.get(url)
        img = Image.open(BytesIO(response.content))
        img = cv2.cvtColor(np.array(img), cv2.COLOR_BGR2RGB)
        #print("url读取图片耗时{:.2f}s".format(time.time() -start_time))
    elif data["Type"] == "BASE64":
        #start_time=time.time()
        img = data["base64"]
        img = base64_image(img)
        img = cv2.cvtColor(np.array(img), cv2.COLOR_BGR2RGB)
        img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        #print("base64读取图片耗时{:.2f}s".format(time.time() -start_time))
    elif data["Type"] == "BYTES":
        img = data["bytes"]
        img = bytes_2cv(img)
        img = cv2.cvtColor(np.array(img), cv2.COLOR_BGR2RGB)
    elif data["Type"] == "BASE64S":
        imgs = []
        for img in data["base64s"]:
            img = base64_image(img)
            img = cv2.cvtColor(np.array(img), cv2.COLOR_BGR2RGB)
            img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            if isDark(cv2.cvtColor(img, cv2.COLOR_RGB2BGR)):
                print("list图片过暗或过亮[yuv直方图均衡化处理]")
                # yuv均衡化处理
                img = yuvequal(img)
            imgs.append(img)
        return imgs
    #图片处理
    if isDark(cv2.cvtColor(img, cv2.COLOR_RGB2BGR)):
        print("图片过暗或过亮[yuv直方图均衡化处理]")
        #yuv均衡化处理
        img = yuvequal(img)
    return img

#字典转json
def dict_json(user):
    user = json.dumps(user,ensure_ascii=False)
    user = user.replace("[", "{")
    user = user.replace("]", "}")
    user = user.replace("\"", "\'")
    return user

#图片均衡化处理
def yuvequal(img):
    imgYUV = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    # cv2.namedWindow('src', cv2.WINDOW_NORMAL)
    # cv2.resizeWindow('src', 500, 500)
    # cv2.imshow("src", img)
    #channelsYUV = cv2.split(imgYUV)
    imgYUV[..., 0] = cv2.equalizeHist(imgYUV[..., 0])
    #channels = cv2.merge(imgYUV)
    result = cv2.cvtColor(imgYUV, cv2.COLOR_YCrCb2BGR)
    # cv2.namedWindow('dst', cv2.WINDOW_NORMAL)
    # cv2.resizeWindow('dst', 500, 500)
    # cv2.imshow("dst", result)
    cv2.waitKey(0)
    return result


#对图片亮度进行判断
def isDark(img):
    # 把图片转换为单通道的灰度图
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 获取形状以及长宽
    img_shape = gray_img.shape
    height, width = img_shape[0], img_shape[1]
    size = gray_img.size
    # 灰度图的直方图
    hist = cv2.calcHist([gray_img], [0], None, [256], [0, 256])

    # 计算灰度图像素点偏离均值(128)程序
    a = 0
    ma = 0
    reduce_matrix = np.full((height, width), 128)
    shift_value = gray_img - reduce_matrix
    shift_sum = sum(map(sum, shift_value))

    da = shift_sum / size

    # 计算偏离128的平均偏差
    for i in range(256):
        ma += (abs(i - 128 - da) * hist[i])
    m = abs(ma / size)
    # 亮度系数
    k = abs(da) / m
    print("亮度值",k)
    if k[0] > 1.3:
        if da > 0:
            # 过亮
            return True;
        else:
            # 过暗
            return True;
    else:
        return False;