import io
import os
from io import BytesIO

import cv2
import numpy as np
from PIL import Image

from nets.arcface import arcface, ArcMarginProduct
from retinaface import Retinaface
from tools.utils import cv_imread, resize_image, preprocess_input
import base64
retinaface = Retinaface()
if __name__ == '__main__':
    img=cv_imread("../face_image/潘康华_70852096@qq.com_7.jpg")
    # res=np.array(cv2.imencode('.png', img)[1]).tobytes()
    # img=cv2.imdecode(np.array(bytearray(res), dtype='uint8'), cv2.IMREAD_UNCHANGED)
    # image = cv2.imencode('.png', img)[1]
    # image_code = str(base64.b64encode(image))[2:-1]
    # img_data = base64.b64decode(image_code)
    # # 转换为np数组
    # print(img_data)
    # img_array = np.frombuffer(img_data, np.uint8)
    # # 转换成opencv可用格式
    # img1 = cv2.imdecode(img_array, cv2.COLOR_RGB2BGR)
    img,_,_,_=retinaface.detect_image(img)
    print(retinaface.found_face(img))
    print(__file__)
    # #
    # best_img,imgs,old_img,_ =retinaface.detect_image(img)
    # # cv2.imwrite("face_dataset/test.png",imgs[0])
    # cv2.namedWindow("rf_image", 0)
    # cv2.imshow("rf_image", img1)
    # # print(type(imgs[0]))
    # # # # # cv2.namedWindow("old", 0)
    # # # # # cv2.imshow("old", old_img)
    cv2.waitKey(0)

    # #
    # retinaface.encoding_face("image","test")
    # #retinaface.found_face(img)
    # retinaface.addFace(img,"1","test")
    # #print(name)

    #
    # face = np.load("model_data/face_encodings_test.npy", allow_pickle=True)
    # print(face)
    #img=cv_imread("facedatas_img/pkh_12.jpg")
    # imgs,old_img,_ =retinaface.detect_image(img)
    # retinaface.addFace("model_data/204_face_encodings.npy",imgs[0],"test")
    # face = np.load("model_data/204_face_encodings.npy", allow_pickle=True)
    # print(face)


# except Exception as e:
# print(e)
# finally:
# pass