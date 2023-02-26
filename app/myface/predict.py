import cv2
from retinaface import Retinaface
from tools.utils import cv_imread

retinaface = Retinaface()

while True:
    path = input('请输入图片路径:')

    image = cv_imread(path)
    if image is None:
        print('找不到图片!')
        continue
    else:
        #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        rf_images,old,_ = retinaface.detect_image(image)
        #rf_image = cv2.cvtColor(rf_images, cv2.COLOR_RGB2BGR)
        print(image.shape)
        cv2.namedWindow("rf_image", 0)
        cv2.imshow("rf_image", rf_images[0])
        cv2.namedWindow("old", 0)
        cv2.imshow("old", old)
        cv2.waitKey(0)

