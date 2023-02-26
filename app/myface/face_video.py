import time
import cv2
import numpy as np
from keras.models import load_model
from retinaface import Retinaface

# 载入预处理权重
model = load_model("model_data/silent.h5")


def get_score(image):
    image = (cv2.resize(image, (224, 224)) - 127.5) / 127.5
    t = model.predict(np.array([image]))[0]
    return t


if __name__ == "__main__":
    retinaface = Retinaface()
    capture = cv2.VideoCapture(0)
    ref, frame = capture.read()
    if not ref:
        raise ValueError("摄像头无法打开")

    fps = 0.0
    while (True):
        t1 = time.time()
        # 读取某一帧
        ref, frame = capture.read()
        if not ref:
            break
        image = frame
        # 获取图片的长和宽
        img_size = np.asarray(image.shape)[0:2]
        # 进行检测,返回图片和预测结果坐标
        best_img,_,image, results = retinaface.detect_image(image)
        image = np.array(image)
        # 有人脸框才判断是否为真人
        if len(results) > 0:
            for b in results:
                b = list(map(int, b))
                # 人脸框左上角x,y(maximum防止边界情况)
                x1 = int(np.maximum(b[0], 0))
                y1 = int(np.maximum(b[1], 0))
                # 人脸框右上角x,y(minimum防止边界情况)
                x2 = int(np.minimum(b[2], img_size[0] - 1))
                y2 = int(np.minimum(b[3], img_size[1] - 1))
                # 人脸框宽
                w = x2 - x1
                # 人脸框高
                h = y2 - y1
                # 进行放大操作，仅保留人脸部分
                _r = int(max(w, h) * 0.6)
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

                x1 = cx - _r
                y1 = cy - _r

                x1 = int(max(x1, 0))
                y1 = int(max(y1, 0))

                x2 = cx + _r
                y2 = cy + _r

                h, w, c = frame.shape
                x2 = int(min(x2, w - 2))
                y2 = int(min(y2, h - 2))

                _frame = frame[y1:y2, x1:x2]
                scope = get_score(_frame)
                #print(scope)
                if scope > 0.9:
                    fps = (fps + (1. / (time.time() - t1))) / 2
                    print("fps= %.2f" % (fps))
                    image = cv2.putText(image, "fps= %.2f" % (fps), (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0),
                                        2)

        cv2.imshow("video", image)
        c = cv2.waitKey(1) & 0xff

        if c == 27:
            capture.release()
            break
    print("Video Detection Done!")
    capture.release()
    cv2.destroyAllWindows()
