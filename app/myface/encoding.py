import os

import numpy as np

from retinaface import Retinaface

'''
在更换facenet网络后一定要重新进行人脸编码，运行encoding.py。
'''
retinaface = Retinaface()

retinaface.encoding_face("app/face_image")