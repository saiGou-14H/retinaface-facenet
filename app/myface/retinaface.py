import os
from threading import Thread

import cv2
import numpy as np
import tensorflow.keras
from PIL import Image
from matplotlib import pyplot as plt
from tqdm import tqdm
from keras.models import load_model

from app.myface.nets.retinaface import RetinaFace
from keras.applications.imagenet_utils import preprocess_input as keras_preprocess_input
from app.myface.tools.anchors import Anchors
from app.myface.tools.config import cfg_mnet
from app.myface.tools.utils import preprocess_input,BBoxUtility, retinaface_correct_boxes, Alignment, resize_image, cv_imwrite, cv_imread, \
    face_contrast


class Retinaface(object):
    _defaults = {
        "model_path": 'app/myface/model_data/my-best.h5',
        "arcface_model_path":"app/myface/model_data/best.h5",
        "silent_model_path": "app/myface/model_data/silent.h5",
        "backbone": "mobilenet",
        "confidence": 0.5,
        "input_shape": [1280, 1280, 3]
    }

    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]

    # 初始化Retinaface
    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults)
        self.cfg = cfg_mnet
        self.bbox_util = BBoxUtility()
        self.anchors = Anchors(self.cfg, image_size=(self.input_shape[0], self.input_shape[1])).get_anchors()
        self.generate()
        self.silentModel = load_model(self.silent_model_path,compile=False)

    # 载入模型
    def generate(self):
        model_path = os.path.expanduser(self.model_path)
        arcface_model_path =os.path.expanduser(self.arcface_model_path)
        assert model_path.endswith('.h5')
        #载入模型与权值
        self.retinaface = RetinaFace(self.cfg, self.backbone)
        self.retinaface.load_weights(self.model_path, by_name=True)
        self.arcface_model=tensorflow.keras.models.load_model(arcface_model_path,compile=False)

    # 检测图片
    def detect_image(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        old_image = image.copy()
        height, width, _ = np.shape(image)

        scale = [
            np.shape(image)[1], np.shape(image)[0], np.shape(image)[1], np.shape(image)[0]
        ]
        scale_for_landmarks = [
            np.shape(image)[1], np.shape(image)[0], np.shape(image)[1], np.shape(image)[0],
            np.shape(image)[1], np.shape(image)[0], np.shape(image)[1], np.shape(image)[0],
            np.shape(image)[1], np.shape(image)[0]
        ]
        # 增加灰度条(图片在增大或缩小时不失真)
        image =resize_image(image, [self.input_shape[1], self.input_shape[0]])
        # 图片预处理，归一化
        photo = np.expand_dims(keras_preprocess_input(np.array(image, np.float32)), 0)
        # 将图片传入网络当中去预测
        import keras.backend.tensorflow_backend as tb
        tb._SYMBOLIC_SCOPE.value = True

        preds = self.retinaface.predict(photo)
        # 将预测结果进行解码和非极大值抑制
        results = self.bbox_util.detection_out(preds, self.anchors, confidence_threshold=self.confidence)
        if len(results) <= 0:
            return None,None,cv2.cvtColor(old_image, cv2.COLOR_BGR2RGB), results
        results = np.array(results)
        # 去掉灰度条
        results = retinaface_correct_boxes(results, np.array([self.input_shape[0], self.input_shape[1]]),
                                          np.array([height, width]))
        results[:, :4] = results[:, :4] * scale
        results[:, 5:] = results[:, 5:] * scale_for_landmarks

        #人脸特征点
        face_feature= []
        #人脸截图
        face_imgs=[]
        silent_imgs=[]
        biggest_area=0
        for result in results:
            #计算最大人脸框
            left, top, right, bottom = result[0:4]
            w = right - left
            h = bottom - top

            #人脸可信度
            reliability = round(result[4],4)
            result = list(map(int, result))
            result[4]=reliability
            new_image,silent_img=Alignment(old_image,result)

            new_image,silent_img = cv2.cvtColor(new_image, cv2.COLOR_RGB2BGR),cv2.cvtColor(silent_img, cv2.COLOR_RGB2BGR)
            face_imgs.append(new_image)
            silent_imgs.append(silent_img)
            face_feature.append(result)

            #保存最大人脸
            if w * h > biggest_area:
                biggest_area = w * h
                best_face = result
                best_face_img = new_image

        #人脸图片集and人脸特征点
        return best_face_img,face_imgs,cv2.cvtColor(old_image, cv2.COLOR_BGR2RGB),silent_imgs

    def encoding_face(self, path,type="default"):
        list_dir = os.listdir(path)
        image_paths = []
        names = []
        userIds = []
        for name_user in list_dir:
            image_paths.append(path+ "/" + name_user)
            names.append(name_user.split("_")[1])
            userIds.append(name_user.split("_")[0])
        print(image_paths)
        face_encodings = []
        for index, path in enumerate(tqdm(image_paths)):
            #方法一： cv2不支持直接读中文路径，用Image读后转换成cv2
            # image = Image.open(path)
            # image = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)

            #方法二： 实现读取中文路径
            try:
                image = cv_imread(path)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                old_image = image.copy()
                height, width, _ = np.shape(image)
            except Exception as e:
                print(e)
            finally:
                pass

            scale = [
                np.shape(image)[1], np.shape(image)[0], np.shape(image)[1], np.shape(image)[0]
            ]
            scale_for_landmarks = [
                np.shape(image)[1], np.shape(image)[0], np.shape(image)[1], np.shape(image)[0],
                np.shape(image)[1], np.shape(image)[0], np.shape(image)[1], np.shape(image)[0],
                np.shape(image)[1], np.shape(image)[0]
            ]
            # 增加灰度条(图片在增大或缩小时不失真)，图片预处理，归一化
            image =resize_image(image, [self.input_shape[1], self.input_shape[0]])
            photo = np.expand_dims(keras_preprocess_input(np.array(image, np.float32)), 0)
            # 将图片传入网络当中去预测
            import keras.backend.tensorflow_backend as tb
            tb._SYMBOLIC_SCOPE.value = True
            preds = self.retinaface.predict(photo)
            # 将预测结果进行解码和非极大值抑制
            results = self.bbox_util.detection_out(preds, self.anchors, confidence_threshold=self.confidence)
            if len(results) <= 0:
                return old_image
            results = np.array(results)
            # 去掉灰度条
            results = retinaface_correct_boxes(results, np.array([self.input_shape[0], self.input_shape[1]]),
                                              np.array([height, width]))
            results[:, :4] = results[:, :4] * scale
            results[:, 5:] = results[:, 5:] * scale_for_landmarks

            best_face = None
            biggest_area = 0
            # 保留最大人脸的照片and人脸特征
            for result in results:
                left, top, right, bottom = result[0:4]
                w = right - left
                h = bottom - top
                if w * h > biggest_area:
                    biggest_area = w * h
                    best_face = result
            reliability = round(best_face[4], 4)
            best_face = list(map(int, best_face))
            best_face[4] = reliability
            #截取最大人脸
            new_image,silent_img= Alignment(old_image, best_face)
            new_image, silent_img = Alignment(old_image, result)

            new_image, silent_img = cv2.cvtColor(new_image, cv2.COLOR_RGB2BGR), cv2.cvtColor(silent_img,cv2.COLOR_RGB2BGR)

            #写入保存
            #cv_imwrite("facedatas_img/"+names[index]+"_"+str(index)+".jpg",new_image)

            # 1.plt转Image:
            new_image = Image.fromarray(cv2.cvtColor(new_image, cv2.COLOR_BGR2RGB))

            # 2.先保存照片在本地再Image.open读取
            #new_image=Image.open("facedatas_img/"+names[index]+"_"+str(index)+".jpg")

            # 增加灰度条(图片在增大或缩小时不失真)，图片预处理，归一化
            new_image = resize_image(new_image, [160,160],True)

            import keras.backend.tensorflow_backend as tb
            tb._SYMBOLIC_SCOPE.value = True
            photo = np.expand_dims(preprocess_input(np.array(new_image, np.float32)), 0)
            #获得128位人脸编码
            face_encoding = self.arcface_model.predict(photo)
            face_encodings.append([face_encoding,userIds[index],names[index]])
        np.save("app/face_data/face_encodings_{type}.npy".format(type=type),np.array(face_encodings,dtype=object))



    def search_face(self,imgs,type="default"):
        #读取已有人脸
        face = np.load("app/face_data/face_encodings_{type}.npy".format(type=type), allow_pickle=True)
        face_encodings=[]
        users=[]
        if(face==[]):
            return;
        if isinstance(imgs,list):
            for img in imgs:
                # cv2转Image
                if isinstance(img, np.ndarray):
                    img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                new_image = resize_image(img, [160, 160], True)
                photo = np.expand_dims(preprocess_input(np.array(new_image, np.float32)), 0)
                # 获得128位人脸编码
                #flask 防报错
                import keras.backend.tensorflow_backend as tb
                tb._SYMBOLIC_SCOPE.value = True
                face_encoding = self.arcface_model.predict(photo)

                face_encodings.append(face_encoding)
                matches, face_distances = face_contrast(face[:, 0], face_encoding)
                min_index = np.argmin(face_distances)
                min = np.min(face_distances)
                if min < 0.80:
                    auto_replace = Thread(target=self.replace, args=(face_encoding,face[min_index][1],face[min_index][2]))
                    auto_replace.start()
                    user = str(face[min_index][1]) + "':'" + str(face[min_index][2])
                    users.append(user)
            return users
        else:
            new_image=imgs
            #cv2转Image
            if isinstance(new_image,np.ndarray):
                new_image = Image.fromarray(cv2.cvtColor(new_image, cv2.COLOR_BGR2RGB))
            new_image = resize_image(new_image, [160, 160], True)
            photo = np.expand_dims(preprocess_input(np.array(new_image, np.float32)), 0)
            # 获得128位人脸编码
            # flask 防报错
            import keras.backend.tensorflow_backend as tb
            tb._SYMBOLIC_SCOPE.value = True
            face_encoding = self.arcface_model.predict(photo)

            matches, face_distances = face_contrast(face[:, 0], face_encoding)
            min_index = np.argmin(face_distances)
            min=np.min(face_distances)
            #print(face_distances)
            if min<0.80:
                auto_replace = Thread(target=self.replace, args=(face_encoding, face[min_index][1], face[min_index][2]))
                auto_replace.start()
                user = str(face[min_index][1]) + "':'" + str(face[min_index][2])
                users.append(user)
            return users

    def add_face(self,img,userid,name,type="default"):
        # 原来基础上添加128位人脸编码
        face_path="app/face_data/face_encodings_{type}.npy"
        face = np.load(face_path.format(type=type), allow_pickle=True)
        face_encodings = face.tolist()
        # cv2转Image
        if isinstance(img, np.ndarray):
            img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        # plt.subplot(1, 2, 1)
        # plt.imshow(np.array(img))
        # plt.show()
        new_image = resize_image(img, [160, 160], True)
        photo = np.expand_dims(preprocess_input(np.array(new_image, np.float32)), 0)
        # 获得128位人脸编码
        # flask 防报错
        import keras.backend.tensorflow_backend as tb
        tb._SYMBOLIC_SCOPE.value = True
        face_encoding = self.arcface_model.predict(photo)

        new_face = []
        id_faces = []
        for face in face_encodings:
            if (int(face[1])==int(userid)):
                id_faces.append(face)
            else:
                new_face.append(face)
        if (len(id_faces) >= 10):
            matches, face_distances = face_contrast(np.array(id_faces)[:, 0], face_encoding)
            print("该用户已有人脸个数：",len(matches))
            max_index = np.argmax(face_distances)
            max = np.max(face_distances)
            id_faces.pop(max_index)
            id_faces.append([face_encoding, userid, name])
            face_encodings = new_face + id_faces
            np.save(face_path.format(type=type), np.array(face_encodings, dtype=object))
            print("人脸上限执行替换人脸")
            return
        face_encodings.append([face_encoding,userid,name])
        np.save("app/face_data/face_encodings_{type}.npy".format(type=type),np.array(face_encodings, dtype=object))

    def delete_face(self,userid, type="default"):
        delete_path="app/face_data/face_encodings_{type}.npy"
        faces = np.load(delete_path.format(type=type), allow_pickle=True)
        face_encodings = faces.tolist()
        new_faces=[]
        for face in face_encodings:
                if(int(face[1])!=int(userid)):
                    new_faces.append(face)
        np.save(delete_path.format(type=type), np.array(new_faces, dtype=object))

    def replace(self,face_encoding,userid,name,type="default"):
        face_path = "app/face_data/face_encodings_{type}.npy"
        try:
            faces = np.load(face_path.format(type=type), allow_pickle=True)
        finally:
            pass
        face_encodings = faces.tolist()
        id_faces = []
        new_face=[]
        for face in face_encodings:
            if (int(face[1])==int(userid)):
                id_faces.append(face)
            else:
                new_face.append(face)
        matches, face_distances = face_contrast(np.array(id_faces, dtype=object)[:, 0], face_encoding)
        Pass=matches.count(True)
        try:
            if(len(matches)<10 and len(matches)>=3 and (Pass/len(matches)>=0.7)):
                face_encodings.append([face_encoding, userid, name])
                np.save(face_path.format(type=type), np.array(face_encodings, dtype=object))
                print("人脸个数：",len(matches),"达标率:{:.3f}".format(Pass / len(matches)),"动态添加人脸成功")
            elif(Pass/len(matches)>=0.7 and len(matches)>=3):
                max_index = np.argmax(face_distances)
                max = np.max(face_distances)
                id_faces.pop(max_index)
                id_faces.append([face_encoding, userid, name])
                face_encodings=new_face+id_faces
                np.save(face_path.format(type=type), np.array(face_encodings, dtype=object))
                print("达标率:{:.3f}".format(Pass / len(matches)),"动态替换人脸成功")
                return True
            else:
                print("人脸个数：",len(matches),"达标率:{:.3f}".format(Pass / len(matches)), "不动态更新人脸")
        finally:
            pass
        return False



    def save_model(self):
        self.retinaface.save("app/myface/model_data/my-best.h5")

    # 活体检测
    def get_score(self, image):
        image = (cv2.resize(image, (224, 224)) - 127.5) / 127.5
        t = self.silentModel.predict(np.array([image]))[0]
        return t

    def silentFace(self,imgs,silent_imgs):
        # 得分数组
        scopes = []
        new_imgs=[]
        # 有人脸框才判断是否为真人
        if len(silent_imgs) > 0:
            for index in range(len(silent_imgs)):
                scope = self.get_score(silent_imgs[index])
                print("活体检测得分",scope)
                if scope >= 0.3:
                    scopes.append(True)
                    new_imgs.append(imgs[index])
                else:
                    scopes.append(False)
        return scopes, new_imgs



