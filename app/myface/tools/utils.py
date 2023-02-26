from functools import reduce
import numpy as np
import keras
import math
import cv2
from PIL import Image
from matplotlib import pyplot as plt


def compose(*funcs):
    if funcs:
        return reduce(lambda f, g: lambda *a, **kw: g(f(*a, **kw)), funcs)
    else:
        raise ValueError('不支持空序列的组成')


# 对输入的图像进行resize操作
def resize_image(image, size,*type):
    if not type:
        ih, iw, _ = np.shape(image)
        w, h = size
        scale = min(w / iw, h / ih)
        nw = int(iw * scale)
        nh = int(ih * scale)

        image = cv2.resize(image, (nw, nh))
        new_image = np.ones([size[1], size[0], 3]) * 128
        new_image[(h - nh) // 2:nh + (h - nh) // 2, (w - nw) // 2:nw + (w - nw) // 2] = image
    else:
        iw, ih = image.size
        w, h = size
        if type:
            scale = min(w / iw, h / ih)
            nw = int(iw * scale)
            nh = int(ih * scale)

            image = image.resize((nw, nh), Image.BICUBIC)
            new_image = Image.new('RGB', size, (128, 128, 128))
            new_image.paste(image, ((w - nw) // 2, (h - nh) // 2))
        else:
            new_image = image.resize((w, h), Image.BICUBIC)
    return new_image



# 将输出调整为相对于原图的大小
def retinaface_correct_boxes(result, input_shape, image_shape):
    new_shape = image_shape * np.min(input_shape / image_shape)

    offset = (input_shape - new_shape) / 2. / input_shape
    scale = input_shape / new_shape

    scale_for_boxs = [scale[1], scale[0], scale[1], scale[0]]
    scale_for_landmarks = [scale[1], scale[0], scale[1], scale[0], scale[1], scale[0], scale[1], scale[0], scale[1],
                           scale[0]]

    offset_for_boxs = [offset[1], offset[0], offset[1], offset[0]]
    offset_for_landmarks = [offset[1], offset[0], offset[1], offset[0], offset[1], offset[0], offset[1], offset[0],
                            offset[1], offset[0]]

    result[:, :4] = (result[:, :4] - np.array(offset_for_boxs)) * np.array(scale_for_boxs)
    result[:, 5:] = (result[:, 5:] - np.array(offset_for_landmarks)) * np.array(scale_for_landmarks)

    return result


class PriorProbability(keras.initializers.Initializer):
    def __init__(self, probability=0.01):
        self.probability = probability

    def get_config(self):
        return {
            'probability': self.probability
        }

    def __call__(self, shape, dtype=None):
        result = np.ones(shape, dtype=dtype) * -math.log((1 - self.probability) / self.probability)

        return result


class BBoxUtility(object):
    def __init__(self, priors=None, overlap_threshold=0.35,
                 nms_thresh=0.45):
        self.priors = priors
        self.num_priors = 0 if priors is None else len(priors)
        self.overlap_threshold = overlap_threshold
        self._nms_thresh = nms_thresh

    def iou(self, box):
        # 计算出每个真实框与所有的先验框的iou
        # 判断真实框与先验框的重合情况
        inter_upleft = np.maximum(self.priors[:, :2], box[:2])
        inter_botright = np.minimum(self.priors[:, 2:4], box[2:])

        inter_wh = inter_botright - inter_upleft
        inter_wh = np.maximum(inter_wh, 0)
        inter = inter_wh[:, 0] * inter_wh[:, 1]
        # 真实框的面积
        area_true = (box[2] - box[0]) * (box[3] - box[1])
        # 先验框的面积
        area_gt = (self.priors[:, 2] - self.priors[:, 0]) * (self.priors[:, 3] - self.priors[:, 1])
        # 计算iou
        union = area_true + area_gt - inter

        iou = inter / union
        return iou

    def encode_box(self, box, return_iou=True):
        iou = self.iou(box[:4])

        encoded_box = np.zeros((self.num_priors, 4 + return_iou + 10))

        # 找到每一个真实框，重合程度较高的先验框
        assign_mask = iou > self.overlap_threshold
        if not assign_mask.any():
            assign_mask[iou.argmax()] = True
        if return_iou:
            encoded_box[:, 4][assign_mask] = iou[assign_mask]

        # 找到对应的先验框
        assigned_priors = self.priors[assign_mask]

        # 先计算真实框的中心与长宽
        box_center = 0.5 * (box[:2] + box[2:4])
        box_wh = box[2:4] - box[:2]
        # 再计算重合度较高的先验框的中心与长宽
        assigned_priors_center = 0.5 * (assigned_priors[:, :2] +
                                        assigned_priors[:, 2:4])
        assigned_priors_wh = (assigned_priors[:, 2:4] -
                              assigned_priors[:, :2])

        # 逆向求取efficientdet应该有的预测结果
        encoded_box[:, :2][assign_mask] = box_center - assigned_priors_center
        encoded_box[:, :2][assign_mask] /= assigned_priors_wh
        encoded_box[:, :2][assign_mask] /= 0.1

        encoded_box[:, 2:4][assign_mask] = np.log(box_wh / assigned_priors_wh)
        encoded_box[:, 2:4][assign_mask] /= 0.2

        ldm_encoded = np.zeros_like(encoded_box[:, 5:][assign_mask])
        ldm_encoded = np.reshape(ldm_encoded, [-1, 5, 2])

        ldm_encoded[:, :, 0] = box[[4, 6, 8, 10, 12]] - np.repeat(assigned_priors_center[:, 0:1], 5, axis=-1)
        ldm_encoded[:, :, 1] = box[[5, 7, 9, 11, 13]] - np.repeat(assigned_priors_center[:, 1:2], 5, axis=-1)

        ldm_encoded[:, :, 0] /= np.repeat(assigned_priors_wh[:, 0:1], 5, axis=-1)
        ldm_encoded[:, :, 1] /= np.repeat(assigned_priors_wh[:, 1:2], 5, axis=-1)

        ldm_encoded[:, :, 0] /= 0.1
        ldm_encoded[:, :, 1] /= 0.1

        encoded_box[:, 5:][assign_mask] = np.reshape(ldm_encoded, [-1, 10])
        # print(encoded_box[assign_mask])
        return encoded_box.ravel()

    def assign_boxes(self, boxes):
        assignment = np.zeros((self.num_priors, 4 + 1 + 2 + 1 + 10 + 1))
        assignment[:, 5] = 1
        if len(boxes) == 0:
            return assignment

        # (n, num_priors, 5)
        encoded_boxes = np.apply_along_axis(self.encode_box, 1, boxes)
        # 每一个真实框的编码后的值，和iou
        # (n, num_priors)
        encoded_boxes = encoded_boxes.reshape(-1, self.num_priors, 15)

        # 取重合程度最大的先验框，并且获取这个先验框的index
        # (num_priors)
        best_iou = encoded_boxes[:, :, 4].max(axis=0)
        # (num_priors)
        best_iou_idx = encoded_boxes[:, :, 4].argmax(axis=0)
        # (num_priors)
        best_iou_mask = best_iou > 0
        # 某个先验框它属于哪个真实框
        best_iou_idx = best_iou_idx[best_iou_mask]

        assign_num = len(best_iou_idx)
        # 保留重合程度最大的先验框的应该有的预测结果
        # 哪些先验框存在真实框
        encoded_boxes = encoded_boxes[:, best_iou_mask, :]

        assignment[:, :4][best_iou_mask] = encoded_boxes[best_iou_idx, np.arange(assign_num), :4]
        assignment[:, 4][best_iou_mask] = 1

        assignment[:, 5][best_iou_mask] = 0
        assignment[:, 6][best_iou_mask] = 1
        assignment[:, 7][best_iou_mask] = 1

        assignment[:, 8:-1][best_iou_mask] = encoded_boxes[best_iou_idx, np.arange(assign_num), 5:]
        assignment[:, -1][best_iou_mask] = boxes[best_iou_idx, -1]
        # 通过assign_boxes我们就获得了，输入进来的这张图片，应该有的预测结果是什么样子的

        return assignment

    def decode_boxes(self, mbox_loc, mbox_ldm, mbox_priorbox):
        # 获得先验框的宽与高
        prior_width = mbox_priorbox[:, 2] - mbox_priorbox[:, 0]
        prior_height = mbox_priorbox[:, 3] - mbox_priorbox[:, 1]
        # 获得先验框的中心点
        prior_center_x = 0.5 * (mbox_priorbox[:, 2] + mbox_priorbox[:, 0])
        prior_center_y = 0.5 * (mbox_priorbox[:, 3] + mbox_priorbox[:, 1])

        # 真实框距离先验框中心的xy轴偏移情况
        decode_bbox_center_x = mbox_loc[:, 0] * prior_width * 0.1
        decode_bbox_center_x += prior_center_x
        decode_bbox_center_y = mbox_loc[:, 1] * prior_height * 0.1
        decode_bbox_center_y += prior_center_y

        # 真实框的宽与高的求取
        decode_bbox_width = np.exp(mbox_loc[:, 2] * 0.2)
        decode_bbox_width *= prior_width
        decode_bbox_height = np.exp(mbox_loc[:, 3] * 0.2)
        decode_bbox_height *= prior_height

        # 获取真实框的左上角与右下角
        decode_bbox_xmin = decode_bbox_center_x - 0.5 * decode_bbox_width
        decode_bbox_ymin = decode_bbox_center_y - 0.5 * decode_bbox_height
        decode_bbox_xmax = decode_bbox_center_x + 0.5 * decode_bbox_width
        decode_bbox_ymax = decode_bbox_center_y + 0.5 * decode_bbox_height

        prior_width = np.expand_dims(prior_width, -1)
        prior_height = np.expand_dims(prior_height, -1)
        prior_center_x = np.expand_dims(prior_center_x, -1)
        prior_center_y = np.expand_dims(prior_center_y, -1)

        mbox_ldm = mbox_ldm.reshape([-1, 5, 2])
        decode_ldm = np.zeros_like(mbox_ldm)
        decode_ldm[:, :, 0] = np.repeat(prior_width, 5, axis=-1) * mbox_ldm[:, :, 0] * 0.1 + np.repeat(prior_center_x,
                                                                                                       5, axis=-1)
        decode_ldm[:, :, 1] = np.repeat(prior_height, 5, axis=-1) * mbox_ldm[:, :, 1] * 0.1 + np.repeat(prior_center_y,
                                                                                                        5, axis=-1)

        # 真实框的左上角与右下角进行堆叠
        decode_bbox = np.concatenate((decode_bbox_xmin[:, None],
                                      decode_bbox_ymin[:, None],
                                      decode_bbox_xmax[:, None],
                                      decode_bbox_ymax[:, None],
                                      np.reshape(decode_ldm, [-1, 10])), axis=-1)
        # 防止超出0与1
        decode_bbox = np.minimum(np.maximum(decode_bbox, 0.0), 1.0)
        return decode_bbox

    def detection_out(self, predictions, mbox_priorbox, confidence_threshold=0.4):

        # 网络预测的结果
        mbox_loc = predictions[0][0]
        # 置信度
        mbox_conf = predictions[1][0][:, 1:2]
        # ldm的调整情况
        mbox_ldm = predictions[2][0]

        decode_bbox = self.decode_boxes(mbox_loc, mbox_ldm, mbox_priorbox)

        conf_mask = (mbox_conf >= confidence_threshold)[:, 0]

        detection = np.concatenate((decode_bbox[conf_mask][:, :4], mbox_conf[conf_mask], decode_bbox[conf_mask][:, 4:]),
                                   -1)

        best_box = []
        scores = detection[:, 4]
        # 根据得分对该种类进行从大到小排序。
        arg_sort = np.argsort(scores)[::-1]
        detection = detection[arg_sort]
        while np.shape(detection)[0] > 0:
            # 每次取出得分最大的框，计算其与其它所有预测框的重合程度，重合程度过大的则剔除。
            best_box.append(detection[0])
            if len(detection) == 1:
                break
            ious = iou(best_box[-1], detection[1:])
            detection = detection[1:][ious < self._nms_thresh]

        return best_box


def iou(b1, b2):
    b1_x1, b1_y1, b1_x2, b1_y2 = b1[0], b1[1], b1[2], b1[3]
    b2_x1, b2_y1, b2_x2, b2_y2 = b2[:, 0], b2[:, 1], b2[:, 2], b2[:, 3]

    inter_rect_x1 = np.maximum(b1_x1, b2_x1)
    inter_rect_y1 = np.maximum(b1_y1, b2_y1)
    inter_rect_x2 = np.minimum(b1_x2, b2_x2)
    inter_rect_y2 = np.minimum(b1_y2, b2_y2)

    inter_area = np.maximum(inter_rect_x2 - inter_rect_x1, 0) * \
                 np.maximum(inter_rect_y2 - inter_rect_y1, 0)

    area_b1 = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
    area_b2 = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)

    iou = inter_area / np.maximum((area_b1 + area_b2 - inter_area), 1e-6)
    return iou

def preprocess_input(image):
    image /= 255.0
    image -= 0.5
    image /= 0.5
    return image

#人脸对齐与截取
def Alignment(img, landmark):
    old_img = img
    x = landmark[7] - landmark[5]
    y = landmark[8] - landmark[6]
    img_size = np.asarray(old_img.shape)[0:2]
    if x == 0:
        angle = 0
    else:
        angle = math.atan(y / x) * 180 / math.pi
    #鼻子作为中心点旋转
    center = (landmark[9], landmark[10])

    RotationMatrix = cv2.getRotationMatrix2D(center, angle, 1)
    new_img = cv2.warpAffine(img, RotationMatrix, (img.shape[1], img.shape[0]))
    if not np.min(landmark) <0:
        new_img = new_img[landmark[1]:landmark[3], landmark[0]:landmark[2]]

        ###############活体检测截取人脸图片##################
        x1 = int(np.maximum(landmark[0], 0))
        y1 = int(np.maximum(landmark[1], 0))
        # 人脸框右下角x,y(minimum防止边界情况)
        x2 = int(np.minimum(landmark[2], img_size[0] - 1))
        y2 = int(np.minimum(landmark[3], img_size[1] - 1))
        # 人脸框宽
        w = x2 - x1
        # 人脸框高
        h = y2 - y1
        # 进行放大操作，仅保留人脸部分
        _r = int(max(w, h) * 0.1)#0.38

        x1 = x1 - _r
        y1 = y1 - _r
        x1 = int(max(x1, 0))
        y1 = int(max(y1, 0))
        x2 = x2 + _r
        y2 = y2 + _r
        h, w, c = old_img.shape
        x2 = int(min(x2, w - 2))
        y2 = int(min(y2, h - 2))
        silent_img = old_img[y1:y2, x1:x2]
        # plt.subplot(1, 2, 1)
        # plt.imshow(np.array(silent_img))
        # plt.subplot(1, 2, 2)
        # plt.imshow(np.array(new_img))
        # plt.show()

    # new_landmark = np.array(landmark).astype(np.int)

    return new_img,silent_img



#   计算人脸距离
def face_distance(face_encodings, face_to_compare):
    if len(face_encodings) == 0:
        return np.empty((0))

    distance=[]
    for face_encoding in face_encodings:
        distance.append(np.linalg.norm(face_encoding - face_to_compare, axis=1))
    return np.array(distance)


# ---------------------------------#
#   比较人脸
# ---------------------------------#
def face_contrast(face_encodings, face_encoding, tolerance=0.80):
    # (n)
    dis = face_distance(face_encodings, face_encoding)
    return list(dis <= tolerance), dis


def cv_imread(file_path):
    cv_img = cv2.imdecode(np.fromfile(file_path, dtype=np.uint8), -1)
    return cv_img
def cv_imwrite(file_path,img):
    cv2.imencode('.jpg', img)[1].tofile(file_path)
    