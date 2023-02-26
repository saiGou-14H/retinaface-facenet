
from app.myface.nets.mobilenet025 import MobileNet

from keras.initializers import random_normal
from keras.layers import Layer, Conv2D, Activation, Add, Concatenate, Input, Reshape
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.normalization import BatchNormalization
from keras.backend import shape
import tensorflow as tf
from keras.models import Model
from app.myface.tools.utils import compose


# 向上采样
class UpsampleLike(Layer):
    def call(self, inputs, **kwargs):
        source, target = inputs
        target_shape = shape(target)
        return tf.compat.v1.image.resize_images(source, (target_shape[1], target_shape[2]),
                                                method=tf.image.ResizeMethod.NEAREST_NEIGHBOR, align_corners=False)

    def compute_output_shape(self, input_shape):
        return (input_shape[0][0],) + input_shape[1][1:3] + (input_shape[0][-1],)


# 卷积块
def Conv2D_BN_Leaky(*args, **kwargs):
    leaky = 0.1
    try:
        leaky = kwargs["leaky"]
        del kwargs["leaky"]
    except:
        pass
    return compose(
        Conv2D(*args, **kwargs),
        BatchNormalization(),
        LeakyReLU(alpha=leaky))


# 卷积块
def Conv2D_BN(*args, **kwargs):
    return compose(
        Conv2D(*args, **kwargs),
        BatchNormalization())


# SSH操作
def SSH(inputs, out_channel, leaky=0.1):
    # 3x3卷积
    conv3x3 = Conv2D_BN(out_channel // 2, kernel_size=3, strides=1, padding='same')(inputs)
    # 2个3x3代替5x5
    temp1 = Conv2D_BN_Leaky(out_channel // 4, kernel_size=3, strides=1, padding='same', leaky=leaky)(inputs)
    conv5x5 = Conv2D_BN(out_channel // 4, kernel_size=3, strides=1, padding='same')(temp1)
    # 3个3x3代替7x7
    temp2 = Conv2D_BN_Leaky(out_channel // 4, kernel_size=3, strides=1, padding='same', leaky=leaky)(temp1)
    conv7x7 = Conv2D_BN(out_channel // 4, kernel_size=3, strides=1, padding='same')(temp2)
    # 堆叠
    out = Concatenate(axis=-1)([conv3x3, conv5x5, conv7x7])
    out = Activation("relu")(out)
    return out


# 分类预测结果(判断先验框内部是否包含物体)
def ClassHead(inputs, num_anchors=2):
    outputs = Conv2D(num_anchors * 2, kernel_size=1, strides=1)(inputs)
    return Activation("softmax")(Reshape([-1, 2])(outputs))


# 框的回归预测结果(对先验框进行调整获得预测框)
def BboxHead(inputs, num_anchors=2):
    outputs = Conv2D(num_anchors * 4, kernel_size=1, strides=1)(inputs)
    return Reshape([-1, 4])(outputs)


# 人脸关键点的回归预测结果(对先验框进行调整获得人脸关键点)
def LandmarkHead(inputs, num_anchors=2):
    outputs = Conv2D(num_anchors * 10, kernel_size=1, strides=1)(inputs)
    return Reshape([-1, 10])(outputs)


def RetinaFace(cfg, backbone="mobilenet"):
    inputs = Input(shape=(None, None, 3))

    if backbone == "mobilenet":
        C3, C4, C5 = MobileNet(inputs)
    leaky = 0
    if cfg['out_channel'] <= 64:
        leaky = 0.1
    # 获得3个shape的有效特征层
    P3 = Conv2D_BN_Leaky(cfg['out_channel'], kernel_size=1, strides=1, padding='same',
                         kernel_initializer=random_normal(stddev=0.02), name='C3_reduced', leaky=leaky)(C3)
    P4 = Conv2D_BN_Leaky(cfg['out_channel'], kernel_size=1, strides=1, padding='same',
                         kernel_initializer=random_normal(stddev=0.02), name='C4_reduced', leaky=leaky)(C4)
    P5 = Conv2D_BN_Leaky(cfg['out_channel'], kernel_size=1, strides=1, padding='same',
                         kernel_initializer=random_normal(stddev=0.02), name='C5_reduced', leaky=leaky)(C5)
    # P5向上采样(P5长和宽调整成P4的大小),之后再与P4特征融合
    P5_UP = UpsampleLike(name='P5_UP')([P5, P4])
    P4 = Add(name='P4_merged')([P5_UP, P4])
    P4 = Conv2D_BN_Leaky(cfg['out_channel'], kernel_size=3, strides=1, padding='same',
                         kernel_initializer=random_normal(stddev=0.02), name='Conv_P4_merged', leaky=leaky)(P4)
    # P4向上采样(P4长和宽调整成P3的大小),之后再与P3特征融合
    P4_UP = UpsampleLike(name='P4_UP')([P4, P3])
    P3 = Add(name='P3_merged')([P4_UP, P3])
    P3 = Conv2D_BN_Leaky(cfg['out_channel'], kernel_size=3, strides=1, padding='same',
                         kernel_initializer=random_normal(stddev=0.02), name='Conv_P3_merged', leaky=leaky)(P3)

    SSH1 = SSH(P3, cfg['out_channel'], leaky=leaky)
    SSH2 = SSH(P4, cfg['out_channel'], leaky=leaky)
    SSH3 = SSH(P5, cfg['out_channel'], leaky=leaky)

    SSH_total = [SSH1, SSH2, SSH3]

    bbox_regressions = Concatenate(axis=1, name="bbox_reg")([BboxHead(feature) for feature in SSH_total])
    classifications = Concatenate(axis=1, name="cls")([ClassHead(feature) for feature in SSH_total])
    ldm_regressions = Concatenate(axis=1, name="ldm_reg")([LandmarkHead(feature) for feature in SSH_total])

    output = [bbox_regressions, classifications, ldm_regressions]

    model = Model(inputs=inputs, outputs=output)
    return model
