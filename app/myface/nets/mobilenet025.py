from keras.layers import Conv2D, BatchNormalization, Activation, DepthwiseConv2D
from keras import backend as K


def _conv_block(inputs, filters, kernel=(3, 3), strides=(1, 1)):
    x = Conv2D(filters, kernel,
               padding='same',
               use_bias=False,
               strides=strides,
               name='conv1')(inputs)
    x = BatchNormalization(name='conv1_bn')(x)
    return Activation(relu6, name='conv1_relu')(x)


def _depthwise_conv_block(inputs, pointwise_conv_filters, depth_multiplier=1, strides=(1, 1), block_id=1):
    x = DepthwiseConv2D((3, 3),
                        padding='same',
                        depth_multiplier=depth_multiplier,
                        strides=strides,
                        use_bias=False,
                        name='conv_dw_%d' % block_id)(inputs)

    x = BatchNormalization(name='conv_dw_%d_bn' % block_id)(x)
    x = Activation(relu6, name='conv_dw_%d_relu' % block_id)(x)

    x = Conv2D(pointwise_conv_filters, (1, 1),
               padding='same',
               use_bias=False,
               strides=(1, 1),
               name='conv_pw_%d' % block_id)(x)
    x = BatchNormalization(name='conv_pw_%d_bn' % block_id)(x)
    return Activation(relu6, name='conv_pw_%d_relu' % block_id)(x)


def relu6(x):
    return K.relu(x, max_value=6)


def MobileNet(img_input, depth_multiplier=1):
    # 600 600 3 -> 300 300 8
    x = _conv_block(img_input, 8, strides=(2, 2))
    # 300 300 8 -> 300 300 16
    x = _depthwise_conv_block(x, 16, depth_multiplier, block_id=1)
    # 300 300 16 -> 150 150 32
    x = _depthwise_conv_block(x, 32, depth_multiplier, strides=(2, 2), block_id=2)
    x = _depthwise_conv_block(x, 32, depth_multiplier, block_id=3)
    # 150 150 32 -> 75 75 64
    x = _depthwise_conv_block(x, 64, depth_multiplier, strides=(2, 2), block_id=4)
    x = _depthwise_conv_block(x, 64, depth_multiplier, block_id=5)
    feat1 = x
    # 75 75 64 -> 37.5 37.5 128
    x = _depthwise_conv_block(x, 128, depth_multiplier, strides=(2, 2), block_id=6)
    x = _depthwise_conv_block(x, 128, depth_multiplier, block_id=7)
    x = _depthwise_conv_block(x, 128, depth_multiplier, block_id=8)
    x = _depthwise_conv_block(x, 128, depth_multiplier, block_id=9)
    x = _depthwise_conv_block(x, 128, depth_multiplier, block_id=10)
    x = _depthwise_conv_block(x, 128, depth_multiplier, block_id=11)
    feat2 = x
    # 37.5 37.5 128 -> 18.75 18.75 256
    x = _depthwise_conv_block(x, 256, depth_multiplier, strides=(2, 2), block_id=12)
    x = _depthwise_conv_block(x, 256, depth_multiplier, block_id=13)
    feat3 = x
    return feat1, feat2, feat3
