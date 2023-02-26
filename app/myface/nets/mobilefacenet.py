import keras
from keras import backend as K
from keras import initializers
from keras.layers import (BatchNormalization, Conv2D, DepthwiseConv2D, PReLU, Flatten,
                          add)


def conv_block(inputs, filters, kernel_size, strides, padding):
    x = Conv2D(filters, kernel_size, strides=strides, padding=padding, use_bias=False, 
               kernel_initializer=initializers.random_normal(stddev=0.1),
               bias_initializer='zeros')(inputs)
    x = BatchNormalization(axis=-1, epsilon=1e-5)(x)
    x = PReLU(alpha_initializer=initializers.constant(0.25), shared_axes=[1, 2])(x)
    return x

def depthwise_conv_block(inputs, filters, kernel_size, strides):
    x = DepthwiseConv2D(kernel_size, strides=strides, padding="same", use_bias=False,
                        depthwise_initializer=initializers.random_normal(stddev=0.1),
                        bias_initializer='zeros')(inputs)
    x = BatchNormalization(axis=-1, epsilon=1e-5)(x)
    x = PReLU(alpha_initializer=initializers.constant(0.25), shared_axes=[1, 2])(x)
    return x

def bottleneck(inputs, filters, kernel, t, strides, r=False):
    tchannel = K.int_shape(inputs)[-1] * t
    x = conv_block(inputs, tchannel, 1, 1, "same")

    x = DepthwiseConv2D(kernel, strides=strides, padding="same", depth_multiplier=1, use_bias=False,
                        depthwise_initializer=initializers.random_normal(stddev=0.1),
                        bias_initializer='zeros')(x)
    x = BatchNormalization(axis=-1, epsilon=1e-5)(x)
    x = PReLU(alpha_initializer=initializers.constant(0.25), shared_axes=[1, 2])(x)
    
    x = Conv2D(filters, 1, strides=1, padding="same", use_bias=False, 
               kernel_initializer=initializers.random_normal(stddev=0.1),
               bias_initializer='zeros')(x)
    x = BatchNormalization(axis=-1, epsilon=1e-5)(x)
    if r:
        x = add([x, inputs])
    return x

def inverted_residual_block(inputs, filters, kernel, t, n):
    x = inputs
    for _ in range(n):
        x = bottleneck(x, filters, kernel, t, 1, True)
    return x

def mobilefacenet(inputs, embedding_size):
    x = conv_block(inputs, 64, 3, 2, "same")  # Output Shape: (56, 56, 64)
    x = depthwise_conv_block(x, 64, 3, 1)  # (56, 56, 64)

    x = bottleneck(x, 64, 3, t=2, strides=2)
    x = inverted_residual_block(x, 64, 3, t=2, n=4)  # (28, 28, 64)

    x = bottleneck(x, 128, 3, t=4, strides=2)  # (14, 14, 128)
    x = inverted_residual_block(x, 128, 3, t=2, n=6)  # (14, 14, 128)
    
    x = bottleneck(x, 128, 3, t=4, strides=2)  # (14, 14, 128)
    x = inverted_residual_block(x, 128, 3, t=2, n=2)  # (7, 7, 128)
    
    x = Conv2D(512, 1, use_bias=False, name="conv2d",
               kernel_initializer=initializers.random_normal(stddev=0.1),
               bias_initializer='zeros')(x)
    x = BatchNormalization(epsilon=1e-5)(x)
    x = PReLU(alpha_initializer=initializers.constant(0.25), shared_axes=[1, 2])(x)
    
    x = DepthwiseConv2D(int(x.shape[1]), depth_multiplier=1, use_bias=False,
                        depthwise_initializer=initializers.random_normal(stddev=0.1),
                        bias_initializer='zeros')(x)
    x = BatchNormalization(epsilon=1e-5)(x)

    x = Conv2D(embedding_size, 1, use_bias=False,
               kernel_initializer=initializers.random_normal(stddev=0.1),
               bias_initializer='zeros')(x)
    x = BatchNormalization(name="embedding", epsilon=1e-5)(x)
    x = Flatten()(x)
    return x
