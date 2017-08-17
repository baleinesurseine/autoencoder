
__all__ = [
    'get_encoder_layers',
    'get_decoder_layers',
    'get_BN_model'
    ]

import tensorflow as tf
import numpy as np
import math

# leaky relu
def lrelu(x, leak=0.2, name="lrelu"):
         negative_part = tf.nn.relu(-x)
         x = tf.nn.relu(x)
         x -= tf.constant(leak, dtype=tf.float32) * negative_part
         return tf.identity(x, name=name)

def weight_variable(shape, name=None):
  #initial = tf.truncated_normal(shape, stddev=0.1)
  n_input = shape[2]
  n_output = shape[3]
  #initial = tf.truncated_normal(shape, stddev = 0.1)
  initial = tf.random_uniform(shape, -1.0, 1.0  )
  return tf.Variable(initial, name=name)

def bias_variable(shape, name=None):
  #initial = tf.constant(0.1, shape=shape)
  initial = tf.zeros(shape=shape)
  return tf.Variable(initial, name=name)

def convolutional(x, W, b, op=tf.identity, stride=(1, 1), padding='SAME', name=None):
    return op(tf.nn.conv2d(x, W, strides=[1, stride[0], stride[1], 1],
                        padding=padding) + b, name=name)
def deconvolutional(x, W, b, out_shape, op=tf.identity, stride=(1,1), padding='SAME', name=None):
    return op(tf.nn.conv2d_transpose(x, W, out_shape, strides=[1, stride[0], stride[1], 1],
                        padding=padding) + b, name=name)

def max_pool(x, ksize=(2, 2), stride=(2, 2), name=None):
  return tf.nn.max_pool(x, ksize=[1, ksize[0], ksize[1], 1],
                        strides=[1, stride[0], stride[1], 1], padding='SAME', name=name)

def batchnorm(Ylogits, is_test, iteration, offset, convolutional=False):
    exp_moving_avg = tf.train.ExponentialMovingAverage(0.999, iteration) # adding the iteration prevents from averaging across non-existing iterations
    bnepsilon = 1e-5
    if convolutional:
        mean, variance = tf.nn.moments(Ylogits, [0, 1, 2])
    else:
        mean, variance = tf.nn.moments(Ylogits, [0])
    update_moving_everages = exp_moving_avg.apply([mean, variance])
    m = tf.cond(is_test, lambda: exp_moving_avg.average(mean), lambda: mean)
    v = tf.cond(is_test, lambda: exp_moving_avg.average(variance), lambda: variance)
    Ybn = tf.nn.batch_normalization(Ylogits, m, v, offset, None, bnepsilon)
    return Ybn, update_moving_everages

def convolutional_BN(x, W, b, iter, is_test, op=tf.identity, stride=(1,1), padding='SAME', name=None):
    Y_c = tf.nn.conv2d(x, W, strides=[1, stride[0], stride[1], 1], padding=padding)
    Y_n, update_ema = batchnorm(Y_c, is_test, iter, b, convolutional=True)
    Y = op(Y_n, name=name)
    return Y, update_ema

def deconvolutional_BN(x ,W, b, out_shape, iter, is_test, op=tf.identity, stride=(1,1), padding='SAME', name=None):
    Y_d = tf.nn.conv2d_transpose(x, W, out_shape, strides=[1, stride[0], stride[1], 1], padding=padding)
    Y_n, update_ema = batchnorm(Y_d, is_test, iter, b, convolutional=True)
    Y = op(Y_n, name=name)
    return Y, update_ema

def get_BN_model():
    tst = tf.placeholder(tf.bool)
    iter = tf.placeholder(tf.int32)

    # weights
    B_dconv_0 = bias_variable([1], name="Bd0")
    
    W_conv_1 = weight_variable([3, 3, 1, 16], name="Wc1")
    B_conv_1 = bias_variable([16], name="Bc1")
    B_dconv_1 = bias_variable([16], name="Bd1")

    W_conv_2 = weight_variable([3, 3, 16, 32], name="Wc2") / math.sqrt(16)
    B_conv_2 = bias_variable([32], name="Bc2")
    B_dconv_2 = bias_variable([32], name="Bd2")

    W_conv_3 = weight_variable([3, 3, 32, 64], name="Wc3") / math.sqrt(32)
    B_conv_3 = bias_variable([64], name="Bc3")
    B_dconv_3 = bias_variable([64], name="Bd3")

    W_conv_4 = weight_variable([3, 3, 64, 128], name="Wc4") / math.sqrt(64)
    B_conv_4 = bias_variable([128], name="Bc4")
    B_dconv_4 = bias_variable([128], name="Bd4")

    W_conv_5 = weight_variable([3, 3, 128, 256], name="Wc5") / math.sqrt(128)
    B_conv_5 = bias_variable([256], name="Bc5")
    B_dconv_5 = bias_variable([256], name="Bd5")

    W_conv_6 = weight_variable([3, 3, 256, 512], name="Wc6") / math.sqrt(256)
    B_conv_6 = bias_variable([512], name="Bc6")
    B_dconv_6 = bias_variable([512], name="Bd6")

    W_conv_7 = weight_variable([3, 3, 512, 1024], name="Wc7") / math.sqrt(512)
    B_conv_7 = bias_variable([1024], name="Bc7")
    B_dconv_7 = bias_variable([1024], name="Bd7")

    W_conv_8 = weight_variable([3, 3, 1024, 512], name="Wc8") / math.sqrt(1024)
    B_conv_8 = bias_variable([512], name="Bc8")
    B_dconv_8 = bias_variable([512], name="Bd8")

    W_conv_9 = weight_variable([1, 1, 512, 15], name="Wc9") / math.sqrt(512)
    B_conv_9 = bias_variable([15], name="Bc9")

    # input layer size [batch, 32*W, 32*H]
    x = tf.placeholder(tf.float32, [None, None, None], name="Input")
    # size [batch, 32*W, 32*H, 1]
    x_expanded = tf.expand_dims(x, 3, name="Expanded")

    #activation = tf.nn.relu
    activation = lrelu
    #activation = tf.sigmoid
    #activation = tf.nn.elu

    L1, upd_ma1 = convolutional_BN(x_expanded, W_conv_1, B_conv_1, iter, tst, stride=(2, 2), op=activation, name = "L1")
    L2, upd_ma2 = convolutional_BN(L1, W_conv_2, B_conv_2, iter, tst, stride=(2, 2), op=activation, name = "L2")
    L3, upd_ma3 = convolutional_BN(L2, W_conv_3, B_conv_3, iter, tst, stride=(2, 2), op=activation, name = "L3")
    L4, upd_ma4 = convolutional_BN(L3, W_conv_4, B_conv_4, iter, tst, stride=(2, 2), op=activation, name = "L4")
    L5, upd_ma5 = convolutional_BN(L4, W_conv_5, B_conv_5, iter, tst, stride=(2, 2), op=activation, name = "L5")
    L6, upd_ma6 = convolutional_BN(L5, W_conv_6, B_conv_6, iter, tst, op=activation, name = "L6")
    L7, upd_ma7 = convolutional_BN(L6, W_conv_7, B_conv_7, iter, tst, op=activation, name = "L7")
    L8, upd_ma8 = convolutional_BN(L7, W_conv_8, B_conv_8, iter, tst, op=activation, name = "L8")
    y, upd_may = convolutional_BN(L8, W_conv_9, B_conv_9, iter, tst, op=activation, name = "Code")

    batch = tf.shape(y)[0]
    W = tf.shape(y)[1]
    H = tf.shape(y)[2]

    D8, upd_mad8 = deconvolutional_BN(y, W_dconv_9, B_dconv_8, [batch, W, H, 512], iter, tst, op=activation, name = 'D8')
    D7, upd_mad7 = deconvolutional_BN(D8, W_dconv_8, B_dconv_7, [batch, W, H, 1024], iter, tst, op=activation, name = 'D7')
    D6, upd_mad6 = deconvolutional_BN(D7, W_dconv_7, B_dconv_6, [batch, W, H, 512], iter, tst, op=activation, name = 'D6')
    D5, upd_mad5 = deconvolutional_BN(D6, W_dconv_6, B_dconv_5, [batch, W, H, 256], iter, tst, op=activation, name = 'D5')
    D4, upd_mad4 = deconvolutional_BN(D5, W_dconv_5, B_dconv_4, [batch, 2*W, 2*H, 128], iter, tst, stride=(2, 2), op=activation, name = 'D4')
    D3, upd_mad3 = deconvolutional_BN(D4, W_dconv_4, B_dconv_3, [batch, 4*W, 4*H, 64], iter, tst, stride=(2, 2), op=activation, name = 'D3')
    D2, upd_mad2 = deconvolutional_BN(D3, W_dconv_3, B_dconv_2, [batch, 8*W, 8*H, 32], iter, tst, stride=(2, 2), op=activation, name = 'D2')
    D1, upd_mad1 = deconvolutional_BN(D2, W_dconv_2, B_dconv_1, [batch, 16*W, 16*H, 16], iter, tst, stride=(2, 2), op=activation, name = 'D1')
    x_, upd_max_ = deconvolutional_BN(D1, W_dconv_1, B_dconv_0, [batch, 32*W, 32*H, 1], iter, tst, stride=(2, 2), op=activation, name = 'reconstruct')

    update_ema = tf.group(upd_ma1, upd_ma2, upd_ma3, upd_ma4, upd_ma5, upd_ma6, upd_ma7, upd_ma8, upd_may,
                        upd_mad1, upd_mad2, upd_mad3, upd_mad4, upd_mad5, upd_mad6, upd_mad7, upd_mad8, upd_max_)

    return x, y, x_, tst, iter, update_ema

def get_encoder_layers():
    """
    Get the convolutional layers of the model

    The input layer is size [batch, 32*W, 32*H] (grayscale image)
    The output layer is size [batch, W, H, 15]

    """
    # weights
    B_dconv_0 = bias_variable([1], name="Bd0")

    W_conv_1 = weight_variable([3, 3, 1, 16], name="Wc1")
    B_conv_1 = bias_variable([16], name="Bc1")
    B_dconv_1 = bias_variable([16], name="Bd1")

    W_conv_2 = weight_variable([3, 3, 16, 32], name="Wc2") / math.sqrt(16)
    B_conv_2 = bias_variable([32], name="Bc2")
    B_dconv_2 = bias_variable([32], name="Bd2")

    W_conv_3 = weight_variable([3, 3, 32, 64], name="Wc3") / math.sqrt(32)
    B_conv_3 = bias_variable([64], name="Bc3")
    B_dconv_3 = bias_variable([64], name="Bd3")

    W_conv_4 = weight_variable([3, 3, 64, 128], name="Wc4") / math.sqrt(64)
    B_conv_4 = bias_variable([128], name="Bc4")
    B_dconv_4 = bias_variable([128], name="Bd4")

    W_conv_5 = weight_variable([3, 3, 128, 256], name="Wc5") / math.sqrt(128)
    B_conv_5 = bias_variable([256], name="Bc5")
    B_dconv_5 = bias_variable([256], name="Bd5")

    W_conv_6 = weight_variable([3, 3, 256, 512], name="Wc6") / math.sqrt(256)
    B_conv_6 = bias_variable([512], name="Bc6")
    B_dconv_6 = bias_variable([512], name="Bd6")

    W_conv_7 = weight_variable([3, 3, 512, 1024], name="Wc7") / math.sqrt(512)
    B_conv_7 = bias_variable([1024], name="Bc7")
    B_donv_7 = bias_variable([1024], name="Bd7")

    W_conv_8 = weight_variable([3, 3, 1024, 512], name="Wc8") / math.sqrt(1024)
    B_conv_8 = bias_variable([512], name="Bc8")
    B_donv_8 = bias_variable([512], name="Bd8")

    W_conv_9 = weight_variable([1, 1, 512, 15], name="Wc9") / math.sqrt(512)
    B_conv_9 = bias_variable([15], name="Bc9")

    # input layer size [batch, 32*W, 32*H]
    x = tf.placeholder(tf.float32, [None, None, None], name="Input")
    # size [batch, 32*W, 32*H, 1]
    x_expanded = tf.expand_dims(x, 3, name="Expanded")

    #activation = tf.nn.relu
    activation = lrelu
    #activation = tf.sigmoid
    #activation = tf.nn.elu

    # tiny yolo layers
    L1 = convolutional(x_expanded, W_conv_1, B_conv_1, stride=(2, 2), op=activation, name = "L1")
    L2 = convolutional(L1, W_conv_2, B_conv_2, stride=(2, 2), op=activation, name = "L2")
    L3 = convolutional(L2, W_conv_3, B_conv_3, stride=(2, 2), op=activation, name = "L3")
    L4 = convolutional(L3, W_conv_4, B_conv_4, stride=(2, 2), op=activation, name = "L4")
    L5 = convolutional(L4, W_conv_5, B_conv_5, stride=(2, 2), op=activation, name = "L5")
    L6 = convolutional(L5, W_conv_6, B_conv_6, op=activation, name = "L6")
    L7 = convolutional(L6, W_conv_7, B_conv_7, op=activation, name = "L7")
    L8 = convolutional(L7, W_conv_8, B_conv_8, op=activation, name = "L8")
    y = convolutional(L8, W_conv_9, B_conv_9, op=activation, name = "Code")

    return x, y, [
                W_conv_1, W_conv_2, W_conv_3, W_conv_4, W_conv_5,
                W_conv_6, W_conv_7, W_conv_8, W_conv_9,
                B_conv_1, B_conv_2, B_conv_3, B_conv_4, B_conv_5,
                B_conv_6, B_conv_7, B_conv_8, B_conv_9
                ]

def get_decoder_layers(y, filters):
    B_dconv_0 = bias_variable([1], name="Bd0")
    W_dconv_1 = filters[0]
    B_dconv_1 = bias_variable([16], name="Bd1")
    W_dconv_2 = filters[1]
    B_dconv_2 = bias_variable([32], name="Bd2")
    W_dconv_3 = filters[2]
    B_dconv_3 = bias_variable([64], name="Bd3")
    W_dconv_4 = filters[3]
    B_dconv_4 = bias_variable([128], name="Bd4")
    W_dconv_5 = filters[4]
    B_dconv_5 = bias_variable([256], name="Bd5")
    W_dconv_6 = filters[5]
    B_dconv_6 = bias_variable([512], name="Bd6")
    W_dconv_7 = filters[6]
    B_dconv_7 = bias_variable([1024], name="Bd7")
    W_dconv_8 = filters[7]
    B_dconv_8 = bias_variable([512], name="Bd8")
    W_dconv_9 = filters[8]

    #activation = tf.nn.relu
    activation = lrelu
    #activation = tf.sigmoid
    #activation = tf.nn.elu

    batch = tf.shape(y)[0]
    W = tf.shape(y)[1]
    H = tf.shape(y)[2]

    D8 = deconvolutional(y, W_dconv_9, B_dconv_8, [batch, W, H, 512], op=activation, name = 'D8')
    D7 = deconvolutional(D8, W_dconv_8, B_dconv_7, [batch, W, H, 1024], op=activation, name = 'D7')
    D6 = deconvolutional(D7, W_dconv_7, B_dconv_6, [batch, W, H, 512], op=activation, name = 'D6')
    D5 = deconvolutional(D6, W_dconv_6, B_dconv_5, [batch, W, H, 256], op=activation, name = 'D5')
    D4 = deconvolutional(D5, W_dconv_5, B_dconv_4, [batch, 2*W, 2*H, 128], stride=(2, 2), op=activation, name = 'D4')
    D3 = deconvolutional(D4, W_dconv_4, B_dconv_3, [batch, 4*W, 4*H, 64], stride=(2, 2), op=activation, name = 'D3')
    D2 = deconvolutional(D3, W_dconv_3, B_dconv_2, [batch, 8*W, 8*H, 32], stride=(2, 2), op=activation, name = 'D2')
    D1 = deconvolutional(D2, W_dconv_2, B_dconv_1, [batch, 16*W, 16*H, 16], stride=(2, 2), op=activation, name = 'D1')
    x_ = deconvolutional(D1, W_dconv_1, B_dconv_0, [batch, 32*W, 32*H, 1], stride=(2, 2), op=activation, name = 'reconstruct')

    return x_[:, :, :, 0], [W_dconv_1, W_dconv_2, W_dconv_3, W_dconv_4, W_dconv_5,
                            W_dconv_6, W_dconv_7, W_dconv_8, W_dconv_9,
                            B_dconv_0, B_dconv_1, B_dconv_2, B_dconv_3, B_dconv_4,
                            B_dconv_5, B_dconv_6, B_dconv_7, B_dconv_8]
