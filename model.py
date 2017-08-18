#!/usr/bin/env python
# Copyright (c) 2017 Edouard FISCHER
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
#     The above copyright notice and this permission notice shall be included
#     in all copies or substantial portions of the Software.
#
#     THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
#     OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
#     MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN
#     NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
#     DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
#     OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE
#     USE OR OTHER DEALINGS IN THE SOFTWARE.

__all__ = [
    'get_encoder_layers',
    'get_decoder_layers',
    'get_BN_model'
    ]

import os
# remove informative logging from tensorflow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import numpy as np
import math

# leaky relu
def lrelu(x, leak=0.2, name="lrelu"):
         negative_part = tf.nn.relu(-x)
         x = tf.nn.relu(x)
         x -= tf.constant(leak, dtype=tf.float32) * negative_part
         return tf.identity(x, name=name)

def weight_var(shape, name=None):
  initial = tf.random_uniform(shape, -1.0, 1.0  )
  return tf.Variable(initial, name=name)

def bias_var(shape, name=None):
  initial = tf.zeros(shape=shape)
  return tf.Variable(initial, name=name)

# convolutional layer
def conv(x, W, b, op=tf.identity, stride=(1, 1), padding='SAME', name=None):
    return op(tf.nn.conv2d(x, W, strides=[1, stride[0], stride[1], 1],
                        padding=padding) + b, name=name)

# deconvolutional layer
def deconv(x, W, b, out_shape, op=tf.identity, stride=(1,1), padding='SAME', name=None):
    return op(tf.nn.conv2d_transpose(x, W, out_shape, strides=[1, stride[0], stride[1], 1],
                        padding=padding) + b, name=name)

def get_encoder_layers():
    """
    Get the convolutional layers of the model
    The input layer is size [batch, 32*W, 32*H] (grayscale image)
    The output layer is size [batch, W, H, 15]
    """
    # weights
    W_c_1 = weight_var([3, 3,    1,   16], name="Wc1")
    W_c_2 = weight_var([3, 3,   16,   32], name="Wc2") / math.sqrt(  16)
    W_c_3 = weight_var([3, 3,   32,   64], name="Wc3") / math.sqrt(  32)
    W_c_4 = weight_var([3, 3,   64,  128], name="Wc4") / math.sqrt(  64)
    W_c_5 = weight_var([3, 3,  128,  256], name="Wc5") / math.sqrt( 128)
    W_c_6 = weight_var([3, 3,  256,  512], name="Wc6") / math.sqrt( 256)
    W_c_7 = weight_var([3, 3,  512, 1024], name="Wc7") / math.sqrt( 512)
    W_c_8 = weight_var([3, 3, 1024,  512], name="Wc8") / math.sqrt(1024)
    W_c_9 = weight_var([1, 1,  512,   15], name="Wc9") / math.sqrt( 512)

    # biases
    B_c_1 = bias_var([  16], name="Bc1")
    B_c_2 = bias_var([  32], name="Bc2")
    B_c_3 = bias_var([  64], name="Bc3")
    B_c_4 = bias_var([ 128], name="Bc4")
    B_c_5 = bias_var([ 256], name="Bc5")
    B_c_6 = bias_var([ 512], name="Bc6")
    B_c_7 = bias_var([1024], name="Bc7")
    B_c_8 = bias_var([ 512], name="Bc8")
    B_c_9 = bias_var([  15], name="Bc9")

    # input layer size [batch, 32*W, 32*H]
    x = tf.placeholder(tf.float32, [None, None, None], name="Input")
    # size [batch, 32*W, 32*H, 1]
    x_ex = tf.expand_dims(x, 3, name="Expanded")

    op = lrelu
    #op = tf.nn.relu
    #op = tf.sigmoid
    #op = tf.nn.elu

    # layers
    L1 = conv(x_ex, W_c_1, B_c_1, stride=(2, 2), op=op, name = "L1")
    L2 = conv(  L1, W_c_2, B_c_2, stride=(2, 2), op=op, name = "L2")
    L3 = conv(  L2, W_c_3, B_c_3, stride=(2, 2), op=op, name = "L3")
    L4 = conv(  L3, W_c_4, B_c_4, stride=(2, 2), op=op, name = "L4")
    L5 = conv(  L4, W_c_5, B_c_5, stride=(2, 2), op=op, name = "L5")
    L6 = conv(  L5, W_c_6, B_c_6,                op=op, name = "L6")
    L7 = conv(  L6, W_c_7, B_c_7,                op=op, name = "L7")
    L8 = conv(  L7, W_c_8, B_c_8,                op=op, name = "L8")
    y  = conv(  L8, W_c_9, B_c_9,                op=op, name = "Code")

    return x, y, [ W_c_1, W_c_2, W_c_3, W_c_4, W_c_5,
                   W_c_6, W_c_7, W_c_8, W_c_9,
                   B_c_1, B_c_2, B_c_3, B_c_4, B_c_5,
                   B_c_6, B_c_7, B_c_8, B_c_9 ]

def get_decoder_layers(y, filters):

    W_d_1 = filters[0]
    W_d_2 = filters[1]
    W_d_3 = filters[2]
    W_d_4 = filters[3]
    W_d_5 = filters[4]
    W_d_6 = filters[5]
    W_d_7 = filters[6]
    W_d_8 = filters[7]
    W_d_9 = filters[8]

    B_d_0 = bias_var([   1], name="Bd0")
    B_d_1 = bias_var([  16], name="Bd1")
    B_d_2 = bias_var([  32], name="Bd2")
    B_d_3 = bias_var([  64], name="Bd3")
    B_d_4 = bias_var([ 128], name="Bd4")
    B_d_5 = bias_var([ 256], name="Bd5")
    B_d_6 = bias_var([ 512], name="Bd6")
    B_d_7 = bias_var([1024], name="Bd7")
    B_d_8 = bias_var([ 512], name="Bd8")

    op = lrelu
    #op = tf.nn.relu
    #op = tf.sigmoid
    #op = tf.nn.elu

    batch = tf.shape(y)[0]
    W = tf.shape(y)[1]
    H = tf.shape(y)[2]

    D8 = deconv( y, W_d_9, B_d_8, [batch,    W,    H,  512],                op=op, name = 'D8')
    D7 = deconv(D8, W_d_8, B_d_7, [batch,    W,    H, 1024],                op=op, name = 'D7')
    D6 = deconv(D7, W_d_7, B_d_6, [batch,    W,    H,  512],                op=op, name = 'D6')
    D5 = deconv(D6, W_d_6, B_d_5, [batch,    W,    H,  256],                op=op, name = 'D5')
    D4 = deconv(D5, W_d_5, B_d_4, [batch,  2*W,  2*H,  128], stride=(2, 2), op=op, name = 'D4')
    D3 = deconv(D4, W_d_4, B_d_3, [batch,  4*W,  4*H,   64], stride=(2, 2), op=op, name = 'D3')
    D2 = deconv(D3, W_d_3, B_d_2, [batch,  8*W,  8*H,   32], stride=(2, 2), op=op, name = 'D2')
    D1 = deconv(D2, W_d_2, B_d_1, [batch, 16*W, 16*H,   16], stride=(2, 2), op=op, name = 'D1')
    x_ = deconv(D1, W_d_1, B_d_0, [batch, 32*W, 32*H,    1], stride=(2, 2), op=op, name = 'reconstruct')

    return x_, [W_d_1, W_d_2, W_d_3, W_d_4, W_d_5,
                W_d_6, W_d_7, W_d_8, W_d_9,
                B_d_0, B_d_1, B_d_2, B_d_3, B_d_4,
                B_d_5, B_d_6, B_d_7, B_d_8]
