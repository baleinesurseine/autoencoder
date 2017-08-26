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

from argparse import ArgumentParser
import cv2
import numpy as np
import os
import colorsys
# remove informative logging from tensorflow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
import model

def gray_heat(map):
    """
    Outputs a heat-map from a gray image
    """
    map = (map - 0.5) * 5.0 + 0.5
    H = map.shape[0]
    W = map.shape[1]
    out = np.zeros((H,W,3))
    for h in range(0,H):
        for w in range(0,W):
            # (240, )
            out[h,w,:] = colorsys.hls_to_rgb((1.0-map[h,w])*0.66667, 0.5, 1.0)
    return out

# for command line interpreter
def build_parser():
    parser = ArgumentParser()
    parser.add_argument('--weights', '-w',
            dest='weights', help="pre-trained weights path",
            metavar='WEIGHTS', required = True)
    parser.add_argument('--input', '-i',
            dest='input', help="input image path",
            metavar='INPUT', required = True)
    parser.add_argument('--output', '-o',
            dest='output', help="output image path",
            metavar='OUTPUT', required = True)
    parser.add_argument('--filter', '-f', action='store_true')
    parser.add_argument('--version', '-V', action='version', version='%(prog)s 0.0')
    return parser

def plotCode(code):
    """
    Plots the features of the most internal layer (code layer) as 15 tiled images
    code if size [W, H, 15]
    """
    # rescale features
    mincode = np.amin(code)
    maxcode = np.amax(code)
    print('Min: ', mincode, 'Max: ', maxcode)
    code = (code - mincode) / (maxcode - mincode)
    # create output image
    sh = np.shape(code)
    W = sh[0]
    H = sh[1]
    out = np.zeros((3*(W+2)-2, 5*(H+2)-2))
    # copy each feature in out
    for w in range(0,3):
        for h in range(0,5):
            c = w*5 + h
            out[w*(W+2):w*(W+2)+W, h*(H+2):h*(H+2)+H] = code[:,:,c]
    return out

def plotWeights(w):
    """
    Plots the weights of the first convolutional layer as a 16 tiles 3x3 images
    """
    w = w[:,:,0,:]
    # rescale w to 0.0 - 1.0
    mincode = np.amin(w)
    maxcode = np.amax(w)
    w = (w - mincode) / (maxcode - mincode)

    out = np.zeros((15, 15))
    for x in range(0,4):
        for y in range(0,4):
            c = x*4+y
            out[x*4:x*4+3, y*4:y*4+3] = w[:,:,c]
    return out

def main():
    parser = build_parser()
    options = parser.parse_args()
    # set Session
    config = tf.ConfigProto()
    config.allow_soft_placement = True
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    # get model
    print('Restoring model ...', end='')
    saver = tf.train.import_meta_graph('./' + options.weights + '/' + options.weights + '.meta')
    print('Done.')
    print('Restoring checkpoint ...', end='')
    saver.restore(sess, tf.train.latest_checkpoint('./' + options.weights))
    print('Done.')
    graph = tf.get_default_graph()

    if options.filter:
        # outputs only first convolutional layer weights
        Wc1 = graph.get_tensor_by_name('Wc1:0')
        w = sess.run(Wc1)
        out = plotWeights(w)
        cv2.imwrite(options.output, out * 255.)
        exit(0)

    x = graph.get_tensor_by_name('Input:0')
    y = graph.get_tensor_by_name('Code:0')
    x_ = graph.get_tensor_by_name('reconstruct:0')
    y_ = tf.placeholder(tf.float32, [None, None, None], name='y_')
    loss = tf.reduce_mean(tf.square(x_[:,:,:,0] - y_), name="loss")
    sparse = tf.reduce_mean(tf.abs(y))
    # read image to encode
    image = cv2.imread(options.input, 0) / 255.
    H = image.shape[0]
    W = image.shape[1]
    # resize and crop image to have multiples of 32 in both dimension
    H1 = int(np.ceil(H/32))*32
    W1 = int(np.ceil(W/32))*32
    ratio = max(H1/H, W1/W)
    im = cv2.resize(image, (int(W * ratio) + 1, int(H * ratio) + 1))
    im = im[0:H1, 0:W1]
    batch_xs = [im]
    print(im.shape)
    # compute inference
    code, rec, ls, sp = sess.run([y, x_, loss, sparse],
        feed_dict={x: batch_xs, y_: batch_xs})
    # print error and sparsity of the encoding
    print(np.sqrt(ls), sp)
    # get reconstructed image
    rec = rec[0,:,:,0]
    # clip to [0..1] (relu activation doesn't ensure output is <1)
    out = np.clip(rec, 0., 1.)
    dif = gray_heat(((out - im)+1.)/2.)
    mapcode = plotCode(code[0, :, :, :])
    # save images : reconstructed, error and code
    cv2.imwrite('rec-'+options.output,     out * 255.)
    cv2.imwrite('dif-'+options.output,     dif * 255.)
    cv2.imwrite('cod-'+options.output, mapcode * 255.)

if __name__ == '__main__':
    main()
