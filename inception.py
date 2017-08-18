#!/usr/bin/env python

from argparse import ArgumentParser
import functools
import glob
import itertools
import multiprocessing
import random
import sys
import time
import signal, traceback

import os
# remove informative logging from tensorflow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import cv2
import numpy
import tensorflow as tf

import generate
import model

# default arguments

# for command line interpreter
def build_parser():
    parser = ArgumentParser()
    parser.add_argument('--weights', '-w',
            dest='weights', help="load pre-trained weights",
            metavar='WEIGHTS')
    parser.add_argument('--output', '-o',
            dest='output', help="result path",
            metavar='OUTOUT', required=True)
    parser.add_argument('--input', '-i',
            dest='input', help="input path",
            metavar='INPUT', required=True)
    parser.add_argument('--version', '-V', action='version', version='%(prog)s 0.0')
    return parser

def inception():

    # set Session
    config = tf.ConfigProto()
    config.allow_soft_placement = True
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    # get model
    print('Restoring model ...')
    saver = tf.train.import_meta_graph(options.weights + '.meta')
    saver.restore(sess, tf.train.latest_checkpoint('./'))
    print('Done.')
    graph = tf.get_default_graph()

    x = graph.get_tensor_by_name('Input:0')
    y = graph.get_tensor_by_name('Code:0')
    x_ = graph.get_tensor_by_name('reconstruct:0')
    #y_ = graph.get_tensor_by_name('y_:0')
    #loss = graph.get_tensor_by_name('loss:0')
    #train_step = graph.get_operation_by_name('train_step')
    loss = - tf.reduce_mean(tf.square(y_), name="loss")
    GD = tf.train.GradientDescentOptimizer(learn_rate, name="GD")
    train_step = GD.minimize(loss, "train_step")
    L3 = graph.get_tensor_by_name('L3:0')
    D7 = graph.get_tensor_by_name('D7:0')


    def do_batch():
        _, ls, rec = sess.run([train_step, loss, x_],
                 feed_dict={x: batch_xs, y_: batch_xs})
        return  ls, rec
    
    def signal_handler(signal, frame):
        print('You pressed Ctrl+C!')
        print('\n\033[1m\033[34m===== train.py interrupted =====\033[0m')
        print('Saving weights')
        saver = tf.train.Saver()
        saver.save(sess, options.save)
        print('done')

        exit(0)

    signal.signal(signal.SIGINT, signal_handler)

    batch_iter = enumerate(read_batches(shape, options, batch_size))
    for batch_idx, batch_xs in batch_iter:
        ls, l3, d7 = do_batch()
        l15 = numpy.percentile(l3, 15)
        l50 = numpy.percentile(l3, 50)
        l85 = numpy.percentile(l3, 85)
        d15 = numpy.percentile(d7, 15)
        d50 = numpy.percentile(d7, 50)
        d85 = numpy.percentile(d7, 85)

        print("{:8d} : {:10.5f} L3[{:10.5f}, {:10.5f}, {:10.5f}] D7[{:10.5f}, {:10.5f}, {:10.5f}]"
                        .format(batch_idx, numpy.sqrt(ls), l15, l50, l85, d15, d50, d85))



def main():
    print('\n\033[1m\033[34m┌───┤Tensorflow version: {}├─┤compiler: {}├─┤git: {}├───┐\033[0m'.format(tf.__version__, tf.__compiler_version__, tf.__git_version__))
    parser = build_parser()
    options = parser.parse_args()

    if options.weights == None:
        print('Training will start from new weights.')

    shape = (options.width, options.height)


    train(learn_rate=0.002,
          report_steps=20,
          batch_size=50,
          shape=shape, options=options)

if __name__ == '__main__':
    main()
