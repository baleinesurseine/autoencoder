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

def mpgen(f):
    def mainmp(q, args, kwargs):
        try:
            for item in f(*args, **kwargs):
                q.put(item)
        finally:
            q.close()

    @functools.wraps(f)
    def wrapped(*args, **kwargs):
        q = multiprocessing.Queue(3)
        proc = multiprocessing.Process(target=mainmp,
                                       args=(q, args, kwargs))
        proc.start()
        try:
            while True:
                item = q.get()
                yield item
        finally:
            proc.terminate()
            proc.join()

    return wrapped

@mpgen
# return a generator of batches
def read_batches(shape, options, batch_size):
    # get an (infinite) image generator
    g = generate.generate_ims(shape, options.bg)
    while True:
        #yield a batch of images of size batch_size
        yield list(itertools.islice(g, batch_size))



# default arguments
TEST = './test'
SAVE = './weights.s'
# for command line interpreter
def build_parser():
    parser = ArgumentParser()
    parser.add_argument('--test', '-t',
            dest='test', help="test path (default '%(default)s')",
            metavar='TEST', default=TEST)
    parser.add_argument('--weights', '-w',
            dest='weights', help="load pre-trained weights",
            metavar='WEIGHTS')
    parser.add_argument('--save', '-s',
            dest='save', help="weight backup path (default '%(default)s')",
            metavar='SAVE', default=SAVE)
    parser.add_argument('--background', '-b',
            dest='bg', help='background images folder',
            metavar='BG', required=True)
    parser.add_argument('--width', '-l', type=int,
            dest='width', help="width of images",
            metavar='WIDTH', required=True)
    parser.add_argument('--height', '-u', type=int,
            dest='height', help="height of images",
            metavar='HEIGHT', required=True)
    parser.add_argument('--version', '-V', action='version', version='%(prog)s 0.0')
    return parser

def train(learn_rate, report_steps, batch_size, shape, options):

    # set Session
    config = tf.ConfigProto()
    config.allow_soft_placement = True
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    # get model
    if options.weights == None:
        x, y, params_enc = model.get_encoder_layers()
        x_, params_dec = model.get_decoder_layers(y, params_enc)
        adam = tf.train.AdamOptimizer(learn_rate,name="adam")
        y_ = tf.placeholder(tf.float32, [None, None, None], name='y_')
        loss = tf.reduce_mean(tf.square(x_ - y_), name="loss")
        train_step = adam.minimize(loss, name="train_step")
        init = tf.global_variables_initializer()
        sess.run(init)
    else:
        print('Restoring model ...')
        saver = tf.train.import_meta_graph(options.weights + '.meta')
        saver.restore(sess, tf.train.latest_checkpoint('./'))
        print('Done.')
        graph = tf.get_default_graph()
        x = graph.get_tensor_by_name('Input:0')
        y = graph.get_tensor_by_name('Code:0')
        x_ = graph.get_tensor_by_name('reconstruct:0')
        y_ = graph.get_tensor_by_name('y_:0')
        loss = graph.get_tensor_by_name('loss:0')
        train_step = graph.get_operation_by_name('train_step')

    def do_batch():
        _, ls = sess.run([train_step, loss],
                 feed_dict={x: batch_xs, y_: batch_xs})
        return ls

    def signal_handler(signal, frame):
        print('You pressed Ctrl+C!')
        print('\n\033[1m\033[34m===== train.py interrupted =====\033[0m')
        print('Saving weights')
        saver = tf.train.Saver()
        saver.save(sess, 'my_model')
        print('done')

        exit(0)

    signal.signal(signal.SIGINT, signal_handler)

    batch_iter = enumerate(read_batches(shape, options, batch_size))
    #for batch_idx, (batch_xs, batch_ys) in batch_iter:
    for batch_idx, batch_xs in batch_iter:
        ls = do_batch()
        print(batch_idx, numpy.sqrt(ls))



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