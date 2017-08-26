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
from functools import wraps
import itertools
import multiprocessing
import signal
import os, errno
import numpy as np

# remove informative logging from tensorflow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf

import generate
import model

# decorator for multiprocessing generator function
def mpgen(f):
    def mainmp(q, args, kwargs):
        try:
            for item in f(*args, **kwargs):
                q.put(item)
        finally:
            q.close()
    @wraps(f)
    def wrapped(*args, **kwargs):
        q = multiprocessing.Queue(3)
        proc = multiprocessing.Process(target=mainmp, args=(q, args, kwargs))
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
def read_batches(shape, options, batch_size):
    """
    Returns a generator of batches
    """
    # get an (infinite) image generator
    g = generate.generate_ims(shape, options.bg)
    while True:
        #yield a batch of images of size batch_size
        yield list(itertools.islice(g, batch_size))

# default arguments
SAVE = 'my_model'
# for command line interpreter
def build_parser():
    parser = ArgumentParser()
    parser.add_argument('--sparse', '-x', type=float,
            dest='sparse', help="sparsity penalty (default '%(default)s')",
            metavar='SPARSE', default=0.0)
    parser.add_argument('--weights', '-w',
            dest='weights', help="load pre-trained weights",
            metavar='WEIGHTS')
    parser.add_argument('--save', '-s',
            dest='save', help="weight backup path (default '%(default)s')",
            metavar='SAVE', default=SAVE)
    parser.add_argument('--images', '-i',
            dest='bg', help='train images folder',
            metavar='BG', required=True)
    parser.add_argument('--width', '-l', type=int,
            dest='width', help="width of train images",
            metavar='WIDTH', required=True)
    parser.add_argument('--height', '-u', type=int,
            dest='height', help="height of train images",
            metavar='HEIGHT', required=True)
    parser.add_argument('--version', '-V', action='version', version='%(prog)s 0.0')
    return parser

def train(learn_rate, batch_size, shape, options):
    """
    Runs a training loop, minimizing L2 norm of the reconstruction error plus
    a L1 penalty on coding layer for sparsity
    """
    # set Session
    config = tf.ConfigProto()
    config.allow_soft_placement = True
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    # get model
    if options.weights == None:
        # create model
        x, y, params_enc = model.get_encoder_layers()
        x_, params_dec = model.get_decoder_layers(y, params_enc)
        graph = tf.get_default_graph()
        adam = tf.train.AdamOptimizer(learn_rate,name="adam")
        y_ = tf.placeholder(tf.float32, [None, None, None], name='y_')
        loss = tf.reduce_mean(tf.square(x_[:,:,:,0] - y_), name="loss")
        cst = tf.constant(options.sparse, name="sparsity")
        spa = tf.reduce_mean(tf.abs(y), name = "sparse")
        train_step = adam.minimize(loss + cst * spa, name="train_step")
        init = tf.global_variables_initializer()
        sess.run(init)
    else:
        # restore model from files
        print('Restoring model graph ...', end='')
        saver = tf.train.import_meta_graph('./' + options.weights + '/' + options.weights + '.meta')
        print('Done.')
        print('Restoring values from checkpoint ...', end='')
        saver.restore(sess, tf.train.latest_checkpoint('./' + options.weights))
        print('Done.')
        graph = tf.get_default_graph()
        x = graph.get_tensor_by_name('Input:0')
        y = graph.get_tensor_by_name('Code:0')
        x_ = graph.get_tensor_by_name('reconstruct:0')
        y_ = graph.get_tensor_by_name('y_:0')
        loss = graph.get_tensor_by_name('loss:0')
        spa = graph.get_tensor_by_name('sparse:0')
        train_step = graph.get_operation_by_name('train_step')

    # save model before exit
    make_sigint_handler(sess, options.save)

    # iterate train steps on batches
    batch_iter = enumerate(read_batches(shape, options, batch_size))
    print('┌────────┬─────────┬─────────┐')
    print('│ #Batch │ Reconst.│ Sparsity│')
    print('├────────┼─────────┼─────────┤')
    for batch_idx, batch_xs in batch_iter:
        _, ls, sp = sess.run([train_step, loss, spa], feed_dict={x: batch_xs, y_: batch_xs})
        print("│{:8d}│{:9.5f}│{:9.5f}│".format(batch_idx, np.sqrt(ls), sp))

def make_sigint_handler(sess,filename):
    """
    Handle CTRL-C from the user input, and saves the state of the network
    """
    def signal_handler(sig, frame):
        print('\n\033[1m\033[34m===== train.py[pid:{}] interrupted =====\033[0m'.format(os.getpid()))
        try:
            # create directory if doesn't exist
            os.makedirs(filename)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise
        saver = tf.train.Saver()
        print('Saving weights on', saver.save(sess, './' + filename + '/' + filename), ': done.')
        exit(0)
    # connect handler callback to SIGINT signal
    signal.signal(signal.SIGINT, signal_handler)

def main():
    print('\033[1m\033[34mTensorflow version: {} compiler: {} git: {}\033[0m'.format(tf.__version__, tf.__compiler_version__, tf.__git_version__))
    parser = build_parser()
    options = parser.parse_args()

    if options.weights == None:
        print('Training will start from new weights.')

    shape = (options.width, options.height)
    train(learn_rate=0.002,
          batch_size=50,
          shape=shape, options=options)

if __name__ == '__main__':
    main()
