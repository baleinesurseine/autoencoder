
from argparse import ArgumentParser
import cv2
import numpy

import os
# remove informative logging from tensorflow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import cv2
import tensorflow as tf
import model

# for command line interpreter
def build_parser():
    parser = ArgumentParser()
    parser.add_argument('--weights', '-w',
            dest='weights', help="load pre-trained weights",
            metavar='WEIGHTS', required = True)
    parser.add_argument('--input', '-i',
            dest='input', help="input image filename",
            metavar='INPUT', required = True)
    parser.add_argument('--output', '-o',
            dest='output', help="output image filename",
            metavar='OUTPUT', required = True)
    parser.add_argument('--version', '-V', action='version', version='%(prog)s 0.0')
    return parser

def plotCode(y):
    code = y[0, : , : , :]
    mincode = numpy.amin(code)
    maxcode = numpy.amax(code)
    print('Min: ', mincode, 'Max: ', maxcode)
    code = (code - mincode) / (maxcode - mincode)
    sh = numpy.shape(code)
    W = sh[0]
    H = sh[1]
    out = numpy.zeros((5*W+4*2, 3*H+2*2))

    for w in range(0,5):
        for h in range(0,3):
            c = w*3 + h
            out[w*(W+2) : (w+1)*(W+2)-2, h*(H+2):(h+1)*(H+2)-2] = code[:,:,c]

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
    print('Restoring model ...')
    saver = tf.train.import_meta_graph(options.weights + '.meta')
    saver.restore(sess, tf.train.latest_checkpoint('./'))
    print('Done.')
    graph = tf.get_default_graph()
    x = graph.get_tensor_by_name('Input:0')
    y = graph.get_tensor_by_name('Code:0')
    x_ = graph.get_tensor_by_name('reconstruct:0')
    y_ = tf.placeholder(tf.float32, [None, None, None], name='y_')
    loss = tf.reduce_mean(tf.square(x_[:,:,:,0] - y_), name="loss")
    sparse = tf.reduce_mean(tf.abs(y))

    image = cv2.imread(options.input, 0) / 255.

    H = image.shape[0]
    W = image.shape[1]

    H1 = int(numpy.ceil(H/32))*32
    W1 = int(numpy.ceil(W/32))*32

    ratio = max(H1/H, W1/W)

    im = cv2.resize(image, (int(W * ratio) + 1, int(H * ratio) + 1))
    im = im[0:H1, 0:W1]

    batch_xs = [im]

    print(im.shape)

    code, rec, ls, sp = sess.run([y, x_, loss, sparse],
        feed_dict={x: batch_xs, y_: batch_xs})

    print(numpy.sqrt(ls), sp)



    rec = rec[0]
    rec = rec[:,:,0]

    out = numpy.clip(rec, 0., 1.)
    dif = ((out - im)+1.0)/2.0
    mapcode = plotCode(code)

    cv2.imwrite(options.output, out * 255.)
    cv2.imwrite('dif-'+options.output, dif * 255.)
    cv2.imwrite('code-'+options.output, mapcode * 255.)

if __name__ == '__main__':
    main()
