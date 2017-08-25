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
    'generate_ims'
    ]

import itertools
import os, errno
import random
import glob
import cv2
import numpy as np
from argparse import ArgumentParser

# default arguments
ITERATIONS = 1000
WIDTH = 32*13
HEIGHT = 32*13
OUTPUT = './test'

# for command line interpreter
def build_parser():
    parser = ArgumentParser()
    parser.add_argument('--background', '-b',
            dest='bg', help='background images folder',
            metavar='BG', required=True)
    parser.add_argument('--output', '-o',
            dest='output', help="output path (default '%(default)s')",
            metavar='OUTPUT', default=OUTPUT)
    parser.add_argument('--iterations','-i', type=int,
            dest='iterations', help='iterations (default %(default)s)',
            metavar='ITERATIONS', default=ITERATIONS)
    parser.add_argument('--width', '-w', type=int,
            dest='width', help='output width (default %(default)s)',
            metavar='WIDTH', default=WIDTH)
    parser.add_argument('--height', '-a', type=int,
            dest='height', help='output height (default %(default)s)',
            metavar='HEIGHT', default=HEIGHT)
    parser.add_argument('--version', '-V', action='version', version='%(prog)s 0.0')
    return parser

def generate_bg(re_files, shape):
    found = False
    nb_files = len(re_files)

    while not found:
        fname = re_files[random.randint(0, nb_files - 1)]
        bg = cv2.imread(fname, 0) / 255. # cv2.CV_LOAD_IMAGE_GRAYSCALE is 0
        ratio = max(shape[1] / bg.shape[1], shape[0] / bg.shape[0])
        bg = cv2.resize(bg, (int(bg.shape[1] * ratio) + 1, int(bg.shape[0] * ratio) + 1))

        if (bg.shape[1] >= shape[1] and
            bg.shape[0] >= shape[0]):
            found = True

    x = random.randint(0, bg.shape[1] - shape[1])
    y = random.randint(0, bg.shape[0] - shape[0])
    bg = bg[y:y + shape[0], x:x + shape[1]]
    return bg

def generate_im(re_files, shape):
    bg = generate_bg(re_files, shape)
    out = bg
    out = cv2.resize(out, (shape[1], shape[0]))
    out = np.clip(out, 0., 1.)
    return out

def generate_ims(shape, bg_dir):
    """
    Generate images.
    :return:
        Iterable of images.
    """
    in_files = os.path.join(bg_dir, '*.jpg')
    re_files = glob.glob(in_files)
    print('Number of base files: ', len(re_files))
    while True:
        yield generate_im(re_files, shape)

def main():
    parser = build_parser()
    options = parser.parse_args()
    shape = (options.height, options.width)

    # check or create output directory
    try:
        os.makedirs(options.output)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

    im_gen = itertools.islice(generate_ims(shape, options.bg), options.iterations)

    for img_idx, im in enumerate(im_gen):
        fname = "{}/{:08d}.png".format(options.output, img_idx)
        print (fname)
        cv2.imwrite(fname, im * 255.)

if __name__ == '__main__':
    main()
