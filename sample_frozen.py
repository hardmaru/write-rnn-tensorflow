# -*- coding: utf-8 -*-
"""
Created on Thu Feb 23 20:25:16 2017

@author: memo

demonstrates inference with frozen graph def
same as sample.py, but:
- instead of loading model + checkpoint, loads frozen graph
- instead of calling model.sample() function, uses own sample() function with named ops
"""

import argparse

import tensorflow as tf

from utils import *

# main code (not in a main function since I want to run this script in
# IPython as well).

parser = argparse.ArgumentParser()
parser.add_argument('--filename', type=str, default='sample',
                    help='filename of .svg file to output, without .svg')
parser.add_argument('--sample_length', type=int, default=800,
                    help='number of strokes to sample')
parser.add_argument(
    '--scale_factor',
    type=int,
    default=10,
    help='factor to scale down by for svg output.  smaller means bigger output')
parser.add_argument('--model_dir', type=str, default='save',
                    help='directory to save model to')
sample_args = parser.parse_args()

sess = tf.InteractiveSession()

# load frozen graph
from tensorflow.python.platform import gfile
with gfile.FastGFile(os.path.join(sample_args.model_dir, 'graph_frz.pb'), 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    sess.graph.as_default()
    tf.import_graph_def(graph_def, name='')


def sample_stroke():
    # don't  call model.sample(), instead call sample() function defined below
    [strokes, params] = sample(sess, sample_args.sample_length)
    draw_strokes(
        strokes,
        factor=sample_args.scale_factor,
        svg_filename=sample_args.filename +
        '.normal.svg')
    draw_strokes_random_color(
        strokes,
        factor=sample_args.scale_factor,
        svg_filename=sample_args.filename +
        '.color.svg')
    draw_strokes_random_color(
        strokes,
        factor=sample_args.scale_factor,
        per_stroke_mode=False,
        svg_filename=sample_args.filename +
        '.multi_color.svg')
    draw_strokes_eos_weighted(
        strokes,
        params,
        factor=sample_args.scale_factor,
        svg_filename=sample_args.filename +
        '.eos_pdf.svg')
    draw_strokes_pdf(
        strokes,
        params,
        factor=sample_args.scale_factor,
        svg_filename=sample_args.filename +
        '.pdf.svg')
    return [strokes, params]


# copied straight from model.sample, but replaced all referenes to 'self'
# with named ops
def sample(sess, num=1200):
    data_in = 'data_in:0'
    data_out_pi = 'data_out_pi:0'
    data_out_mu1 = 'data_out_mu1:0'
    data_out_mu2 = 'data_out_mu2:0'
    data_out_sigma1 = 'data_out_sigma1:0'
    data_out_sigma2 = 'data_out_sigma2:0'
    data_out_corr = 'data_out_corr:0'
    data_out_eos = 'data_out_eos:0'
    state_in = 'state_in:0'
    state_out = 'state_out:0'

    def get_pi_idx(x, pdf):
        N = pdf.size
        accumulate = 0
        for i in range(0, N):
            accumulate += pdf[i]
            if (accumulate >= x):
                return i
        print('error with sampling ensemble')
        return -1

    def sample_gaussian_2d(mu1, mu2, s1, s2, rho):
        mean = [mu1, mu2]
        cov = [[s1 * s1, rho * s1 * s2], [rho * s1 * s2, s2 * s2]]
        x = np.random.multivariate_normal(mean, cov, 1)
        return x[0][0], x[0][1]

    prev_x = np.zeros((1, 1, 3), dtype=np.float32)
    prev_x[0, 0, 2] = 1  # initially, we want to see beginning of new stroke
    prev_state = sess.run(state_in)

    strokes = np.zeros((num, 3), dtype=np.float32)
    mixture_params = []

    for i in range(num):

        feed = {data_in: prev_x, state_in: prev_state}

        [o_pi,
         o_mu1,
         o_mu2,
         o_sigma1,
         o_sigma2,
         o_corr,
         o_eos,
         next_state] = sess.run([data_out_pi,
                                 data_out_mu1,
                                 data_out_mu2,
                                 data_out_sigma1,
                                 data_out_sigma2,
                                 data_out_corr,
                                 data_out_eos,
                                 state_out],
                                feed)

        idx = get_pi_idx(random.random(), o_pi[0])

        eos = 1 if random.random() < o_eos[0][0] else 0

        next_x1, next_x2 = sample_gaussian_2d(
            o_mu1[0][idx], o_mu2[0][idx], o_sigma1[0][idx], o_sigma2[0][idx], o_corr[0][idx])

        strokes[i, :] = [next_x1, next_x2, eos]

        params = [
            o_pi[0],
            o_mu1[0],
            o_mu2[0],
            o_sigma1[0],
            o_sigma2[0],
            o_corr[0],
            o_eos[0]]
        mixture_params.append(params)

        prev_x = np.zeros((1, 1, 3), dtype=np.float32)
        prev_x[0][0] = np.array([next_x1, next_x2, eos], dtype=np.float32)
        prev_state = next_state

    # self.args.data_scale # TODO: fix mega hack hardcoding the scale
    strokes[:, 0:2] *= 20
    return strokes, mixture_params


# check output
[strokes, params] = sample_stroke()
