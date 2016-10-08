import numpy as np
import tensorflow as tf

import time
import os
import pickle
import argparse

from utils import *
from model import Model
import random


import svgwrite
from IPython.display import SVG, display

# main code (not in a main function since I want to run this script in IPython as well).

parser = argparse.ArgumentParser()
parser.add_argument('--filename', type=str, default='sample',
                   help='filename of .svg file to output, without .svg')
parser.add_argument('--sample_length', type=int, default=800,
                   help='number of strokes to sample')
parser.add_argument('--scale_factor', type=int, default=10,
                   help='factor to scale down by for svg output.  smaller means bigger output')
sample_args = parser.parse_args()

with open(os.path.join('save', 'config.pkl'), 'rb') as f:
    saved_args = pickle.load(f)

model = Model(saved_args, True)
sess = tf.InteractiveSession()
saver = tf.train.Saver(tf.all_variables())

ckpt = tf.train.get_checkpoint_state('save')
print("loading model: ", ckpt.model_checkpoint_path)

saver.restore(sess, ckpt.model_checkpoint_path)

def sample_stroke():
  [strokes, params] = model.sample(sess, sample_args.sample_length)
  draw_strokes(strokes, factor=sample_args.scale_factor, svg_filename = sample_args.filename+'.normal.svg')
  draw_strokes_random_color(strokes, factor=sample_args.scale_factor, svg_filename = sample_args.filename+'.color.svg')
  draw_strokes_random_color(strokes, factor=sample_args.scale_factor, per_stroke_mode = False, svg_filename = sample_args.filename+'.multi_color.svg')
  draw_strokes_eos_weighted(strokes, params, factor=sample_args.scale_factor, svg_filename = sample_args.filename+'.eos_pdf.svg')
  draw_strokes_pdf(strokes, params, factor=sample_args.scale_factor, svg_filename = sample_args.filename+'.pdf.svg')
  return [strokes, params]

[strokes, params] = sample_stroke()


