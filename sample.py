import argparse

import tensorflow as tf

from model import Model
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
parser.add_argument(
    '--freeze_graph',
    dest='freeze_graph',
    action='store_true',
    help='if true, freeze (replace variables with consts), prune (for inference) and save graph')

sample_args = parser.parse_args()

with open(os.path.join(sample_args.model_dir, 'config.pkl'), 'rb') as f:
    saved_args = pickle.load(f)

model = Model(saved_args, True)
sess = tf.InteractiveSession()
#saver = tf.train.Saver(tf.all_variables())
saver = tf.train.Saver()

ckpt = tf.train.get_checkpoint_state(sample_args.model_dir)
print("loading model: ", ckpt.model_checkpoint_path)

saver.restore(sess, ckpt.model_checkpoint_path)


def sample_stroke():
    [strokes, params] = model.sample(sess, sample_args.sample_length)
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


def freeze_and_save_graph(sess, folder, out_nodes, as_text=False):
    # save graph definition
    graph_raw = sess.graph_def
    graph_frz = tf.graph_util.convert_variables_to_constants(
        sess, graph_raw, out_nodes)
    ext = '.txt' if as_text else '.pb'
    #tf.train.write_graph(graph_raw, folder, 'graph_raw'+ext, as_text=as_text)
    tf.train.write_graph(graph_frz, folder, 'graph_frz' + ext, as_text=as_text)


if(sample_args.freeze_graph):
    freeze_and_save_graph(
        sess, sample_args.model_dir, [
            'data_out_mdn', 'data_out_eos', 'state_out'], False)

[strokes, params] = sample_stroke()
