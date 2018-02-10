import argparse
import os
import pickle
import time

import tensorflow as tf

from model import Model
from utils import DataLoader


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--rnn_size', type=int, default=256,
                        help='size of RNN hidden state')
    parser.add_argument('--num_layers', type=int, default=2,
                        help='number of layers in the RNN')
    parser.add_argument('--model', type=str, default='lstm',
                        help='rnn, gru, or lstm')
    parser.add_argument('--batch_size', type=int, default=50,
                        help='minibatch size')
    parser.add_argument('--seq_length', type=int, default=300,
                        help='RNN sequence length')
    parser.add_argument('--num_epochs', type=int, default=30,
                        help='number of epochs')
    parser.add_argument('--save_every', type=int, default=500,
                        help='save frequency')
    parser.add_argument('--model_dir', type=str, default='save',
                        help='directory to save model to')
    parser.add_argument('--grad_clip', type=float, default=10.,
                        help='clip gradients at this value')
    parser.add_argument('--learning_rate', type=float, default=0.005,
                        help='learning rate')
    parser.add_argument('--decay_rate', type=float, default=0.95,
                        help='decay rate for rmsprop')
    parser.add_argument('--num_mixture', type=int, default=20,
                        help='number of gaussian mixtures')
    parser.add_argument('--data_scale', type=float, default=20,
                        help='factor to scale raw data down by')
    parser.add_argument('--keep_prob', type=float, default=0.8,
                        help='dropout keep probability')
    args = parser.parse_args()
    train(args)


def train(args):
    data_loader = DataLoader(args.batch_size, args.seq_length, args.data_scale)

    if args.model_dir != '' and not os.path.exists(args.model_dir):
        os.makedirs(args.model_dir)

    with open(os.path.join(args.model_dir, 'config.pkl'), 'wb') as f:
        pickle.dump(args, f)

    model = Model(args)

    with tf.Session() as sess:
        summary_writer = tf.summary.FileWriter(
            os.path.join(args.model_dir, 'log'), sess.graph)

        tf.global_variables_initializer().run()
        saver = tf.train.Saver(tf.global_variables())
        for e in range(args.num_epochs):
            sess.run(tf.assign(model.lr,
                               args.learning_rate * (args.decay_rate ** e)))
            data_loader.reset_batch_pointer()
            v_x, v_y = data_loader.validation_data()
            valid_feed = {
                model.input_data: v_x,
                model.target_data: v_y,
                model.state_in: model.state_in.eval()}
            state = model.state_in.eval()
            for b in range(data_loader.num_batches):
                ith_train_step = e * data_loader.num_batches + b
                start = time.time()
                x, y = data_loader.next_batch()
                feed = {
                    model.input_data: x,
                    model.target_data: y,
                    model.state_in: state}
                train_loss_summary, train_loss, state, _ = sess.run(
                    [model.train_loss_summary, model.cost, model.state_out, model.train_op], feed)
                summary_writer.add_summary(train_loss_summary, ith_train_step)

                valid_loss_summary, valid_loss, = sess.run(
                    [model.valid_loss_summary, model.cost], valid_feed)
                summary_writer.add_summary(valid_loss_summary, ith_train_step)

                end = time.time()
                print(
                    "{}/{} (epoch {}), train_loss = {:.3f}, valid_loss = {:.3f}, time/batch = {:.3f}"
                    .format(
                        ith_train_step,
                        args.num_epochs * data_loader.num_batches,
                        e,
                        train_loss,
                        valid_loss,
                        end - start))
                if (ith_train_step %
                        args.save_every == 0) and (ith_train_step > 0):
                    checkpoint_path = os.path.join(
                        args.model_dir, 'model.ckpt')
                    saver.save(
                        sess,
                        checkpoint_path,
                        global_step=ith_train_step)
                    print("model saved to {}".format(checkpoint_path))


if __name__ == '__main__':
    main()
