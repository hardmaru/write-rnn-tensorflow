import random

import numpy as np
import tensorflow as tf


class Model():
    def __init__(self, args, infer=False):
        self.args = args
        if infer:
            args.batch_size = 1
            args.seq_length = 1

        if args.model == 'rnn':
            cell_fn = tf.contrib.rnn.BasicRNNCell
        elif args.model == 'gru':
            cell_fn = tf.contrib.rnn.GRUCell
        elif args.model == 'lstm':
            cell_fn = tf.contrib.rnn.BasicLSTMCell
        else:
            raise Exception("model type not supported: {}".format(args.model))

        def get_cell():
            return cell_fn(args.rnn_size, state_is_tuple=False)

        cell = tf.contrib.rnn.MultiRNNCell(
            [get_cell() for _ in range(args.num_layers)])

        if (infer == False and args.keep_prob < 1):  # training mode
            cell = tf.contrib.rnn.DropoutWrapper(
                cell, output_keep_prob=args.keep_prob)

        self.cell = cell

        self.input_data = tf.placeholder(
            dtype=tf.float32, shape=[
                None, args.seq_length, 3], name='data_in')
        self.target_data = tf.placeholder(
            dtype=tf.float32, shape=[
                None, args.seq_length, 3], name='targets')
        zero_state = cell.zero_state(
            batch_size=args.batch_size, dtype=tf.float32)
        self.state_in = tf.identity(zero_state, name='state_in')

        self.num_mixture = args.num_mixture
        # end_of_stroke + prob + 2*(mu + sig) + corr
        NOUT = 1 + self.num_mixture * 6

        with tf.variable_scope('rnnlm'):
            output_w = tf.get_variable("output_w", [args.rnn_size, NOUT])
            output_b = tf.get_variable("output_b", [NOUT])

        # inputs = tf.split(axis=1, num_or_size_splits=args.seq_length, value=self.input_data)
        # inputs = [tf.squeeze(input_, [1]) for input_ in inputs]
        inputs = tf.unstack(self.input_data, axis=1)

        # outputs, state_out = tf.contrib.legacy_seq2seq.rnn_decoder(inputs, self.state_in, cell, loop_function=None, scope='rnnlm')
        outputs, state_out = tf.contrib.legacy_seq2seq.rnn_decoder(
            inputs, zero_state, cell, loop_function=None, scope='rnnlm')

        output = tf.reshape(
            tf.concat(axis=1, values=outputs), [-1, args.rnn_size])
        output = tf.nn.xw_plus_b(output, output_w, output_b)
        self.state_out = tf.identity(state_out, name='state_out')

        # reshape target data so that it is compatible with prediction shape
        flat_target_data = tf.reshape(self.target_data, [-1, 3])
        [x1_data, x2_data, eos_data] = tf.split(
            axis=1, num_or_size_splits=3, value=flat_target_data)

        # long method:
        #flat_target_data = tf.split(1, args.seq_length, self.target_data)
        #flat_target_data = [tf.squeeze(flat_target_data_, [1]) for flat_target_data_ in flat_target_data]
        #flat_target_data = tf.reshape(tf.concat(1, flat_target_data), [-1, 3])

        def tf_2d_normal(x1, x2, mu1, mu2, s1, s2, rho):
            # eq # 24 and 25 of http://arxiv.org/abs/1308.0850
            norm1 = tf.subtract(x1, mu1)
            norm2 = tf.subtract(x2, mu2)
            s1s2 = tf.multiply(s1, s2)
            z = tf.square(tf.div(norm1, s1)) + tf.square(tf.div(norm2, s2)) - \
                2 * tf.div(tf.multiply(rho, tf.multiply(norm1, norm2)), s1s2)
            negRho = 1 - tf.square(rho)
            result = tf.exp(tf.div(-z, 2 * negRho))
            denom = 2 * np.pi * tf.multiply(s1s2, tf.sqrt(negRho))
            result = tf.div(result, denom)
            return result

        def get_lossfunc(
                z_pi,
                z_mu1,
                z_mu2,
                z_sigma1,
                z_sigma2,
                z_corr,
                z_eos,
                x1_data,
                x2_data,
                eos_data):
            result0 = tf_2d_normal(
                x1_data,
                x2_data,
                z_mu1,
                z_mu2,
                z_sigma1,
                z_sigma2,
                z_corr)
            # implementing eq # 26 of http://arxiv.org/abs/1308.0850
            epsilon = 1e-20
            result1 = tf.multiply(result0, z_pi)
            result1 = tf.reduce_sum(result1, 1, keep_dims=True)
            # at the beginning, some errors are exactly zero.
            result1 = -tf.log(tf.maximum(result1, 1e-20))

            result2 = tf.multiply(z_eos, eos_data) + \
                tf.multiply(1 - z_eos, 1 - eos_data)
            result2 = -tf.log(result2)

            result = result1 + result2
            return tf.reduce_sum(result)

        # below is where we need to do MDN splitting of distribution params
        def get_mixture_coef(output):
            # returns the tf slices containing mdn dist params
            # ie, eq 18 -> 23 of http://arxiv.org/abs/1308.0850
            z = output
            z_eos = z[:, 0:1]
            z_pi, z_mu1, z_mu2, z_sigma1, z_sigma2, z_corr = tf.split(
                axis=1, num_or_size_splits=6, value=z[:, 1:])

            # process output z's into MDN paramters

            # end of stroke signal
            z_eos = tf.sigmoid(z_eos)  # should be negated, but doesn't matter.

            # softmax all the pi's:
            max_pi = tf.reduce_max(z_pi, 1, keep_dims=True)
            z_pi = tf.subtract(z_pi, max_pi)
            z_pi = tf.exp(z_pi)
            normalize_pi = tf.reciprocal(
                tf.reduce_sum(z_pi, 1, keep_dims=True))
            z_pi = tf.multiply(normalize_pi, z_pi)

            # exponentiate the sigmas and also make corr between -1 and 1.
            z_sigma1 = tf.exp(z_sigma1)
            z_sigma2 = tf.exp(z_sigma2)
            z_corr = tf.tanh(z_corr)

            return [z_pi, z_mu1, z_mu2, z_sigma1, z_sigma2, z_corr, z_eos]

        [o_pi, o_mu1, o_mu2, o_sigma1, o_sigma2,
            o_corr, o_eos] = get_mixture_coef(output)

        # I could put all of these in a single tensor for reading out, but this
        # is more human readable
        data_out_pi = tf.identity(o_pi, "data_out_pi")
        data_out_mu1 = tf.identity(o_mu1, "data_out_mu1")
        data_out_mu2 = tf.identity(o_mu2, "data_out_mu2")
        data_out_sigma1 = tf.identity(o_sigma1, "data_out_sigma1")
        data_out_sigma2 = tf.identity(o_sigma2, "data_out_sigma2")
        data_out_corr = tf.identity(o_corr, "data_out_corr")
        data_out_eos = tf.identity(o_eos, "data_out_eos")

        # sticking them all (except eos) in one op anyway, makes it easier for freezing the graph later
        # IMPORTANT, this needs to stack the named ops above (data_out_XXX), not the prev ops (o_XXX)
        # otherwise when I freeze the graph up to this point, the named versions will be cut
        # eos is diff size to others, so excluding that
        data_out_mdn = tf.identity([data_out_pi,
                                    data_out_mu1,
                                    data_out_mu2,
                                    data_out_sigma1,
                                    data_out_sigma2,
                                    data_out_corr],
                                   name="data_out_mdn")

        self.pi = o_pi
        self.mu1 = o_mu1
        self.mu2 = o_mu2
        self.sigma1 = o_sigma1
        self.sigma2 = o_sigma2
        self.corr = o_corr
        self.eos = o_eos

        lossfunc = get_lossfunc(
            o_pi,
            o_mu1,
            o_mu2,
            o_sigma1,
            o_sigma2,
            o_corr,
            o_eos,
            x1_data,
            x2_data,
            eos_data)
        self.cost = lossfunc / (args.batch_size * args.seq_length)

        self.train_loss_summary = tf.summary.scalar('train_loss', self.cost)
        self.valid_loss_summary = tf.summary.scalar(
            'validation_loss', self.cost)

        self.lr = tf.Variable(0.0, trainable=False)
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(
            tf.gradients(self.cost, tvars), args.grad_clip)
        optimizer = tf.train.AdamOptimizer(self.lr)
        self.train_op = optimizer.apply_gradients(zip(grads, tvars))

    def sample(self, sess, num=1200):

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
        prev_state = sess.run(self.cell.zero_state(1, tf.float32))

        strokes = np.zeros((num, 3), dtype=np.float32)
        mixture_params = []

        for i in range(num):

            feed = {self.input_data: prev_x, self.state_in: prev_state}

            [o_pi,
             o_mu1,
             o_mu2,
             o_sigma1,
             o_sigma2,
             o_corr,
             o_eos,
             next_state] = sess.run([self.pi,
                                     self.mu1,
                                     self.mu2,
                                     self.sigma1,
                                     self.sigma2,
                                     self.corr,
                                     self.eos,
                                     self.state_out],
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

        strokes[:, 0:2] *= self.args.data_scale
        return strokes, mixture_params
