"""Deep Recurrent Attentive Writer.
"""
"""
Copyright 2017 Parag K. Mital.  See also NOTICE.md.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
import matplotlib.pyplot as plt
import tensorflow as tf
from cadl.datasets import MNIST, CIFAR10
from cadl.dataset_utils import create_input_pipeline
from cadl import utils, gif
import numpy as np


def linear(x, n_output):
    """Summary

    Parameters
    ----------
    x : TYPE
        Description
    n_output : TYPE
        Description

    Returns
    -------
    TYPE
        Description
    """
    w = tf.get_variable(
        "w", [x.get_shape()[1], n_output],
        initializer=tf.contrib.layers.xavier_initializer())
    b = tf.get_variable(
        "b", [n_output], initializer=tf.constant_initializer(0.0))
    return tf.add(tf.matmul(x, w), b)


def encoder(x, rnn, batch_size, state=None, n_enc=64, reuse=None):
    """Summary

    Parameters
    ----------
    x : TYPE
        Description
    rnn : TYPE
        Description
    batch_size : TYPE
        Description
    state : None, optional
        Description
    n_enc : int, optional
        Description
    reuse : None, optional
        Description

    Returns
    -------
    name : TYPE
        Description
    """
    with tf.variable_scope('encoder', reuse=reuse):
        if state is None:
            h_enc, state = rnn(x, rnn.zero_state(batch_size, tf.float32))
        else:
            h_enc, state = rnn(x, state)
    return h_enc, state


def variational_layer(h_enc, noise, n_z=2, reuse=None):
    """Summary

    Parameters
    ----------
    h_enc : TYPE
        Description
    noise : TYPE
        Description
    n_z : int, optional
        Description
    reuse : None, optional
        Description

    Returns
    -------
    name : TYPE
        Description
    """
    with tf.variable_scope('variational', reuse=reuse):
        # Equation 1: use the encoder to parameterize the mean of the approximate
        # posterior distribution Q
        with tf.variable_scope('mu', reuse=reuse):
            h_z_mu = linear(h_enc, n_z)

        # Equation 2: Similarly for the standard deviation
        with tf.variable_scope('log_sigma', reuse=reuse):
            h_z_log_sigma = linear(h_enc, n_z)

        # sample z_t ~ q(Z_t | h_enc_t)
        z_t = h_z_mu + tf.multiply(tf.exp(h_z_log_sigma), noise)

    # return the sampled value from the latent distribution and its parameters
    return z_t, h_z_mu, h_z_log_sigma


def decoder(z, rnn, batch_size, state=None, n_dec=64, reuse=None):
    """Summary

    Parameters
    ----------
    z : TYPE
        Description
    rnn : TYPE
        Description
    batch_size : TYPE
        Description
    state : None, optional
        Description
    n_dec : int, optional
        Description
    reuse : None, optional
        Description

    Returns
    -------
    name : TYPE
        Description
    """
    with tf.variable_scope('decoder', reuse=reuse):
        if state is None:
            h_dec, state = rnn(z, rnn.zero_state(batch_size, tf.float32))
        else:
            h_dec, state = rnn(z, state)
    return h_dec, state


def create_attention_map(h_dec, reuse=None):
    """Summary

    Parameters
    ----------
    h_dec : TYPE
        Description
    reuse : None, optional
        Description

    Returns
    -------
    name : TYPE
        Description
    """
    with tf.variable_scope("attention", reuse=reuse):
        p = linear(h_dec, 5)
        g_tilde_x, g_tilde_y, log_sigma, log_delta_tilde, log_gamma = \
            tf.split(p, 5, axis=1)

    return g_tilde_x, g_tilde_y, log_sigma, log_delta_tilde, log_gamma


def create_filterbank(g_x, g_y, log_sigma_sq, log_delta, A=28, B=28, C=1, N=12):
    """summary

    Parameters
    ----------
    g_x : TYPE
        Description
    g_y : TYPE
        Description
    log_sigma_sq : TYPE
        Description
    log_delta : TYPE
        Description
    A : int, optional
        Description
    B : int, optional
        Description
    C : int, optional
        Description
    N : int, optional
        Description

    Returns
    -------
    name : TYPE
        Description

    Deleted Parameters
    ------------------
    log_sigma : type
        description
    """
    with tf.name_scope("filterbank"):
        # Equation 22 and 23
        g_x = (A + 1) / 2 * (g_x + 1)
        g_y = (B + 1) / 2 * (g_y + 1)

        # The authors suggest to use a real-valued center and stride, meaning
        # the center of this grid is not necessarily located directly on a
        # pixel, but can be between pixels. To compute the stride, we use
        # equation 24:

        # Equation 24 delta = (max(A, B) - 1) / (N - 1) * tf.exp(log_delta)
        delta = (max(A, B) - 1) / (N - 1) * tf.exp(log_delta)
        # Note that we've converted our `log_delta` to `delta` by taking the
        # exponential.

        # To determine the mean location of the ith and jth filter of the N x N
        # grid of filters, we can use the formulas from the paper, equations 19
        # and 20.  We'll create grid positions for the x and y positions
        # independently.  So for each observation in our mini batch, we'll have
        # N number of positions for our x and our y grid positions, or 12 x 12
        # = 144 grid positions in total for each observation in our mini batch.

        # Equations 19 and 20
        ns = tf.expand_dims(tf.cast(tf.range(N), tf.float32), 0)
        mu_x = tf.reshape(g_x + (ns - N / 2 - 0.5) * delta, [-1, N, 1])
        mu_y = tf.reshape(g_y + (ns - N / 2 - 0.5) * delta, [-1, N, 1])

        # Finally we're ready to define the filterbank matrices `F_x` and `F_y`
        # from equations 25 and 26.  `F_x` and `F_x` require us to use $2 *
        # \sigma^2$.  So we'll calculate that first for each of our sigmas, one
        # per observation in our mini batch.  We take exponential of
        # `log_sigma` to get $\sigma^2$ and then multiply by 2.  we'll also
        # reshape it to the number of observations we have in the first
        # dimension, and create singleton dimensions for broadcasting them
        # across our filterbanks.
        sigma_sq = tf.reshape(tf.exp(log_sigma_sq), [-1, 1, 1])

        # Now we'll create a range for our entire image:
        xs = tf.reshape(tf.cast(tf.range(B), tf.float32), [1, 1, -1])
        ys = tf.reshape(tf.cast(tf.range(A), tf.float32), [1, 1, -1])

        # And then using broadcasting, we can calculate the Gaussian defining
        # the filterbank:
        F_x = tf.exp(-tf.square(xs - mu_x) / (2 * sigma_sq))
        F_y = tf.exp(-tf.square(ys - mu_y) / (2 * sigma_sq))

        # Finally we'll normalize the filterbank across each location so that
        # the sum of the energy across the x and y locations sum to 1.  We'll
        # also ensure that we do not divide by zero by making sure the maximum
        # value is at least epsilon.  There will be one filterbank defining the
        # horizontal filters, and another for the vertical filters.  The
        # horizontal filterbanks, `F_x[i, a]` will be N x B, so N filters
        # across the B number of pixels.  Same for the vertical ones,
        # `F_y[j, b]`, there will be N filters across the A number of pixels.

        # Normalize
        epsilon = 1e-10
        F_x = F_x / tf.maximum(tf.reduce_sum(F_x, 2, keep_dims=True), epsilon)
        F_y = F_y / tf.maximum(tf.reduce_sum(F_y, 2, keep_dims=True), epsilon)

    # return the filterbanks
    return F_x, F_y


def filter_image(x, F_x, F_y, log_gamma, A, B, C, N, inverted=False):
    """Summary

    Parameters
    ----------
    x : TYPE
        Description
    F_x : TYPE
        Description
    F_y : TYPE
        Description
    log_gamma : TYPE
        Description
    A : TYPE
        Description
    B : TYPE
        Description
    C : TYPE
        Description
    N : TYPE
        Description
    inverted : bool, optional
        Description

    Returns
    -------
    name : TYPE
        Description

    Deleted Parameters
    ------------------
    gamma : TYPE
        Description
    """
    with tf.name_scope("filter"):
        # To filter the image, we'll want to transpose our filterbanks
        # dimensions allowing to to multiply the image in the next step
        # For the read operation, we transpose X (equation 27)
        # For write, we transpose Y, and use inverse gamma (equation 29)
        gamma = tf.exp(log_gamma)
        if inverted:
            F_y = tf.transpose(F_y, perm=[0, 2, 1])
            gamma = 1.0 / gamma
            # Now we left and right multiply the image in `x` by each filter
            if C == 1:
                glimpse = tf.matmul(F_y,
                                    tf.matmul(tf.reshape(x, [-1, N, N]), F_x))
            else:
                x = tf.reshape(x, [-1, N, N, C])
                xs = tf.split(x, C, axis=3)
                glimpses = []
                for x_i in xs:
                    glimpses.append(
                        tf.matmul(F_y, tf.matmul(tf.squeeze(x_i), F_x)))
                glimpse = tf.concat(
                    [tf.expand_dims(x_i, -1) for x_i in glimpses], axis=3)
        else:
            F_x = tf.transpose(F_x, perm=[0, 2, 1])
            # Now we left and right multiply the image in `x` by each filter
            if C == 1:
                glimpse = tf.matmul(F_y,
                                    tf.matmul(tf.reshape(x, [-1, A, B]), F_x))
            else:
                x = tf.reshape(x, [-1, A, B, C])
                xs = tf.split(x, C, axis=3)
                glimpses = []
                for x_i in xs:
                    glimpses.append(
                        tf.matmul(F_y, tf.matmul(tf.squeeze(x_i), F_x)))
                glimpse = tf.concat(
                    [tf.expand_dims(x_i, -1) for x_i in glimpses], axis=3)
        # Finally, we'll flatten the filtered image to a vector
        glimpse = tf.reshape(glimpse,
                             [-1, np.prod(glimpse.get_shape().as_list()[1:])])

        # And weight the filtered glimpses by gamma
        return glimpse * tf.reshape(gamma, [-1, 1])


def read(x_t,
         x_hat_t,
         h_dec_t,
         read_n=5,
         A=28,
         B=28,
         C=1,
         use_attention=True,
         reuse=None):
    """Read from the input image, `x`, and reconstruction error image `x_hat`.

    Optionally apply a filterbank w/ `use_attention`.

    Parameters
    ----------
    x_t : tf.Tensor
        Input image to optionally filter
    x_hat_t : tf.Tensor
        Reconstruction error to optionally filter
    h_dec_t : tf.Tensor
        Output of the decoder of the network (could also be the encoder but the
        authors suggest to use the decoder instead, see end of section 2.1)
    read_n : int, optional
        Description
    A : int, optional
        Description
    B : int, optional
        Description
    C : int, optional
        Description
    use_attention : bool, optional
        Description
    reuse : None, optional
        Description

    Returns
    -------
    TYPE
        Description
    """
    with tf.variable_scope('read', reuse=reuse):
        if use_attention:
            # Use the decoder's output to create 5 measures to define the
            # placement and characteristics of the filterbank
            g_x_tilde, g_y_tilde, \
                log_sigma_sq_tilde, log_delta_tilde, log_gamma_tilde = \
                create_attention_map(h_dec_t, reuse=reuse)

            # Now create the filterbank
            F_x_tilde, F_y_tilde = create_filterbank(
                g_x_tilde,
                g_y_tilde,
                log_sigma_sq_tilde,
                log_delta_tilde,
                N=read_n,
                A=A,
                B=B,
                C=C)

            # And apply the filterbanks to the input image
            x_t = filter_image(x_t, F_x_tilde, F_y_tilde, log_gamma_tilde, A, B,
                               C, read_n)

            # And similarly, apply the filterbanks to the error image
            x_hat_t = filter_image(x_hat_t, F_x_tilde, F_y_tilde,
                                   log_gamma_tilde, A, B, C, read_n)

    # Equation 27, concat the two N x N patches from the image and the error
    # image.  If we aren't using attention, these are just the unfiltered
    # images.
    return tf.concat([x_t, x_hat_t], axis=1)


def write(h_dec_t, write_n=5, A=28, B=28, C=1, use_attention=True, reuse=None):
    """Summary

    Parameters
    ----------
    h_dec_t : TYPE
        Description
    write_n : int, optional
        Description
    A : int, optional
        Description
    B : int, optional
        Description
    C : int, optional
        Description
    use_attention : bool, optional
        Description
    reuse : None, optional
        Description

    Returns
    -------
    name : TYPE
        Description
    """
    # Equation 28: again, like in the read layer, we can add an additional
    # nonlinearity here to enforce the characteristics of the final activation
    # we expect to see.  For instance, if our images are normalized 0 to 1,
    # then we can use a sigmoid activation.
    with tf.variable_scope("write", reuse=reuse):
        # Next, we'll want to apply a few more additional operations if we're
        # using attention
        if use_attention:
            w_t = linear(h_dec_t, write_n * write_n * C)
            if C == 1:
                w_t = tf.reshape(w_t, [-1, write_n, write_n])
            else:
                w_t = tf.reshape(w_t, [-1, write_n, write_n, C])

            # Use the decoder's output to create 5 measures to define the
            # placement and characteristics of the filterbank
            g_x_hat, g_y_hat, log_sigma_sq_hat, log_delta_hat, log_gamma_hat = \
                create_attention_map(h_dec_t, reuse=reuse)

            # Now create the filterbank
            F_x_hat, F_y_hat = create_filterbank(
                g_x_hat,
                g_y_hat,
                log_sigma_sq_hat,
                log_delta_hat,
                N=write_n,
                A=A,
                B=B,
                C=C)

            # And apply the filterbanks to the input image, Equation 29
            w_t = filter_image(
                w_t,
                F_x_hat,
                F_y_hat,
                log_gamma_hat,
                A,
                B,
                C,
                write_n,
                inverted=True)

            return w_t
        else:
            return linear(h_dec_t, A * B * C)


def binary_cross_entropy(t, o, eps=1e-10):
    """Summary

    Parameters
    ----------
    t : TYPE
        Description
    o : TYPE
        Description
    eps : float, optional
        Description

    Returns
    -------
    TYPE
        Description
    """
    return -(t * tf.log(o + eps) + (1.0 - t) * tf.log(1.0 - o + eps))


def create_model(
        A=28,  # img_h
        B=28,  # img_w
        C=1,  # img_c
        T=16,
        batch_size=100,
        n_enc=128,
        n_z=32,
        n_dec=128,
        read_n=12,
        write_n=12):
    """<FRESHLY_INSERTED>"""
    x = tf.placeholder(tf.float32, shape=[None, A * B * C], name='x')
    noise = tf.placeholder(tf.float32, shape=[None, n_z], name='noise')
    rnn_enc = tf.contrib.rnn.GRUCell(n_enc)
    rnn_dec = tf.contrib.rnn.GRUCell(n_dec)
    enc_state, dec_state = None, None

    canvas = [tf.zeros([batch_size, A * B * C], name='c_0')]
    h_enc_t = tf.zeros([batch_size, n_dec])
    h_dec_t = tf.zeros([batch_size, n_dec])

    reuse = False
    z_mus, z_log_sigmas = [], []
    for t_i in range(1, T):
        # This assumes the input image is normalized between 0 - 1
        x_hat_t = x - tf.nn.sigmoid(canvas[t_i - 1])
        r_t = read(
            x_t=x,
            x_hat_t=x_hat_t,
            h_dec_t=h_dec_t,
            read_n=read_n,
            A=A,
            B=B,
            C=C,
            use_attention=True,
            reuse=reuse)
        h_enc_t, enc_state = encoder(
            x=tf.concat([r_t, h_dec_t], axis=1),
            rnn=rnn_enc,
            batch_size=batch_size,
            state=enc_state,
            n_enc=n_enc,
            reuse=reuse)
        z_t, z_mu, z_log_sigma = variational_layer(
            h_enc=h_enc_t, noise=noise, n_z=n_z, reuse=reuse)

        z_mus.append(z_mu)
        z_log_sigmas.append(z_log_sigma)
        h_dec_t, dec_state = decoder(
            z=z_t,
            rnn=rnn_dec,
            batch_size=batch_size,
            state=dec_state,
            n_dec=n_dec,
            reuse=reuse)
        w_t = write(
            h_dec_t=h_dec_t,
            write_n=write_n,
            A=A,
            B=B,
            C=C,
            use_attention=True,
            reuse=reuse)
        c_t = canvas[-1] + w_t
        canvas.append(c_t)
        reuse = True

    x_recon = tf.nn.sigmoid(canvas[-1])
    with tf.variable_scope('loss'):
        loss_x = tf.reduce_mean(
            tf.reduce_sum(binary_cross_entropy(x, x_recon), 1))
        loss_zs = []
        for z_mu, z_log_sigma in zip(z_mus, z_log_sigmas):
            loss_zs.append(
                tf.reduce_sum(
                    tf.square(z_mu) + tf.square(tf.exp(z_log_sigma)) -
                    2 * z_log_sigma, 1))
        loss_z = tf.reduce_mean(0.5 * tf.reduce_sum(loss_zs, 0) - T * 0.5)
        cost = loss_x + loss_z

    return {
        'x': x,
        'loss_x': loss_x,
        'loss_z': loss_z,
        'canvas': [tf.nn.sigmoid(c_i) for c_i in canvas],
        'cost': cost,
        'recon': x_recon,
        'noise': noise
    }


def test_mnist():
    A = 28  # img_h
    B = 28  # img_w
    C = 1
    T = 10
    n_enc = 256
    n_z = 100
    n_dec = 256
    read_n = 5
    write_n = 5
    batch_size = 64
    mnist = MNIST(split=[0.8, 0.1, 0.1])

    n_examples = batch_size
    zs = np.random.uniform(-1.0, 1.0, [4, n_z]).astype(np.float32)
    zs = utils.make_latent_manifold(zs, n_examples)

    # We create a session to use the graph
    g = tf.Graph()
    with tf.Session(graph=g) as sess:
        draw = create_model(
            A=A,
            B=B,
            C=C,
            T=T,
            batch_size=batch_size,
            n_enc=n_enc,
            n_z=n_z,
            n_dec=n_dec,
            read_n=read_n,
            write_n=write_n)
        opt = tf.train.AdamOptimizer(learning_rate=0.0001)
        grads = opt.compute_gradients(draw['cost'])
        for i, (g, v) in enumerate(grads):
            if g is not None:
                grads[i] = (tf.clip_by_norm(g, 5), v)
        train_op = opt.apply_gradients(grads)
        init_op = tf.group(tf.global_variables_initializer(),
                           tf.local_variables_initializer())
        sess.run(init_op)
        saver = tf.train.Saver()

        # Fit all training data
        batch_i = 0
        n_epochs = 100
        test_xs = mnist.test.images[:n_examples]
        utils.montage(test_xs.reshape((-1, A, B)), 'test_xs.png')
        for epoch_i in range(n_epochs):
            for batch_xs, _ in mnist.train.next_batch(batch_size):
                noise = np.random.randn(batch_size, n_z)
                lx, lz = sess.run(
                    [draw['loss_x'], draw['loss_z'], train_op],
                    feed_dict={draw['x']: batch_xs,
                               draw['noise']: noise})[0:2]
                print('x:', lx, 'z:', lz)
                if batch_i % 1000 == 0:
                    # Plot example reconstructions
                    recon = sess.run(
                        draw['canvas'],
                        feed_dict={draw['x']: test_xs,
                                   draw['noise']: noise})
                    recon = [utils.montage(r.reshape(-1, A, B)) for r in recon]
                    gif.build_gif(
                        recon,
                        cmap='gray',
                        saveto='manifold_%08d.gif' % batch_i)

                    saver.save(sess, './draw.ckpt', global_step=batch_i)

                batch_i += 1


def train_dataset(ds,
                  A,
                  B,
                  C,
                  T=20,
                  n_enc=512,
                  n_z=200,
                  n_dec=512,
                  read_n=12,
                  write_n=12,
                  batch_size=100,
                  n_epochs=100):

    if ds is None:
        ds = CIFAR10(split=[0.8, 0.1, 0.1])
        A, B, C = (32, 32, 3)

    n_examples = batch_size
    zs = np.random.uniform(-1.0, 1.0, [4, n_z]).astype(np.float32)
    zs = utils.make_latent_manifold(zs, n_examples)

    # We create a session to use the graph
    g = tf.Graph()
    with tf.Session(graph=g) as sess:
        draw = create_model(
            A=A,
            B=B,
            C=C,
            T=T,
            batch_size=batch_size,
            n_enc=n_enc,
            n_z=n_z,
            n_dec=n_dec,
            read_n=read_n,
            write_n=write_n)
        opt = tf.train.AdamOptimizer(learning_rate=0.0001)

        # Clip gradients
        grads = opt.compute_gradients(draw['cost'])
        for i, (g, v) in enumerate(grads):
            if g is not None:
                grads[i] = (tf.clip_by_norm(g, 5), v)
        train_op = opt.apply_gradients(grads)

        # Add summary variables
        tf.summary.scalar(name='cost', tensor=draw['cost'])
        tf.summary.scalar(name='loss_z', tensor=draw['loss_z'])
        tf.summary.scalar(name='loss_x', tensor=draw['loss_x'])
        tf.summary.histogram(
            name='recon_t0_histogram', values=draw['canvas'][0])
        tf.summary.histogram(
            name='recon_t-1_histogram', values=draw['canvas'][-1])
        tf.summary.image(
            name='recon_t0_image',
            tensor=tf.reshape(draw['canvas'][0], (-1, A, B, C)),
            max_outputs=2)
        tf.summary.image(
            name='recon_t-1_image',
            tensor=tf.reshape(draw['canvas'][-1], (-1, A, B, C)),
            max_outputs=2)
        sums = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(logdir='draw/train')
        valid_writer = tf.summary.FileWriter(logdir='draw/valid')

        # Init
        init_op = tf.group(tf.global_variables_initializer(),
                           tf.local_variables_initializer())
        sess.run(init_op)
        saver = tf.train.Saver()

        # Fit all training data
        batch_i = 0
        test_xs = ds.test.images[:n_examples] / 255.0
        utils.montage(test_xs.reshape((-1, A, B, C)), 'draw/test_xs.png')
        for epoch_i in range(n_epochs):
            for batch_xs, _ in ds.train.next_batch(batch_size):
                noise = np.random.randn(batch_size, n_z)
                cost, summary = sess.run(
                    [draw['cost'], sums, train_op],
                    feed_dict={
                        draw['x']: batch_xs / 255.0,
                        draw['noise']: noise
                    })[0:2]
                train_writer.add_summary(summary, batch_i)
                print('train cost:', cost)

                if batch_i % 1000 == 0:
                    # Plot example reconstructions
                    recon = sess.run(
                        draw['canvas'],
                        feed_dict={draw['x']: test_xs,
                                   draw['noise']: noise})
                    recon = [
                        utils.montage(r.reshape(-1, A, B, C)) for r in recon
                    ]
                    gif.build_gif(
                        recon,
                        cmap='gray',
                        saveto='draw/manifold_%08d.gif' % batch_i)
                    saver.save(sess, './draw/draw.ckpt', global_step=batch_i)
                batch_i += 1

            # Run validation
            if batch_i % 1000 == 0:
                for batch_xs, _ in ds.valid.next_batch(batch_size):
                    noise = np.random.randn(batch_size, n_z)
                    cost, summary = sess.run(
                        [draw['cost'], sums],
                        feed_dict={
                            draw['x']: batch_xs / 255.0,
                            draw['noise']: noise
                        })[0:2]
                    valid_writer.add_summary(summary, batch_i)
                    print('valid cost:', cost)
                    batch_i += 1


def train_input_pipeline(
        files,
        A,  # img_h
        B,  # img_w
        C,
        T=20,
        n_enc=512,
        n_z=256,
        n_dec=512,
        read_n=15,
        write_n=15,
        batch_size=64,
        n_epochs=1e9,
        input_shape=(64, 64, 3)):

    # We create a session to use the graph
    g = tf.Graph()
    with tf.Session(graph=g) as sess:
        batch = create_input_pipeline(
            files=files,
            batch_size=batch_size,
            n_epochs=n_epochs,
            crop_shape=(A, B, C),
            shape=input_shape)

        draw = create_model(
            A=A,
            B=B,
            C=C,
            T=T,
            batch_size=batch_size,
            n_enc=n_enc,
            n_z=n_z,
            n_dec=n_dec,
            read_n=read_n,
            write_n=write_n)
        opt = tf.train.AdamOptimizer(learning_rate=0.0001)
        grads = opt.compute_gradients(draw['cost'])
        for i, (g, v) in enumerate(grads):
            if g is not None:
                grads[i] = (tf.clip_by_norm(g, 5), v)
        train_op = opt.apply_gradients(grads)
        init_op = tf.group(tf.global_variables_initializer(),
                           tf.local_variables_initializer())
        sess.run(init_op)
        coord = tf.train.Coordinator()
        tf.get_default_graph().finalize()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        # Fit all training data
        batch_i = 0
        epoch_i = 0
        n_files = len(files)
        test_xs = sess.run(batch).reshape((-1, A * B * C)) / 255.0
        utils.montage(test_xs.reshape((-1, A, B, C)), 'test_xs.png')
        try:
            while not coord.should_stop() and epoch_i < n_epochs:
                batch_xs = sess.run(batch) / 255.0
                noise = np.random.randn(batch_size, n_z)
                lx, lz = sess.run(
                    [draw['loss_x'], draw['loss_z'], train_op],
                    feed_dict={
                        draw['x']: batch_xs.reshape((-1, A * B * C)) / 255.0,
                        draw['noise']: noise
                    })[0:2]
                print('x:', lx, 'z:', lz)
                if batch_i % n_files == 0:
                    batch_i = 0
                    epoch_i += 1
                if batch_i % 1000 == 0:
                    # Plot example reconstructions
                    recon = sess.run(
                        draw['canvas'],
                        feed_dict={draw['x']: test_xs,
                                   draw['noise']: noise})
                    recon = [
                        utils.montage(r.reshape(-1, A, B, C)) for r in recon
                    ]
                    gif.build_gif(recon, saveto='manifold_%08d.gif' % batch_i)
                    plt.close('all')
                batch_i += 1
        except tf.errors.OutOfRangeError:
            print('Done.')
        finally:
            # One of the threads has issued an exception.  So let's tell all the
            # threads to shutdown.
            coord.request_stop()

        # Wait until all threads have finished.
        coord.join(threads)

        # Clean up the session.
        sess.close()
