"""Convolutional/Variational autoencoder, including demonstration of
training such a network on MNIST, CelebNet and the film, "Sita Sings The Blues"
using an image pipeline.

Parag K. Mital, Jan 2016
"""
import tensorflow as tf
import numpy as np
import os
from libs.dataset_utils import create_input_pipeline
from libs.datasets import CELEB
from libs.utils import *


def encoder(x, n_hidden=None, dimensions=[], filter_sizes=[],
            convolutional=False, activation=tf.nn.relu,
            output_activation=tf.nn.sigmoid):
    """Summary

    Parameters
    ----------
    x : TYPE
        Description
    n_hidden : None, optional
        Description
    dimensions : list, optional
        Description
    filter_sizes : list, optional
        Description
    convolutional : bool, optional
        Description
    activation : TYPE, optional
        Description
    output_activation : TYPE, optional
        Description

    Returns
    -------
    name : TYPE
        Description
    """
    if convolutional:
        x_tensor = to_tensor(x)
    else:
        x_tensor = tf.reshape(
            tensor=x,
            shape=[-1, dimensions[0]])
        dimensions = dimensions[1:]
    current_input = x_tensor

    Ws = []
    hs = []
    shapes = []
    for layer_i, n_output in enumerate(dimensions):
        with tf.variable_scope(str(layer_i)):
            shapes.append(current_input.get_shape().as_list())
            if convolutional:
                h, W = conv2d(
                    x=current_input,
                    n_output=n_output,
                    k_h=filter_sizes[layer_i],
                    k_w=filter_sizes[layer_i],
                    padding='SAME')
            else:
                h, W = linear(
                    x=current_input,
                    n_output=n_output)
            h = activation(h)
            Ws.append(W)
            hs.append(h)

        current_input = h

    shapes.append(h.get_shape().as_list())

    with tf.variable_scope('flatten'):
        flattened = flatten(current_input)

    with tf.variable_scope('hidden'):
        if n_hidden:
            h, W = linear(flattened, n_hidden, name='linear')
            h = activation(h)
        else:
            h = flattened

    return {'z': h, 'Ws': Ws, 'hs': hs, 'shapes': shapes}


def decoder(z, shapes, n_hidden=None,
            dimensions=[], filter_sizes=[],
            convolutional=False, activation=tf.nn.relu,
            output_activation=tf.nn.relu):
    """Summary

    Parameters
    ----------
    z : TYPE
        Description
    shapes : TYPE
        Description
    n_hidden : None, optional
        Description
    dimensions : list, optional
        Description
    filter_sizes : list, optional
        Description
    convolutional : bool, optional
        Description
    activation : TYPE, optional
        Description
    output_activation : TYPE, optional
        Description

    Returns
    -------
    name : TYPE
        Description
    """
    with tf.variable_scope('hidden/1'):
        if n_hidden:
            h = linear(z, n_hidden, name='linear')[0]
            h = activation(h)
        else:
            h = z

    with tf.variable_scope('hidden/2'):
        dims = shapes[0]
        size = dims[1] * dims[2] * dims[3] if convolutional else dims[1]
        h = linear(h, size, name='linear')[0]
        current_input = activation(h)
        if convolutional:
            current_input = tf.reshape(
                current_input,
                tf.pack([tf.shape(current_input)[0], dims[1], dims[2], dims[3]]))

    Ws = []
    hs = []
    for layer_i, n_output in enumerate(dimensions[1:]):
        with tf.variable_scope('decoder/{}'.format(layer_i)):
            if convolutional:
                shape = shapes[layer_i + 1]
                h, W = deconv2d(x=current_input,
                                n_output_h=shape[1],
                                n_output_w=shape[2],
                                n_output_ch=shape[3],
                                n_input_ch=shapes[layer_i][3],
                                k_h=filter_sizes[layer_i],
                                k_w=filter_sizes[layer_i])
            else:
                h, W = linear(x=current_input,
                              n_output=n_output)
            if (layer_i + 1) < len(dimensions):
                h = activation(h)
            else:
                h = output_activation(h)
            Ws.append(W)
            hs.append(h)
            current_input = h

    z = tf.identity(current_input, name="x_tilde")
    return {'x_tilde': current_input, 'Ws': Ws, 'hs': hs}


def variational_bayes(h, n_code):
    """Summary

    Parameters
    ----------
    h : TYPE
        Description
    n_code : TYPE
        Description

    Returns
    -------
    name : TYPE
        Description
    """
    z_mu = tf.nn.tanh(linear(h, n_code, name='mu')[0])
    z_log_sigma = 0.5 * tf.nn.tanh(linear(h, n_code, name='log_sigma')[0])

    # Sample from noise distribution p(eps) ~ N(0, 1)
    epsilon = tf.random_normal(tf.pack([tf.shape(h)[0], n_code]))

    # Sample from posterior
    z = tf.add(z_mu, tf.mul(epsilon, tf.exp(z_log_sigma)), name='z')
    # -log(p(z)/q(z|x)), bits by coding.
    # variational bound coding costs kl(p(z|x)||q(z|x))
    # d_kl(q(z|x)||p(z))
    loss_z = -0.5 * tf.reduce_sum(
        1.0 + 2.0 * z_log_sigma - tf.square(z_mu) - tf.exp(2.0 * z_log_sigma),
        1)
    return z, z_mu, z_log_sigma, loss_z


def discriminator(x, convolutional=True,
                  filter_sizes=[5, 5, 5, 5],
                  activation=tf.nn.relu,
                  n_filters=[100, 100, 100, 100]):
    """Summary

    Parameters
    ----------
    x : TYPE
        Description
    convolutional : bool, optional
        Description
    filter_sizes : list, optional
        Description
    n_filters : list, optional
        Description

    Returns
    -------
    name : TYPE
        Description
    """
    encoding = encoder(x=x,
                       convolutional=convolutional,
                       dimensions=n_filters,
                       filter_sizes=filter_sizes,
                       activation=activation)

    # flatten, then linear to 1 value
    res = flatten(encoding['z'], name='flatten')
    if res.get_shape().as_list()[-1] > 1:
        res = linear(res, 1)[0]

    return {'logits': res, 'probs': tf.nn.sigmoid(res),
            'Ws': encoding['Ws'], 'hs': encoding['hs']}


def VAE(input_shape=[None, 784],
        n_filters=[64, 64, 64],
        filter_sizes=[4, 4, 4],
        n_hidden=32,
        n_code=2,
        activation=tf.nn.tanh,
        convolutional=False,
        variational=False):
    """Summary

    Parameters
    ----------
    input_shape : list, optional
        Description
    n_filters : list, optional
        Description
    filter_sizes : list, optional
        Description
    n_hidden : int, optional
        Description
    n_code : int, optional
        Description
    activation : TYPE, optional
        Description
    convolutional : bool, optional
        Description
    variational : bool, optional
        Description

    Returns
    -------
    name : TYPE
        Description
    """
    # network input / placeholders for train (bn)
    x = tf.placeholder(tf.float32, input_shape, 'x')

    with tf.variable_scope('encoder'):
        encoding = encoder(x=x,
                           n_hidden=n_hidden,
                           convolutional=convolutional,
                           dimensions=n_filters,
                           filter_sizes=filter_sizes,
                           activation=activation)

    if variational:
        with tf.variable_scope('variational'):
            z, z_mu, z_log_sigma, loss_z = variational_bayes(
                h=encoding['z'], n_code=n_code)
    else:
        z = encoding['z']
        loss_z = None

    shapes = encoding['shapes'].copy()
    shapes.reverse()
    n_filters = n_filters.copy()
    n_filters.reverse()
    n_filters += [input_shape[-1]]

    with tf.variable_scope('generator'):
        decoding = decoder(z=z,
                           shapes=shapes,
                           n_hidden=n_hidden,
                           dimensions=n_filters,
                           filter_sizes=filter_sizes,
                           convolutional=convolutional,
                           activation=activation)

    x_tilde = decoding['x_tilde']
    x_flat = flatten(x)
    x_tilde_flat = flatten(x_tilde)

    # -log(p(x|z))
    loss_x = tf.reduce_sum(tf.squared_difference(x_flat, x_tilde_flat), 1)
    return {'loss_x': loss_x, 'loss_z': loss_z, 'x': x, 'z': z,
            'Ws': encoding['Ws'], 'hs': decoding['hs'],
            'x_tilde': x_tilde}


def VAEGAN(input_shape=[None, 784],
           n_filters=[64, 64, 64],
           filter_sizes=[4, 4, 4],
           n_hidden=32,
           n_code=2,
           activation=tf.nn.tanh,
           convolutional=False,
           variational=False):
    """Summary

    Parameters
    ----------
    input_shape : list, optional
        Description
    n_filters : list, optional
        Description
    filter_sizes : list, optional
        Description
    n_hidden : int, optional
        Description
    n_code : int, optional
        Description
    activation : TYPE, optional
        Description
    convolutional : bool, optional
        Description
    variational : bool, optional
        Description

    Returns
    -------
    name : TYPE
        Description
    """
    # network input / placeholders for train (bn)
    x = tf.placeholder(tf.float32, input_shape, 'x')
    z_samp = tf.placeholder(tf.float32, [None, n_code], 'z_samp')

    with tf.variable_scope('encoder'):
        encoding = encoder(x=x,
                           n_hidden=n_hidden,
                           convolutional=convolutional,
                           dimensions=n_filters,
                           filter_sizes=filter_sizes,
                           activation=activation)

        with tf.variable_scope('variational'):
            z, z_mu, z_log_sigma, loss_z = variational_bayes(
                h=encoding['z'], n_code=n_code)

    shapes = encoding['shapes'].copy()
    shapes.reverse()
    n_filters_decoder = n_filters.copy()
    n_filters_decoder.reverse()
    n_filters_decoder += [input_shape[-1]]

    with tf.variable_scope('generator'):
        decoding_actual = decoder(z=z,
                                  shapes=shapes,
                                  n_hidden=n_hidden,
                                  convolutional=convolutional,
                                  dimensions=n_filters_decoder,
                                  filter_sizes=filter_sizes,
                                  activation=activation)

    with tf.variable_scope('generator', reuse=True):
        decoding_sampled = decoder(z=z_samp,
                                   shapes=shapes,
                                   n_hidden=n_hidden,
                                   convolutional=convolutional,
                                   dimensions=n_filters_decoder,
                                   filter_sizes=filter_sizes,
                                   activation=activation)

    with tf.variable_scope('discriminator'):
        D_real = discriminator(x,
                               filter_sizes=filter_sizes,
                               n_filters=n_filters,
                               activation=activation)

    with tf.variable_scope('discriminator', reuse=True):
        D_fake = discriminator(decoding_actual['x_tilde'],
                               filter_sizes=filter_sizes,
                               n_filters=n_filters,
                               activation=activation)

    with tf.variable_scope('discriminator', reuse=True):
        D_samp = discriminator(decoding_sampled['x_tilde'],
                               filter_sizes=filter_sizes,
                               n_filters=n_filters,
                               activation=activation)

    with tf.variable_scope('loss'):
        # Weights influence of content/style of decoder
        gamma = tf.placeholder(tf.float32, name='gamma')

        # Discriminator_l Log Likelihood Loss
        loss_D_llike = 0
        for h_fake, h_real in zip(D_fake['hs'][3:], D_real['hs'][3:]):
            loss_D_llike += tf.reduce_sum(
                0.5 * tf.squared_difference(
                    flatten(h_fake), flatten(h_real)), 1)

        # GAN Loss
        eps = 1e-12
        loss_real = tf.reduce_sum(tf.log(D_real['probs'] + eps), 1)
        loss_fake = tf.reduce_sum(tf.log(1 - D_fake['probs'] + eps), 1)
        loss_samp = tf.reduce_sum(tf.log(1 - D_samp['probs'] + eps), 1)

        loss_GAN = (loss_real + loss_fake + loss_samp) / 3.0

        loss_enc = tf.reduce_mean(loss_z + loss_D_llike)
        loss_gen = tf.reduce_mean(gamma * loss_D_llike - loss_GAN)
        loss_dis = -tf.reduce_mean(loss_GAN)

    return {'x': x, 'z': z, 'x_tilde': decoding_actual['x_tilde'],
            'z_samp': z_samp, 'x_tilde_samp': decoding_sampled['x_tilde'],
            'loss_real': loss_real, 'loss_fake': loss_fake, 'loss_samp': loss_samp,
            'loss_GAN': loss_GAN, 'loss_D_llike': loss_D_llike,
            'loss_enc': loss_enc, 'loss_gen': loss_gen, 'loss_dis': loss_dis,
            'gamma': gamma}


def train_vaegan(files,
                 learning_rate=0.00001,
                 batch_size=64,
                 n_epochs=250,
                 n_examples=10,
                 input_shape=[218, 178, 3],
                 crop_shape=[64, 64, 3],
                 crop_factor=0.8,
                 n_filters=[100, 100, 100, 100],
                 n_hidden=None,
                 n_code=128,
                 convolutional=True,
                 variational=True,
                 filter_sizes=[3, 3, 3, 3],
                 activation=tf.nn.elu,
                 ckpt_name="vaegan.ckpt"):
    """Summary

    Parameters
    ----------
    files : TYPE
        Description
    learning_rate : float, optional
        Description
    batch_size : int, optional
        Description
    n_epochs : int, optional
        Description
    n_examples : int, optional
        Description
    input_shape : list, optional
        Description
    crop_shape : list, optional
        Description
    crop_factor : float, optional
        Description
    n_filters : list, optional
        Description
    n_hidden : int, optional
        Description
    n_code : int, optional
        Description
    convolutional : bool, optional
        Description
    variational : bool, optional
        Description
    filter_sizes : list, optional
        Description
    activation : TYPE, optional
        Description
    ckpt_name : str, optional
        Description

    Returns
    -------
    name : TYPE
        Description
    """

    ae = VAEGAN(input_shape=[None] + crop_shape,
                convolutional=convolutional,
                variational=variational,
                n_filters=n_filters,
                n_hidden=n_hidden,
                n_code=n_code,
                filter_sizes=filter_sizes,
                activation=activation)

    batch = create_input_pipeline(
        files=files,
        batch_size=batch_size,
        n_epochs=n_epochs,
        crop_shape=crop_shape,
        crop_factor=crop_factor,
        shape=input_shape)

    zs = np.random.randn(4, n_code).astype(np.float32)
    zs = make_latent_manifold(zs, n_examples)

    opt_enc = tf.train.AdamOptimizer(
        learning_rate=learning_rate).minimize(
        ae['loss_enc'],
        var_list=[var_i for var_i in tf.trainable_variables()
                  if var_i.name.startswith('encoder')])

    opt_gen = tf.train.AdamOptimizer(
        learning_rate=learning_rate).minimize(
        ae['loss_gen'],
        var_list=[var_i for var_i in tf.trainable_variables()
                  if var_i.name.startswith('generator')])

    opt_dis = tf.train.AdamOptimizer(
        learning_rate=learning_rate).minimize(
        ae['loss_dis'],
        var_list=[var_i for var_i in tf.trainable_variables()
                  if var_i.name.startswith('discriminator')])

    sess = tf.Session()
    saver = tf.train.Saver()
    sess.run(tf.initialize_all_variables())
    coord = tf.train.Coordinator()
    tf.get_default_graph().finalize()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    if os.path.exists(ckpt_name):
        saver.restore(sess, ckpt_name)
        print("VAE model restored.")

    t_i = 0
    batch_i = 0
    epoch_i = 0

    equilibrium = 0.693
    margin = 0.4

    n_files = len(files)
    test_xs = sess.run(batch) / 255.0
    montage(test_xs, 'test_xs.png')
    try:
        while not coord.should_stop() or epoch_i < n_epochs:
            if batch_i % (n_files // batch_size) == 0:
                batch_i = 0
                epoch_i += 1
                print('---------- EPOCH:', epoch_i)

            batch_i += 1
            batch_xs = sess.run(batch) / 255.0
            batch_zs = np.random.randn(batch_size, n_code).astype(np.float32)
            real_cost, fake_cost, _ = sess.run([
                ae['loss_real'], ae['loss_fake'], opt_enc],
                feed_dict={
                    ae['x']: batch_xs,
                    ae['gamma']: 0.5})
            real_cost = -np.mean(real_cost)
            fake_cost = -np.mean(fake_cost)
            print('real:', real_cost, '/ fake:', fake_cost)

            gen_update = True
            dis_update = True

            if real_cost > (equilibrium + margin) or \
               fake_cost > (equilibrium + margin):
                gen_update = False

            if real_cost < (equilibrium - margin) or \
               fake_cost < (equilibrium - margin):
                dis_update = False

            if not (gen_update or dis_update):
                gen_update = True
                dis_update = True

            if gen_update:
                sess.run(opt_gen, feed_dict={
                    ae['x']: batch_xs,
                    ae['z_samp']: batch_zs,
                    ae['gamma']: 0.5})
            if dis_update:
                sess.run(opt_dis, feed_dict={
                    ae['x']: batch_xs,
                    ae['z_samp']: batch_zs,
                    ae['gamma']: 0.5})

            if batch_i % 50 == 0:

                # Plot example reconstructions from latent layer
                recon = sess.run(
                    ae['x_tilde'], feed_dict={
                        ae['z']: zs})
                print('recon:', recon.min(), recon.max())
                recon = np.clip(recon / recon.max(), 0, 1)
                montage(recon.reshape([-1] + crop_shape),
                        'imgs/manifold_%08d.png' % t_i)

                # Plot example reconstructions
                recon = sess.run(
                    ae['x_tilde'], feed_dict={
                        ae['x']: test_xs})
                print('recon:', recon.min(), recon.max())
                recon = np.clip(recon / recon.max(), 0, 1)
                montage(recon.reshape([-1] + crop_shape),
                        'imgs/reconstruction_%08d.png' % t_i)
                t_i += 1

            if batch_i % 100 == 0:
                # Save the variables to disk.
                save_path = saver.save(sess, "./" + ckpt_name,
                                       global_step=batch_i,
                                       write_meta_graph=False)
                print("Model saved in file: %s" % save_path)
    except tf.errors.OutOfRangeError:
        print('Done training -- epoch limit reached')
    finally:
        # One of the threads has issued an exception.  So let's tell all the
        # threads to shutdown.
        coord.request_stop()

    # Wait until all threads have finished.
    coord.join(threads)

    # Clean up the session.
    sess.close()


def test_celeb():
    """Summary

    Returns
    -------
    name : TYPE
        Description
    """
    files = CELEB()
    train_vaegan(
        files=files,
        batch_size=64,
        n_epochs=100,
        crop_shape=[100, 100, 3],
        crop_factor=0.8,
        input_shape=[218, 178, 3],
        convolutional=True,
        variational=True,
        n_filters=[256, 384, 512, 1024, 2048],
        n_hidden=None,
        n_code=512,
        filter_sizes=[3, 3, 3, 3, 3],
        activation=tf.nn.elu,
        ckpt_name='celeb.ckpt')


def test_sita():
    """Summary

    Returns
    -------
    name : TYPE
        Description
    """
    if not os.path.exists('sita'):
        os.system('wget http://ossguy.com/sita/Sita_Sings_the_Blues_640x360_XviD.avi')
        os.mkdir('sita')
        os.system('ffmpeg -i Sita_Sings_the_Blues_640x360_XviD.avi -r 60 -f' +
                  ' image2 -s 160x90 sita/sita-%08d.jpg')
    files = [os.path.join('sita', f) for f in os.listdir('sita')]

    train_vaegan(
        files=files,
        batch_size=64,
        n_epochs=50,
        crop_shape=[90, 160, 3],
        crop_factor=1.0,
        input_shape=[218, 178, 3],
        convolutional=True,
        variational=True,
        n_filters=[100, 100, 100, 100, 100],
        n_hidden=250,
        n_code=100,
        filter_sizes=[3, 3, 3, 3, 2],
        activation=tf.nn.elu,
        ckpt_name='sita.ckpt')


if __name__ == '__main__':
    test_celeb()
