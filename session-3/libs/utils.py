"""Utilities used in the Kadenze Academy Course on Deep Learning w/ Tensorflow.

Creative Applications of Deep Learning w/ Tensorflow.
Kadenze, Inc.
Parag K. Mital

Copyright Parag K. Mital, June 2016.
"""
import matplotlib.pyplot as plt
import tensorflow as tf
import urllib
import numpy as np
import zipfile
import os
from scipy.io import wavfile


def load_audio(filename, b_normalize=True):
    """Load the audiofile at the provided filename using scipy.io.wavfile.

    Optionally normalizes the audio to the maximum value.

    Parameters
    ----------
    filename : str
        File to load.
    b_normalize : bool, optional
        Normalize to the maximum value.
    """
    sr, s = wavfile.read(filename)
    if b_normalize:
        s = s.astype(np.float32)
        s = (s / np.max(np.abs(s)))
        s -= np.mean(s)
    return s


def corrupt(x):
    """Take an input tensor and add uniform masking.

    Parameters
    ----------
    x : Tensor/Placeholder
        Input to corrupt.
    Returns
    -------
    x_corrupted : Tensor
        50 pct of values corrupted.
    """
    return tf.mul(x, tf.cast(tf.random_uniform(shape=tf.shape(x),
                                               minval=0,
                                               maxval=2,
                                               dtype=tf.int32), tf.float32))


def interp(l, r, n_samples):
    """Intepolate between the arrays l and r, n_samples times.

    Parameters
    ----------
    l : np.ndarray
        Left edge
    r : np.ndarray
        Right edge
    n_samples : int
        Number of samples

    Returns
    -------
    arr : np.ndarray
        Inteporalted array
    """
    return np.array([
        l + step_i / (n_samples - 1) * (r - l)
        for step_i in range(n_samples)])


def make_latent_manifold(corners, n_samples):
    """Create a 2d manifold out of the provided corners: n_samples * n_samples.

    Parameters
    ----------
    corners : list of np.ndarray
        The four corners to intepolate.
    n_samples : int
        Number of samples to use in interpolation.

    Returns
    -------
    arr : np.ndarray
        Stacked array of all 2D interpolated samples
    """
    left = interp(corners[0], corners[1], n_samples)
    right = interp(corners[2], corners[3], n_samples)

    embedding = []
    for row_i in range(n_samples):
        embedding.append(interp(left[row_i], right[row_i], n_samples))
    return np.vstack(embedding)


def imcrop_tosquare(img):
    """Make any image a square image.

    Parameters
    ----------
    img : np.ndarray
        Input image to crop, assumed at least 2d.

    Returns
    -------
    crop : np.ndarray
        Cropped image.
    """
    size = np.min(img.shape[:2])
    extra = img.shape[:2] - size
    crop = img
    for i in np.flatnonzero(extra):
        crop = np.take(crop, extra[i] // 2 + np.r_[:size], axis=i)
    return crop


def slice_montage(montage, img_h, img_w, n_imgs):
    """Slice a montage image into n_img h x w images.

    Performs the opposite of the montage function.  Takes a montage image and
    slices it back into a N x H x W x C image.

    Parameters
    ----------
    montage : np.ndarray
        Montage image to slice.
    img_h : int
        Height of sliced image
    img_w : int
        Width of sliced image
    n_imgs : int
        Number of images to slice

    Returns
    -------
    sliced : np.ndarray
        Sliced images as 4d array.
    """
    sliced_ds = []
    for i in range(int(np.sqrt(n_imgs))):
        for j in range(int(np.sqrt(n_imgs))):
            sliced_ds.append(montage[
                1 + i + i * img_h:1 + i + (i + 1) * img_h,
                1 + j + j * img_w:1 + j + (j + 1) * img_w])
    return np.array(sliced_ds)


def montage(images, saveto='montage.png'):
    """Draw all images as a montage separated by 1 pixel borders.

    Also saves the file to the destination specified by `saveto`.

    Parameters
    ----------
    images : numpy.ndarray
        Input array to create montage of.  Array should be:
        batch x height x width x channels.
    saveto : str
        Location to save the resulting montage image.

    Returns
    -------
    m : numpy.ndarray
        Montage image.
    """
    if isinstance(images, list):
        images = np.array(images)
    img_h = images.shape[1]
    img_w = images.shape[2]
    n_plots = int(np.ceil(np.sqrt(images.shape[0])))
    if len(images.shape) == 4 and images.shape[3] == 3:
        m = np.ones(
            (images.shape[1] * n_plots + n_plots + 1,
             images.shape[2] * n_plots + n_plots + 1, 3)) * 0.5
    else:
        m = np.ones(
            (images.shape[1] * n_plots + n_plots + 1,
             images.shape[2] * n_plots + n_plots + 1)) * 0.5
    for i in range(n_plots):
        for j in range(n_plots):
            this_filter = i * n_plots + j
            if this_filter < images.shape[0]:
                this_img = images[this_filter]
                m[1 + i + i * img_h:1 + i + (i + 1) * img_h,
                  1 + j + j * img_w:1 + j + (j + 1) * img_w] = this_img
    plt.imsave(arr=m, fname=saveto)
    return m


def montage_filters(W):
    """Draws all filters (n_input * n_output filters) as a
    montage image separated by 1 pixel borders.

    Parameters
    ----------
    W : Tensor
        Input tensor to create montage of.

    Returns
    -------
    m : numpy.ndarray
        Montage image.
    """
    W = np.reshape(W, [W.shape[0], W.shape[1], 1, W.shape[2] * W.shape[3]])
    n_plots = int(np.ceil(np.sqrt(W.shape[-1])))
    m = np.ones(
        (W.shape[0] * n_plots + n_plots + 1,
         W.shape[1] * n_plots + n_plots + 1)) * 0.5
    for i in range(n_plots):
        for j in range(n_plots):
            this_filter = i * n_plots + j
            if this_filter < W.shape[-1]:
                m[1 + i + i * W.shape[0]:1 + i + (i + 1) * W.shape[0],
                  1 + j + j * W.shape[1]:1 + j + (j + 1) * W.shape[1]] = (
                    np.squeeze(W[:, :, :, this_filter]))
    return m


def get_celeb_files(dst='img_align_celeba', max_images=100):
    """Download the first 100 images of the celeb dataset.

    Files will be placed in a directory 'img_align_celeba' if one
    doesn't exist.

    Returns
    -------
    files : list of strings
        Locations to the first 100 images of the celeb net dataset.
    """
    # Create a directory
    if not os.path.exists(dst):
        os.mkdir(dst)

    # Now perform the following 100 times:
    for img_i in range(1, max_images + 1):

        # create a string using the current loop counter
        f = '000%03d.jpg' % img_i

        if not os.path.exists(os.path.join(dst, f)):

            # and get the url with that string appended the end
            url = 'https://s3.amazonaws.com/cadl/celeb-align/' + f

            # We'll print this out to the console so we can see how far we've gone
            print(url, end='\r')

            # And now download the url to a location inside our new directory
            urllib.request.urlretrieve(url, os.path.join(dst, f))

    files = [os.path.join(dst, file_i)
             for file_i in os.listdir(dst)
             if '.jpg' in file_i][:max_images]
    return files


def get_celeb_imgs(max_images=100):
    """Load the first `max_images` images of the celeb dataset.

    Returns
    -------
    imgs : list of np.ndarray
        List of the first 100 images from the celeb dataset
    """
    return [plt.imread(f_i) for f_i in get_celeb_files(max_images=max_images)]


def gauss(mean, stddev, ksize):
    """Use Tensorflow to compute a Gaussian Kernel.

    Parameters
    ----------
    mean : float
        Mean of the Gaussian (e.g. 0.0).
    stddev : float
        Standard Deviation of the Gaussian (e.g. 1.0).
    ksize : int
        Size of kernel (e.g. 16).

    Returns
    -------
    kernel : np.ndarray
        Computed Gaussian Kernel using Tensorflow.
    """
    g = tf.Graph()
    with tf.Session(graph=g):
        x = tf.linspace(-3.0, 3.0, ksize)
        z = (tf.exp(tf.neg(tf.pow(x - mean, 2.0) /
                           (2.0 * tf.pow(stddev, 2.0)))) *
             (1.0 / (stddev * tf.sqrt(2.0 * 3.1415))))
        return z.eval()


def gauss2d(mean, stddev, ksize):
    """Use Tensorflow to compute a 2D Gaussian Kernel.

    Parameters
    ----------
    mean : float
        Mean of the Gaussian (e.g. 0.0).
    stddev : float
        Standard Deviation of the Gaussian (e.g. 1.0).
    ksize : int
        Size of kernel (e.g. 16).

    Returns
    -------
    kernel : np.ndarray
        Computed 2D Gaussian Kernel using Tensorflow.
    """
    z = gauss(mean, stddev, ksize)
    g = tf.Graph()
    with tf.Session(graph=g):
        z_2d = tf.matmul(tf.reshape(z, [ksize, 1]), tf.reshape(z, [1, ksize]))
        return z_2d.eval()


def convolve(img, kernel):
    """Use Tensorflow to convolve a 4D image with a 4D kernel.

    Parameters
    ----------
    img : np.ndarray
        4-dimensional image shaped N x H x W x C
    kernel : np.ndarray
        4-dimensional image shape K_H, K_W, C_I, C_O corresponding to the
        kernel's height and width, the number of input channels, and the
        number of output channels.  Note that C_I should = C.

    Returns
    -------
    result : np.ndarray
        Convolved result.
    """
    g = tf.Graph()
    with tf.Session(graph=g):
        convolved = tf.nn.conv2d(img, kernel, strides=[1, 1, 1, 1], padding='SAME')
        res = convolved.eval()
    return res


def gabor(ksize=32):
    """Use Tensorflow to compute a 2D Gabor Kernel.

    Parameters
    ----------
    ksize : int, optional
        Size of kernel.

    Returns
    -------
    gabor : np.ndarray
        Gabor kernel with ksize x ksize dimensions.
    """
    g = tf.Graph()
    with tf.Session(graph=g):
        z_2d = gauss2d(0.0, 1.0, ksize)
        ones = tf.ones((1, ksize))
        ys = tf.sin(tf.linspace(-3.0, 3.0, ksize))
        ys = tf.reshape(ys, [ksize, 1])
        wave = tf.matmul(ys, ones)
        gabor = tf.mul(wave, z_2d)
        return gabor.eval()


def build_submission(filename, file_list, optional_file_list=()):
    """Helper utility to check homework assignment submissions and package them.

    Parameters
    ----------
    filename : str
        Output zip file name
    file_list : tuple
        Tuple of files to include
    """
    # check each file exists
    for part_i, file_i in enumerate(file_list):
        if not os.path.exists(file_i):
            print('\nYou are missing the file {}.  '.format(file_i) +
                  'It does not look like you have completed Part {}.'.format(
                part_i + 1))

    def zipdir(path, zf):
        for root, dirs, files in os.walk(path):
            for file in files:
                # make sure the files are part of the necessary file list
                if file.endswith(file_list) or file.endswith(optional_file_list):
                    zf.write(os.path.join(root, file))

    # create a zip file with the necessary files
    zipf = zipfile.ZipFile(filename, 'w', zipfile.ZIP_DEFLATED)
    zipdir('.', zipf)
    zipf.close()
    print('Your assignment zip file has been created!')
    print('Now submit the file:\n{}\nto Kadenze for grading!'.format(
        os.path.abspath(filename)))


def normalize(a, s=0.1):
    '''Normalize the image range for visualization'''
    return np.uint8(np.clip(
        (a - a.mean()) / max(a.std(), 1e-4) * s + 0.5,
        0, 1) * 255)


# %%
def weight_variable(shape, **kwargs):
    '''Helper function to create a weight variable initialized with
    a normal distribution
    Parameters
    ----------
    shape : list
        Size of weight variable
    '''
    if isinstance(shape, list):
        initial = tf.random_normal(tf.pack(shape), mean=0.0, stddev=0.01)
        initial.set_shape(shape)
    else:
        initial = tf.random_normal(shape, mean=0.0, stddev=0.01)
    return tf.Variable(initial, **kwargs)


# %%
def bias_variable(shape, **kwargs):
    '''Helper function to create a bias variable initialized with
    a constant value.
    Parameters
    ----------
    shape : list
        Size of weight variable
    '''
    if isinstance(shape, list):
        initial = tf.random_normal(tf.pack(shape), mean=0.0, stddev=0.01)
        initial.set_shape(shape)
    else:
        initial = tf.random_normal(shape, mean=0.0, stddev=0.01)
    return tf.Variable(initial, **kwargs)


def binary_cross_entropy(z, x):
    """Binary Cross Entropy measures cross entropy of a binary variable.

    loss(x, z) = - sum_i (x[i] * log(z[i]) + (1 - x[i]) * log(1 - z[i]))

    Parameters
    ----------
    z : tf.Tensor
        A `Tensor` of the same type and shape as `x`.
    x : tf.Tensor
        A `Tensor` of type `float32` or `float64`.
    """
    eps = 1e-12
    return (-(x * tf.log(z + eps) +
              (1. - x) * tf.log(1. - z + eps)))


def conv2d(x, n_output,
           k_h=5, k_w=5, d_h=2, d_w=2,
           padding='SAME', name='conv2d', reuse=None):
    """Helper for creating a 2d convolution operation.

    Parameters
    ----------
    x : tf.Tensor
        Input tensor to convolve.
    n_output : int
        Number of filters.
    k_h : int, optional
        Kernel height
    k_w : int, optional
        Kernel width
    d_h : int, optional
        Height stride
    d_w : int, optional
        Width stride
    padding : str, optional
        Padding type: "SAME" or "VALID"
    name : str, optional
        Variable scope

    Returns
    -------
    op : tf.Tensor
        Output of convolution
    """
    with tf.variable_scope(name or 'conv2d', reuse=reuse):
        W = tf.get_variable(
            name='W',
            shape=[k_h, k_w, x.get_shape()[-1], n_output],
            initializer=tf.contrib.layers.xavier_initializer_conv2d())

        conv = tf.nn.conv2d(
            name='conv',
            input=x,
            filter=W,
            strides=[1, d_h, d_w, 1],
            padding=padding)

        b = tf.get_variable(
            name='b',
            shape=[n_output],
            initializer=tf.constant_initializer(0.0))

        h = tf.nn.bias_add(
            name='h',
            value=conv,
            bias=b)

    return h, W


def deconv2d(x, n_output_h, n_output_w, n_output_ch, n_input_ch=None,
             k_h=5, k_w=5, d_h=2, d_w=2,
             padding='SAME', name='deconv2d', reuse=None):
    """Deconvolution helper.

    Parameters
    ----------
    x : tf.Tensor
        Input tensor to convolve.
    n_output_h : int
        Height of output
    n_output_w : int
        Width of output
    n_output_ch : int
        Number of filters.
    k_h : int, optional
        Kernel height
    k_w : int, optional
        Kernel width
    d_h : int, optional
        Height stride
    d_w : int, optional
        Width stride
    padding : str, optional
        Padding type: "SAME" or "VALID"
    name : str, optional
        Variable scope

    Returns
    -------
    op : tf.Tensor
        Output of deconvolution
    """
    with tf.variable_scope(name or 'deconv2d', reuse=reuse):
        W = tf.get_variable(
            name='W',
            shape=[k_h, k_h, n_output_ch, n_input_ch or x.get_shape()[-1]],
            initializer=tf.contrib.layers.xavier_initializer_conv2d())

        conv = tf.nn.conv2d_transpose(
            name='conv_t',
            value=x,
            filter=W,
            output_shape=tf.pack(
                [tf.shape(x)[0], n_output_h, n_output_w, n_output_ch]),
            strides=[1, d_h, d_w, 1],
            padding=padding)

        conv.set_shape([None, n_output_h, n_output_w, n_output_ch])

        b = tf.get_variable(
            name='b',
            shape=[n_output_ch],
            initializer=tf.constant_initializer(0.0))

        h = tf.nn.bias_add(name='h', value=conv, bias=b)

    return h, W


def lrelu(features, leak=0.2):
    """Leaky rectifier.

    Parameters
    ----------
    features : tf.Tensor
        Input to apply leaky rectifier to.
    leak : float, optional
        Percentage of leak.

    Returns
    -------
    op : tf.Tensor
        Resulting output of applying leaky rectifier activation.
    """
    f1 = 0.5 * (1 + leak)
    f2 = 0.5 * (1 - leak)
    return f1 * features + f2 * abs(features)


def linear(x, n_output, name=None, activation=None, reuse=None):
    """Fully connected layer.

    Parameters
    ----------
    x : tf.Tensor
        Input tensor to connect
    n_output : int
        Number of output neurons
    name : None, optional
        Scope to apply

    Returns
    -------
    h, W : tf.Tensor, tf.Tensor
        Output of fully connected layer and the weight matrix
    """
    if len(x.get_shape()) != 2:
        x = flatten(x, reuse=reuse)

    n_input = x.get_shape().as_list()[1]

    with tf.variable_scope(name or "fc", reuse=reuse):
        W = tf.get_variable(
            name='W',
            shape=[n_input, n_output],
            dtype=tf.float32,
            initializer=tf.contrib.layers.xavier_initializer())

        b = tf.get_variable(
            name='b',
            shape=[n_output],
            dtype=tf.float32,
            initializer=tf.constant_initializer(0.0))

        h = tf.nn.bias_add(
            name='h',
            value=tf.matmul(x, W),
            bias=b)

        if activation:
            h = activation(h)

        return h, W


def flatten(x, name=None, reuse=None):
    """Flatten Tensor to 2-dimensions.

    Parameters
    ----------
    x : tf.Tensor
        Input tensor to flatten.
    name : None, optional
        Variable scope for flatten operations

    Returns
    -------
    flattened : tf.Tensor
        Flattened tensor.
    """
    with tf.variable_scope('flatten'):
        dims = x.get_shape().as_list()
        if len(dims) == 4:
            flattened = tf.reshape(
                x,
                shape=[-1, dims[1] * dims[2] * dims[3]])
        elif len(dims) == 2 or len(dims) == 1:
            flattened = x
        else:
            raise ValueError('Expected n dimensions of 1, 2 or 4.  Found:',
                             len(dims))

        return flattened


def to_tensor(x):
    """Convert 2 dim Tensor to a 4 dim Tensor ready for convolution.

    Performs the opposite of flatten(x).  If the tensor is already 4-D, this
    returns the same as the input, leaving it unchanged.

    Parameters
    ----------
    x : tf.Tesnor
        Input 2-D tensor.  If 4-D already, left unchanged.

    Returns
    -------
    x : tf.Tensor
        4-D representation of the input.

    Raises
    ------
    ValueError
        If the tensor is not 2D or already 4D.
    """
    if len(x.get_shape()) == 2:
        n_input = x.get_shape().as_list()[1]
        x_dim = np.sqrt(n_input)
        if x_dim == int(x_dim):
            x_dim = int(x_dim)
            x_tensor = tf.reshape(
                x, [-1, x_dim, x_dim, 1], name='reshape')
        elif np.sqrt(n_input / 3) == int(np.sqrt(n_input / 3)):
            x_dim = int(np.sqrt(n_input / 3))
            x_tensor = tf.reshape(
                x, [-1, x_dim, x_dim, 3], name='reshape')
        else:
            x_tensor = tf.reshape(
                x, [-1, 1, 1, n_input], name='reshape')
    elif len(x.get_shape()) == 4:
        x_tensor = x
    else:
        raise ValueError('Unsupported input dimensions')
    return x_tensor
