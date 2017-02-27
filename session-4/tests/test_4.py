import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from libs import utils
from libs import dataset_utils
from libs import vgg16, inception, i2v
from libs import stylenet


def test_libraries():
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    from skimage.transform import resize
    from skimage import data
    from scipy.misc import imresize
    from scipy.ndimage.filters import gaussian_filter
    import IPython.display as ipyd
    import tensorflow as tf
    from libs import utils, gif, datasets, dataset_utils, vae, dft, vgg16, nb_utils


def test_vgg():
    net = vgg16.get_vgg_model()
    guide_og = plt.imread('clinton.png')[..., :3]
    dream_og = plt.imread('arles.png')[..., :3]
    guide_img = net['preprocess'](guide_og)[np.newaxis]
    dream_img = net['preprocess'](dream_og)[np.newaxis]
    assert(guide_img.shape == (1, 224, 224, 3))
    assert(dream_img.shape == (1, 224, 224, 3))
    assert(guide_img.dtype == np.dtype('float32'))
    assert(guide_img.dtype == np.dtype('float32'))
    g = tf.Graph()
    with tf.Session(graph=g) as sess:
        tf.import_graph_def(net['graph_def'], name='net')
        names = [op.name for op in g.get_operations()]
        x = g.get_tensor_by_name(names[0] + ':0')
        softmax = g.get_tensor_by_name(names[-2] + ':0')
        res = softmax.eval(feed_dict={x: guide_img,
                                      'net/dropout_1/random_uniform:0': [[1.0] * 4096],
                                      'net/dropout/random_uniform:0': [[1.0] * 4096]})[0]
        assert(np.argmax(res) == 681)
        res = softmax.eval(feed_dict={x: dream_img,
                                      'net/dropout_1/random_uniform:0': [[1.0] * 4096],
                                      'net/dropout/random_uniform:0': [[1.0] * 4096]})[0]
        assert(np.argmax(res) == 540)
    return


def test_inception():
    net = inception.get_inception_model(version='v5')
    guide_og = plt.imread('clinton.png')[..., :3]
    dream_og = plt.imread('arles.png')[..., :3]
    guide_img = net['preprocess'](guide_og)[np.newaxis]
    dream_img = net['preprocess'](dream_og)[np.newaxis]
    assert(guide_img.shape == (1, 299, 299, 3))
    assert(dream_img.shape == (1, 299, 299, 3))
    assert(guide_img.dtype == np.dtype('float32'))
    assert(guide_img.dtype == np.dtype('float32'))
    g = tf.Graph()
    with tf.Session(graph=g) as sess:
        tf.import_graph_def(net['graph_def'], name='net')
        names = [op.name for op in g.get_operations()]
        x = g.get_tensor_by_name(names[0] + ':0')
        softmax = g.get_tensor_by_name(names[-1] + ':0')
        res = softmax.eval(feed_dict={x: guide_img})
        assert(np.argmax(res[0][:1000]) == 899)
        res = softmax.eval(feed_dict={x: dream_img})
        assert(np.argmax(res[0][:1000]) == 834)
    return


def test_stylenet():
    stylenet.test()


#def test_stylenet_video():
#    stylenet.test_video()
