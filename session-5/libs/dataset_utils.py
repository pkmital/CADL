"""Utils for dataset creation.

Creative Applications of Deep Learning w/ Tensorflow.
Kadenze, Inc.
Copyright Parag K. Mital, June 2016.
"""

import os
import pickle
import numpy as np
import tensorflow as tf
from . import dft
from .utils import download_and_extract_tar


def create_input_pipeline(files, batch_size, n_epochs, shape, crop_shape=None,
                          crop_factor=1.0, n_threads=2):
    """Creates a pipefile from a list of image files.
    Includes batch generator/central crop/resizing options.
    The resulting generator will dequeue the images batch_size at a time until
    it throws tf.errors.OutOfRangeError when there are no more images left in
    the queue.

    Parameters
    ----------
    files : list
        List of paths to image files.
    batch_size : int
        Number of image files to load at a time.
    n_epochs : int
        Number of epochs to run before raising tf.errors.OutOfRangeError
    shape : list
        [height, width, channels]
    crop_shape : list
        [height, width] to crop image to.
    crop_factor : float
        Percentage of image to take starting from center.
    n_threads : int, optional
        Number of threads to use for batch shuffling
    """

    # We first create a "producer" queue.  It creates a production line which
    # will queue up the file names and allow another queue to deque the file
    # names all using a tf queue runner.
    # Put simply, this is the entry point of the computational graph.
    # It will generate the list of file names.
    # We also specify it's capacity beforehand.
    producer = tf.train.string_input_producer(
        files, capacity=len(files))

    # We need something which can open the files and read its contents.
    reader = tf.WholeFileReader()

    # We pass the filenames to this object which can read the file's contents.
    # This will create another queue running which dequeues the previous queue.
    keys, vals = reader.read(producer)

    # And then have to decode its contents as we know it is a jpeg image
    imgs = tf.image.decode_jpeg(
        vals,
        channels=3 if len(shape) > 2 and shape[2] == 3 else 0)

    # We have to explicitly define the shape of the tensor.
    # This is because the decode_jpeg operation is still a node in the graph
    # and doesn't yet know the shape of the image.  Future operations however
    # need explicit knowledge of the image's shape in order to be created.
    imgs.set_shape(shape)

    # Next we'll centrally crop the image to the size of 100x100.
    # This operation required explicit knowledge of the image's shape.
    if shape[0] > shape[1]:
        rsz_shape = [int(shape[0] / shape[1] * crop_shape[0] / crop_factor),
                     int(crop_shape[1] / crop_factor)]
    else:
        rsz_shape = [int(crop_shape[0] / crop_factor),
                     int(shape[1] / shape[0] * crop_shape[1] / crop_factor)]
    rszs = tf.image.resize_images(imgs, rsz_shape[0], rsz_shape[1])
    crops = (tf.image.resize_image_with_crop_or_pad(
        rszs, crop_shape[0], crop_shape[1])
        if crop_shape is not None
        else imgs)

    # Now we'll create a batch generator that will also shuffle our examples.
    # We tell it how many it should have in its buffer when it randomly
    # permutes the order.
    min_after_dequeue = len(files) // 100

    # The capacity should be larger than min_after_dequeue, and determines how
    # many examples are prefetched.  TF docs recommend setting this value to:
    # min_after_dequeue + (num_threads + a small safety margin) * batch_size
    capacity = min_after_dequeue + (n_threads + 1) * batch_size

    # Randomize the order and output batches of batch_size.
    batch = tf.train.shuffle_batch([crops],
                                   enqueue_many=False,
                                   batch_size=batch_size,
                                   capacity=capacity,
                                   min_after_dequeue=min_after_dequeue,
                                   num_threads=n_threads)

    # alternatively, we could use shuffle_batch_join to use multiple reader
    # instances, or set shuffle_batch's n_threads to higher than 1.

    return batch


def gtzan_music_speech_download(dst='gtzan_music_speech'):
    """Download the GTZAN music and speech dataset.

    Parameters
    ----------
    dst : str, optional
        Location to put the GTZAN music and speech datset.
    """
    path = 'http://opihi.cs.uvic.ca/sound/music_speech.tar.gz'
    download_and_extract_tar(path, dst)


def gtzan_music_speech_load(dst='gtzan_music_speech'):
    """Load the GTZAN Music and Speech dataset.

    Downloads the dataset if it does not exist into the dst directory.

    Parameters
    ----------
    dst : str, optional
        Location of GTZAN Music and Speech dataset.

    Returns
    -------
    Xs, ys : np.ndarray, np.ndarray
        Array of data, Array of labels
    """
    from scipy.io import wavfile

    if not os.path.exists(dst):
        gtzan_music_speech_download(dst)
    music_dir = os.path.join(os.path.join(dst, 'music_speech'), 'music_wav')
    music = [os.path.join(music_dir, file_i)
             for file_i in os.listdir(music_dir)
             if file_i.endswith('.wav')]
    speech_dir = os.path.join(os.path.join(dst, 'music_speech'), 'speech_wav')
    speech = [os.path.join(speech_dir, file_i)
              for file_i in os.listdir(speech_dir)
              if file_i.endswith('.wav')]
    Xs = []
    ys = []
    for i in music:
        sr, s = wavfile.read(i)
        s = s / 16384.0 - 1.0
        re, im = dft.dft_np(s)
        mag, phs = dft.ztoc(re, im)
        Xs.append((mag, phs))
        ys.append(0)
    for i in speech:
        sr, s = wavfile.read(i)
        s = s / 16384.0 - 1.0
        re, im = dft.dft_np(s)
        mag, phs = dft.ztoc(re, im)
        Xs.append((mag, phs))
        ys.append(1)
    Xs = np.array(Xs)
    Xs = np.transpose(Xs, [0, 2, 3, 1])
    ys = np.array(ys)
    return Xs, ys


def cifar10_download(dst='cifar10'):
    """Download the CIFAR10 dataset.

    Parameters
    ----------
    dst : str, optional
        Directory to download into.
    """
    path = 'http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'
    download_and_extract_tar(path, dst)


def cifar10_load(dst='cifar10'):
    """Load the CIFAR10 dataset.

    Downloads the dataset if it does not exist into the dst directory.

    Parameters
    ----------
    dst : str, optional
        Location of CIFAR10 dataset.

    Returns
    -------
    Xs, ys : np.ndarray, np.ndarray
        Array of data, Array of labels
    """
    if not os.path.exists(dst):
        cifar10_download(dst)
    Xs = None
    ys = None
    for f in range(1, 6):
        cf = pickle.load(open(
            '%s/cifar-10-batches-py/data_batch_%d' % (dst, f), 'rb'),
            encoding='LATIN')
        if Xs is not None:
            Xs = np.r_[Xs, cf['data']]
            ys = np.r_[ys, np.array(cf['labels'])]
        else:
            Xs = cf['data']
            ys = cf['labels']
    Xs = np.swapaxes(np.swapaxes(Xs.reshape(-1, 3, 32, 32), 1, 3), 1, 2)
    return Xs, ys


def dense_to_one_hot(labels, n_classes=2):
    """Convert class labels from scalars to one-hot vectors.

    Parameters
    ----------
    labels : array
        Input labels to convert to one-hot representation.
    n_classes : int, optional
        Number of possible one-hot.

    Returns
    -------
    one_hot : array
        One hot representation of input.
    """
    return np.eye(n_classes).astype(np.float32)[labels]


class DatasetSplit(object):
    """Utility class for batching data and handling multiple splits.

    Attributes
    ----------
    current_batch_idx : int
        Description
    images : np.ndarray
        Xs of the dataset.  Not necessarily images.
    labels : np.ndarray
        ys of the dataset.
    n_labels : int
        Number of possible labels
    num_examples : int
        Number of total observations
    """

    def __init__(self, images, labels):
        """Initialize a DatasetSplit object.

        Parameters
        ----------
        images : np.ndarray
            Xs/inputs
        labels : np.ndarray
            ys/outputs
        """
        self.images = np.array(images).astype(np.float32)
        if labels is not None:
            self.labels = np.array(labels).astype(np.int32)
            self.n_labels = len(np.unique(labels))
        else:
            self.labels = None
        self.num_examples = len(self.images)

    def next_batch(self, batch_size=100):
        """Batch generator with randomization.

        Parameters
        ----------
        batch_size : int, optional
            Size of each minibatch.

        Returns
        -------
        Xs, ys : np.ndarray, np.ndarray
            Next batch of inputs and labels (if no labels, then None).
        """
        # Shuffle each epoch
        current_permutation = np.random.permutation(range(len(self.images)))
        epoch_images = self.images[current_permutation, ...]
        if self.labels is not None:
            epoch_labels = self.labels[current_permutation, ...]

        # Then iterate over the epoch
        self.current_batch_idx = 0
        while self.current_batch_idx < len(self.images):
            end_idx = min(
                self.current_batch_idx + batch_size, len(self.images))
            this_batch = {
                'images': epoch_images[self.current_batch_idx:end_idx],
                'labels': epoch_labels[self.current_batch_idx:end_idx]
                if self.labels is not None else None
            }
            self.current_batch_idx += batch_size
            yield this_batch['images'], this_batch['labels']


class Dataset(object):
    """Create a dataset from data and their labels.

    Allows easy use of train/valid/test splits; Batch generator.

    Attributes
    ----------
    all_idxs : list
        All indexes across all splits.
    all_inputs : list
        All inputs across all splits.
    all_labels : list
        All labels across all splits.
    n_labels : int
        Number of labels.
    split : list
        Percentage split of train, valid, test sets.
    test_idxs : list
        Indexes of the test split.
    train_idxs : list
        Indexes of the train split.
    valid_idxs : list
        Indexes of the valid split.
    """

    def __init__(self, Xs, ys=None, split=[1.0, 0.0, 0.0], one_hot=False):
        """Initialize a Dataset object.

        Parameters
        ----------
        Xs : np.ndarray
            Images/inputs to a network
        ys : np.ndarray
            Labels/outputs to a network
        split : list, optional
            Percentage of train, valid, and test sets.
        one_hot : bool, optional
            Whether or not to use one-hot encoding of labels (ys).
        """
        self.all_idxs = []
        self.all_labels = []
        self.all_inputs = []
        self.train_idxs = []
        self.valid_idxs = []
        self.test_idxs = []
        self.n_labels = 0
        self.split = split

        # Now mix all the labels that are currently stored as blocks
        self.all_inputs = Xs
        n_idxs = len(self.all_inputs)
        idxs = range(n_idxs)
        rand_idxs = np.random.permutation(idxs)
        self.all_inputs = self.all_inputs[rand_idxs, ...]
        if ys is not None:
            self.all_labels = ys if not one_hot else dense_to_one_hot(ys)
            self.all_labels = self.all_labels[rand_idxs, ...]
        else:
            self.all_labels = None

        # Get splits
        self.train_idxs = idxs[:round(split[0] * n_idxs)]
        self.valid_idxs = idxs[len(self.train_idxs):
                               len(self.train_idxs) + round(split[1] * n_idxs)]
        self.test_idxs = idxs[len(self.valid_idxs):
                              len(self.valid_idxs) + round(split[2] * n_idxs)]

    @property
    def X(self):
        """Inputs/Xs/Images.

        Returns
        -------
        all_inputs : np.ndarray
            Original Inputs/Xs.
        """
        return self.all_inputs

    @property
    def Y(self):
        """Outputs/ys/Labels.

        Returns
        -------
        all_labels : np.ndarray
            Original Outputs/ys.
        """
        return self.all_labels

    @property
    def train(self):
        """Train split.

        Returns
        -------
        split : DatasetSplit
            Split of the train dataset.
        """
        if len(self.train_idxs):
            inputs = self.all_inputs[self.train_idxs, ...]
            if self.all_labels is not None:
                labels = self.all_labels[self.train_idxs, ...]
            else:
                labels = None
        else:
            inputs, labels = [], []
        return DatasetSplit(inputs, labels)

    @property
    def valid(self):
        """Validation split.

        Returns
        -------
        split : DatasetSplit
            Split of the validation dataset.
        """
        if len(self.valid_idxs):
            inputs = self.all_inputs[self.valid_idxs, ...]
            if self.all_labels is not None:
                labels = self.all_labels[self.valid_idxs, ...]
            else:
                labels = None
        else:
            inputs, labels = [], []
        return DatasetSplit(inputs, labels)

    @property
    def test(self):
        """Test split.

        Returns
        -------
        split : DatasetSplit
            Split of the test dataset.
        """
        if len(self.test_idxs):
            inputs = self.all_inputs[self.test_idxs, ...]
            if self.all_labels is not None:
                labels = self.all_labels[self.test_idxs, ...]
            else:
                labels = None
        else:
            inputs, labels = [], []
        return DatasetSplit(inputs, labels)

    def mean(self):
        """Mean of the inputs/Xs.

        Returns
        -------
        mean : np.ndarray
            Calculates mean across 0th (batch) dimension.
        """
        return np.mean(self.all_inputs, axis=0)

    def std(self):
        """Standard deviation of the inputs/Xs.

        Returns
        -------
        std : np.ndarray
            Calculates std across 0th (batch) dimension.
        """
        return np.std(self.all_inputs, axis=0)
