"""Utilities used in the Kadenze Academy Course on Deep Learning w/ Tensorflow.

Creative Applications of Deep Learning w/ Tensorflow.
Kadenze, Inc.
Parag K. Mital

Copyright Parag K. Mital, June 2016.
"""
import matplotlib.pyplot as plt
import numpy as np
import zipfile
import os


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
    if img.shape[0] > img.shape[1]:
        extra = (img.shape[0] - img.shape[1]) // 2
        crop = img[extra:-extra, :]
    elif img.shape[1] > img.shape[0]:
        extra = (img.shape[1] - img.shape[1]) // 2
        crop = img[:, extra:-extra]
    else:
        crop = img
    return crop


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


def build_submission(filename, file_list):
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
        assert os.path.exists(file_i), \
            '\nYou are missing the file {}.  '.format(file_i) + \
            'It does not look like you have completed Part {}.'.format(
                part_i + 1)

    # great, each file exists
    print('It looks like you have completed each part!')

    def zipdir(path, zf):
        for root, dirs, files in os.walk(path):
            for file in files:
                # make sure the files are part of the necessary file list
                if file.endswith(file_list):
                    zf.write(os.path.join(root, file))

    # create a zip file with the necessary files
    zipf = zipfile.ZipFile(filename, 'w', zipfile.ZIP_DEFLATED)
    zipdir('.', zipf)
    zipf.close()
    print('Great job!!!')
    print('Now submit the file:\n{}\nto Kadenze for grading!'.format(
        os.path.abspath(filename)))
