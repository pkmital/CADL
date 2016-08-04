"""Utility for creating a GIF.

Creative Applications of Deep Learning w/ Tensorflow.
Kadenze, Inc.
Copyright Parag K. Mital, June 2016.
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


def build_gif(imgs, interval=0.1, dpi=72,
              save_gif=True, saveto='animation.gif',
              show_gif=False, cmap=None):
    """Take an array or list of images and create a GIF.

    Parameters
    ----------
    imgs : np.ndarray or list
        List of images to create a GIF of
    interval : float, optional
        Spacing in seconds between successive images.
    dpi : int, optional
        Dots per inch.
    save_gif : bool, optional
        Whether or not to save the GIF.
    saveto : str, optional
        Filename of GIF to save.
    show_gif : bool, optional
        Whether or not to render the GIF using plt.
    cmap : None, optional
        Optional colormap to apply to the images.

    Returns
    -------
    ani : matplotlib.animation.ArtistAnimation
        The artist animation from matplotlib.  Likely not useful.
    """
    imgs = np.asarray(imgs)
    h, w, *c = imgs[0].shape
    fig, ax = plt.subplots(figsize=(np.round(w / dpi), np.round(h / dpi)))
    fig.subplots_adjust(bottom=0)
    fig.subplots_adjust(top=1)
    fig.subplots_adjust(right=1)
    fig.subplots_adjust(left=0)
    ax.set_axis_off()

    if cmap is not None:
        axs = list(map(lambda x: [
            ax.imshow(x, cmap=cmap)], imgs))
    else:
        axs = list(map(lambda x: [
            ax.imshow(x)], imgs))

    ani = animation.ArtistAnimation(
        fig, axs, interval=interval, repeat_delay=0, blit=False)

    if save_gif:
        try:
            ani.save(saveto, writer='imagemagick', dpi=dpi)
        except:
            print('You do not have imagemagick installed.\n\nOn OSX ' +
                  'you can install this by first installing homebrew: ' +
                  'http://brew.sh\nThen run: "brew install imagemagick".\n' +
                  'Windows users can obtain a binary installation here: ' +
                  'https://www.imagemagick.org/script/binary-releases.php\n' +
                  'And Linux users should be able to install imagemagick using ' +
                  'their package manager, e.g.: sudo apt-get install imagemagick.')

    if show_gif:
        plt.show()

    return ani
