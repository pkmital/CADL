[![Build Status](https://travis-ci.org/pkmital/CADL.svg?branch=master)](https://travis-ci.org/pkmital/CADL) [![Slack Channel](https://cadl.herokuapp.com/badge.svg)](https://cadl.herokuapp.com)

# <a href="https://www.kadenze.com/courses/creative-applications-of-deep-learning-with-tensorflow/info">Creative Applications of Deep Learning w/ Tensorflow</a>

This repository contains lecture transcripts and homework assignments as Jupyter Notebooks for the <a href="https://www.kadenze.com/partners/kadenze-academy">Kadenze Academy</a> course on <a href="https://www.kadenze.com/courses/creative-applications-of-deep-learning-with-tensorflow/info">Creative Applications of Deep Learning w/ Tensorflow</a>.

This course introduces you to deep learning: the state-of-the-art approach to building artificial intelligence algorithms. We cover the basic components of deep learning, what it means, how it works, and develop code necessary to build various algorithms such as deep convolutional networks, variational autoencoders, generative adversarial networks, and recurrent neural networks. A major focus of this course will be to not only understand how to build the necessary components of these algorithms, but also how to apply them for exploring creative applications. We'll see how to train a computer to recognize objects in an image and use this knowledge to drive new and interesting behaviors, from understanding the similarities and differences in large datasets and using them to self-organize, to understanding how to infinitely generate entirely new content or match the aesthetics or contents of another image. Deep learning offers enormous potential for creative applications and in this course we interrogate what's possible. Through practical applications and guided homework assignments, you'll be expected to create datasets, develop and train neural networks, explore your own media collections using existing state-of-the-art deep nets, synthesize new content from generative algorithms, and understand deep learning's potential for creating entirely new aesthetics and new ways of interacting with large amounts of data.

[Join our Slack channel.](https://cadl.herokuapp.com)

# Schedule

## Session 1: Introduction To Tensorflow
We'll cover the importance of data with machine and deep learning algorithms, the basics of creating a dataset, how to preprocess datasets, then jump into Tensorflow, a library for creating computational graphs built by Google Research. We'll learn the basic components of Tensorflow and see how to use it to filter images.

## Session 2: Training A Network W/ Tensorflow
We'll see how neural networks work, how they are "trained", and see the basic components of training a neural network. We'll then build our first neural network and use it for a fun application of teaching a neural network how to paint an image, and explore such a network can be extended to produce different aesthetics.

## Session 3: Unsupervised And Supervised Learning
We explore deep neural networks capable of encoding a large dataset, and see how we can use this encoding to explore "latent" dimensions of a dataset or for generating entirely new content. We'll see what this means, how "autoencoders" can be built, and learn a lot of state-of-the-art extensions that make them incredibly powerful. We'll also learn about another type of model that performs discriminative learning and see how this can be used to predict labels of an image.

## Session 4: Visualizing And Hallucinating Representations
This sessions works with state of the art networks and sees how to understand what "representations" they learn. We'll see how this process actually allows us to perform some really fun visualizations including "Deep Dream" which can produce infinite generative fractals, or "Style Net" which allows us to combine the content of one image and the style of another to produce widely different painterly aesthetics automatically.

## Session 5: Generative Models
The last session offers a teaser into some of the future directions of generative modeling, including some state of the art models such as the "generative adversarial network", and its implementation within a "variational autoencoder", which allows for some of the best encodings and generative modeling of datasets that currently exist. We also see how to begin to model time, and give neural networks memory by creating "recurrent neural networks" and see how to use such networks to create entirely generative text.

# Github Contents Overview

This github contains lecture transcripts from the Kadenze videos and homeworks contained in Jupyter Notebooks in the following folders:

| | Session | Description | Transcript | Homework |
| --- | --- | --- | --- | --- |
|Installation| **[Installation](#installation-preliminaries)** | Setting up Python/Notebook and necessary Libraries. | N/A | N/A |
|Preliminaries| **[Preliminaries with Python](session-0)** | Basics of working with Python and images. | N/A | N/A |
|1| **[Computing with Tensorflow](session-1)** | Working with a small dataset of images.  Dataset preprocessing.  Tensorflow basics.  Sorting/organizing a dataset. | [lecture-1.ipynb](session-1/lecture-1.ipynb) | [session-1.ipynb](session-1/session-1.ipynb) |
|2| **[Basics of Neural Networks](session-2)** | Learn how to create a Neural Network.  Learn to use a neural network to paint an image.  Apply creative thinking to the inputs, outputs, and definition of a network. | [lecture-2.ipynb](session-2/lecture-2.ipynb) | [session-2.ipynb](session-2/session-2.ipynb) |
|3| **[Unsupervised and Supervised Learning](session-3)** | Build an autoencoder.  Extend it with convolution, denoising, and variational layers.  Build a deep classification network.  Apply softmax and onehot encodings to classify audio using a Deep Convolutional Network. | [lecture-3.ipynb](session-3/lecture-3.ipynb) | [session-3.ipynb](session-3/session-3.ipynb) |
|4| **[Visualizing Representations](session-4)** | Visualize backpropped gradients, use them to create Deep Dream, extend Deep Dream w/ regularization.  Stylize images or synthesize new images with painterly or hallucinated aesthetics of another image. | [lecture-4.ipynb](session-4/lecture-4.ipynb) | [session-4.ipynb](session-4/session-4.ipynb) |
|5| **[Generative Models](session-5)** | Build a Generative Adversarial Network and extend it with a Variational Autoencoder.  Use the latent space of this network to perform latent arithmetic.  Build a character level Recurrent Neural Network using LSTMs.  Understand different ways of inferring with Recurrent Networks.  | [lecture-5.ipynb](session-5/lecture-5.ipynb) | [session-5-part-1.ipynb](session-5/session-5-part-1.ipynb), [session-5-part-2.ipynb](session-5/session-5-part-2.ipynb) |

<a name="installation-preliminaries"></a>
# Installation Preliminaries

<!-- MarkdownTOC autolink=true autoanchor=true bracket=round -->

- [Quickstart Guide](#quickstart-guide)
    - [pip Install](#pip-install)
    - [Docker Installation](#docker-installation)
- [What is Notebook?](#what-is-notebook)
- [Docker Toolbox](#docker-toolbox)
- [Jupyter Notebook](#jupyter-notebook)
    - [OSX/Linux](#osxlinux)
    - [Windows/Docker Containers](#windowsdocker-containers)
- [Navigating to Notebook](#navigating-to-notebook)
- [Installing Python Packages](#installing-python-packages)
    - [Ubuntu/Linux 64-bit for Python 3.4](#ubuntulinux-64-bit-for-python-34)
    - [Ubuntu/Linux 64-bit for Python 3.5](#ubuntulinux-64-bit-for-python-35)
    - [OSX for Python 3.4 or Python 3.5](#osx-for-python-34-or-python-35)
    - [Other Linux/OSX varieties](#other-linuxosx-varieties)
- [CUDA/GPU instructions](#cudagpu-instructions)
- [Testing it](#testing-it)
- [CUDA/GPU instructions for MacOS](#cudagpu-instructions-for-macos)
- [Troubleshooting](#troubleshooting)
    - [ImportError: No module named 'tensorflow'](#importerror-no-module-named-tensorflow)
    - [AttributeError: module 'tensorflow' has no attribute '\_\_version\_\_'](#attributeerror-module-tensorflow-has-no-attribute-\\version\\)
    - [GPU-related issues](#gpu-related-issues)
    - [Protobuf library related issues](#protobuf-library-related-issues)
    - [Cannot import name 'descriptor'](#cannot-import-name-descriptor)
    - [Can't find setup.py](#cant-find-setuppy)
    - [SSLError: SSL_VERIFY_FAILED](#sslerror-sslverifyfailed)
    - [Jupyter Notebook Kernel is always busy \(Windows\)](#jupyter-notebook-kernel-is-always-busy-windows)
    - [Something Else!](#something-else)

<!-- /MarkdownTOC -->

We will be using Jupyter Notebook.  This will be necessary for submitting the homeworks and interacting with the guided session notebooks I will provide for each assignment.  Follow along this guide and we'll see how to obtain all of the necessary libraries that we'll be using.  By the end of this, you'll have installed Jupyter Notebook, NumPy, SciPy, and Matplotlib.  While many of these libraries aren't necessary for performing the Deep Learning which we'll get to in later lectures, they are incredibly useful for manipulating data on your computer, preparing data for learning, and exploring results.

<a name="quickstart-guide"></a>
## Quickstart Guide

Important! Please skip this section and read the rest of this readme if you are unfamiliar w/ Jupyter Notebook or installing Python libraries.  This section is only for advanced users who want to get started quickly.

There are two ways to get started.  You can use a native pip installation or use Docker.  There is a quickstart guide for both methods below.  If you have trouble with these, then please skip to the more in depth guides below these sections.

<a name="pip-install"></a>
### pip Install

For those of you that are proficient w/ Python programming, you'll need Python 3.4+ and the latest TensorFlow which you can install via pip, e.g.:

```bash
$ pip install tensorflow
```

or w/ CUDA as:

```bash
$ pip install tensorflow-gpu
```

<a name="docker-installation"></a>
### Docker Installation

If you want a controlled environment w/ all dependencies installed for you, and are proficient w/ Docker and Jupyter, you can get started w/ this repo like so:

```bash
$ cd
$ git clone https://github.com/pkmital/CADL.git
$ cd CADL
$ docker build -t cadl .
$ docker run -it -p 8888:8888 -p 6006:6006 -v /$(pwd)/session-1:/notebooks --name tf cadl /bin/bash
```

Note that you can skip the build step and download from docker hub instead like so:

```bash
$ docker run -it -p 8888:8888 -p 6006:6006 -v /$(pwd)/session-1:/notebooks --name tf pkmital/cadl /bin/bash
```

Be sure to replace "session-1" with whichever session you are working on, e.g. "session-2", "session-3"...  This will give you a bash prompt with the files for each session:

```bash
root@39c4441bcde8:/notebooks# ls
README.md  lecture-1.ipynb  libs  session-1.ipynb  tests
```

Which you can use to launch jupyter like so:

```bash
root@39c4441bcde8:/notebooks# jupyter notebook
[I 01:45:27.712 NotebookApp] [nb_conda_kernels] enabled, 2 kernels found
[I 01:45:27.715 NotebookApp] Writing notebook server cookie secret to /root/.local/share/jupyter/runtime/notebook_cookie_secret
[W 01:45:27.729 NotebookApp] WARNING: The notebook server is listening on all IP addresses and not using encryption. This is not recommended.
[I 01:45:27.799 NotebookApp] [nb_anacondacloud] enabled
[I 01:45:27.802 NotebookApp] [nb_conda] enabled
[I 01:45:27.856 NotebookApp] ✓ nbpresent HTML export ENABLED
[W 01:45:27.856 NotebookApp] ✗ nbpresent PDF export DISABLED: No module named 'nbbrowserpdf'
[I 01:45:27.858 NotebookApp] Serving notebooks from local directory: /notebooks
[I 01:45:27.858 NotebookApp] 0 active kernels
[I 01:45:27.858 NotebookApp] The Jupyter Notebook is running at: http://[all ip addresses on your system]:8888/?token=dd68eeffd8f227dd789327c981d16b24631866e909bd6469
[I 01:45:27.858 NotebookApp] Use Control-C to stop this server and shut down all kernels (twice to skip confirmation).
```

Jupyter should then be running if you navigate Google Chrome (suggested!) to "http://localhost:8888".  If you navigate to the session-1.ipynb file, you will see the homework, or to "lecture-1.ipynb", to find the lecture transcripts.  The same goes for every other session.

If you need to relaunch the docker image again, you can write:

```bash
$ cd
$ cd CADL
$ docker start -i tf
```

If you want to use a GPU version, and have a Linux machine, and have an NVIDIA GPU, you can use [nvidia-docker](https://github.com/NVIDIA/nvidia-docker) (this only works for Linux machines! for non-Linux machines that want to use GPU, please follow the expanded directions below, or the quickstart pip installation above):

```bash
$ wget -P /tmp https://github.com/NVIDIA/nvidia-docker/releases/download/v1.0.0-rc.3/nvidia-docker_1.0.0.rc.3-1_amd64.deb
$ sudo dpkg -i /tmp/nvidia-docker*.deb && rm /tmp/nvidia-docker*.deb
$ nvidia-docker build -t cadl-gpu -f Dockerfile-gpu .
$ nvidia-docker run -it -p 8888:8888 -p 6006:6006 -v /$(pwd)/session-1:/notebooks --name tf cadl-gpu /bin/bash 
$ nvidia-docker start -i tf
```

If you had any trouble w/ this setup then please go through the rest of this document which provides much more in depth details.


<a name="what-is-notebook"></a>
## What is Notebook?

Jupyter Notebook, previously called "iPython Notebook" prior to version 4.0, is a way of interacting with Python code using a web browser.  It is a very useful instructional tool that we will be using for all of our homework assignments.  Notebooks have the file extensions "ipynb" which are abbreviations of "iPython Notebook".  Some websites such as [nbviewer.ipython.org](http://nbviewer.ipython.org) or [www.github.com](http://www.github.com) can view `.ipynb` files directly as rendered HTML.  However, these are not *interactive* versions of the notebook, meaning, they are not running the python kernel which evaluates/interacts with the code.  So the notebook is just a static version of the code contained inside of it.

In order to interact with notebook and start coding, you will need to launch Terminal (for Mac and Linux users).  For Windows users, or for anyone having any problems with the Linux/Mac instructions, please follow the next section on [Docker Toolbox](#docker-toolbox) very closely!  If you are not a Windows user, please first try skipping over the next section and use the installation instructions in [Jupyter Notebook](#jupyter-notebook) before trying Docker as this solution will be much faster than running Docker.

<a name="docker-toolbox"></a>
## Docker Toolbox

Currently, Windows users can only install Tensorflow via [pip using a 64-bit Python 3.5 environment](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/g3doc/get_started/os_setup.md#pip-installation-on-windows) or using Docker, as outlined below.

The easiest way to get up an running on any type of system is to use Docker.  Docker is a way of managing a "virtual" Linux machine on your computer which will aid the creation a machine capable of running Tensorflow.  First, please download and install the Docker Toolbox:

https://www.docker.com/products/docker-toolbox

Linux users can install docker using their favorite package manager.

For OSX and Windows users, you'll then need to run the "Docker Quickstart Terminal" which will launch a Terminal environment running on a virtual Linux machine on your computer. A virtual machine is basically an emulation of another machine. This is important because we'll use this machine to run Linux and install all of the necessary libraries for running Tensorflow.

Note for Windows users, if you have trouble launching the Docker Quickstart Terminal because you have "Hyper-V", please instead try using https://docs.docker.com/docker-for-windows/.  Then launch the newly installed "Docker CLI" program.

Once the Terminal is launched, either via Docker CLI or Docker Quickstart Terminal, run the following command (ignoring the `$` sign at the beginning of each line, which just denote that each line is a terminal command that you should type out exactly and then hit ENTER afterwards):

```shell
$ cd
$ docker-machine ip
```

If you are using Docker Toolbox, you should see your virtual machine's IP address as a result of the last command.  This is the location of your virtual machine.  <b>NOTE THIS IP ADDRESS</b>, as we'll need it in a second.  If you are using "Docker for Windows" instead, then you won't need this IP as we'll just use "localhost".

This next command will move to your "home" directory.  We'll then "clone" the github repo.  This will download everything for the course using "git".  If you have trouble w/ this step, make sure you have installed [git](https://git-scm.com/downloads).

```shell
$ cd
$ git clone https://github.com/pkmital/CADL.git
```

We'll now print out what the full path to that directory is.  PLEASE NOTE DOWN THIS DIRECTORY.  This is where everything will happen, and I'll explain that in a minute.

```shell
$ echo /$(pwd)/CADL
```

Now run the following command, which will download everything we need to run tensorflow, python, and jupyter notebook (again, ignore the "$" at the beginning of the line only)!

```shell
$ docker run -it -p 8888:8888 -p 6006:6006 -v /$(pwd)/CADL:/notebooks --name tf pkmital/cadl
```

What this is doing is:
    * Running the docker image [pkmital/cadl](https://hub.docker.com/r/pkmital/cadl/)
    * --name is giving it a shorthand name of "tf"
    * -v is mirroring the directory "/$(pwd)/CADL" to the virtual machine's directory of "/notebooks"
    * -p is forwarding ports from the virtual machine to your local machine so that you can access the virtual machine's port
    * -it is running it as an interactive process

You will want to put files inside the "/notebooks" directory *only*.  If you place files on the virtual machine outside of the "/notebooks" directory, which is the SAME as the "CADL" directory on your local machine, they will *not* be saved.  We are using Docker to mirror the "CADL" directory on a virtual machine which has everything necessary for us to code in Python and Tensorflow.  _Whatever is in that directory will be mirrored on the virtual machine's directory under `/notebooks`._

You can also try running the docker run command with any other directory. For instance:

```shell
$ docker run -it -p 8888:8888 -p 6006:6006 -v /Users/YOURUSERNAME/Desktop:/notebooks --name tf pkmital/cadl
```

Which would mean that your Desktop is where you can move files around so that on the virtual machine, you can interact with them under the `/notebooks`directory.

For OSX users, if you are installing Docker because you had installation problems using Anaconda and pip, you would instead write the following command (note the missing slash):

```shell
$ docker run -it -p 8888:8888 -p 6006:6006 -v $(pwd)/CADL:/notebooks --name tf pkmital/cadl
```

When you want to start this machine, you will launch the Docker Quickstart Terminal and then write:

```shell
$ cd
$ docker start -i tf
```

Notice that the command prompt will now be `#` instead of `$`.  You should have a new folder "tensorflow" inside your Home directory.  This directory will be empty to begin with.  Please make sure you do everything inside this directory only or else any files you make on your virtual machine WILL BE ERASED once it is shutdown!  When you clone the CADL repository, or expand the zip file downloads contents inside this directory via your Windows machine (it will be in your Home directory under a folder "cadl"), then you will be able to access it via your Docker instance.

For instance, after running the `docker start -i tf` command, try going into the directory `/notebooks`:

```shell
# cd /notebooks
```

<a name="jupyter-notebook"></a>
## Jupyter Notebook

<a name="osxlinux"></a>
### OSX/Linux

Note: Windows/Docker users should scroll past this section to ["Windows/Docker"](#windows-docker-containers).  For OSX/Linux users, the easiest way to ensure you have Python 3.4 or higher and Jupter Notebook is to install Anaconda for Python 3.5 located here:

[OSX](https://docs.continuum.io/anaconda/install#anaconda-for-os-x-command-line-install) or [Linux](https://docs.continuum.io/anaconda/install#linux-install)

Make sure you restart your Terminal after you install Anaconda as there are some PATH variables that have to be set.

Then run the following:

```shell
$ curl https://bootstrap.pypa.io/ez_setup.py -o - | python
```

If you already have conda, but only have Python 2, you can very easily [add a new environment w/ Python 3](http://conda.pydata.org/docs/py2or3.html#create-a-python-3-5-environment) and switch back and forth as needed.  Or if you do not have Anaconda, but have a system based install, I'd really recommend either using Anaconda or [pyenv](https://github.com/yyuu/pyenv) to help you manage both python installations.

With Anaconda installed, you will have python and the package "ipython[notebook]", along with a ton of other very useful packages such as numpy, matplotlib, scikit-learn, scikit-image, and many others.

With everything installed, restart your Terminal application (on OSX, you can use Spotlight to find the Terminal application), and then navigate to the directory containing the "ipynb", or "iPython Notebook" file, by "cd'ing" (pronounced, see-dee-ing), into that directory.  This involves typing the command: "cd some_directory".  Once inside the directory of the notebook file, you will then type: "jupyter notebook".  If this command does not work, it means you do not have notebook installed!  Try installed anaconda as above, restart your Terminal application, or manually install notebook like so (ignore the "$" signs which just denote that this is a Terminal command that you should type out exactly and then hit ENTER!):

```shell
$ pip3 install ipython[notebook]
$ jupyter notebook
```

If you run into issues that say something such as:

```
[W 20:37:40.543 NotebookApp] Kernel not found: None
```

Then please try first running:

```shell
$ ipython3 kernel install
```

<a name="windows-docker-containers">
<a name="windowsdocker-containers"></a>
### Windows/Docker Containers

For users running firewalls, you must make sure you have an exception as per [Jupyter Notebooks Firewall Instructions](http://jupyter-notebook.readthedocs.io/en/latest/public_server.html#firewall-setup) otherwise you may not be able to interact with the notebook.  Namely, you will need to allow connections from 127.0.0.1 (localhost) on ports from 49152 to 65535.  Once inside your Docker container as outlined above, you can now launch notebook like so:

```shell
$ cd /notebooks
$ jupyter notebook &
```

Note on Virtual versus Windows Directories:

This is tricky to grasp, mostly because I didn't explain it. Docker is "virtual" computer running inside your computer. It has its own filesystem and its own directories. So you can't reference your Windows machine's directories inside this machine. When you first ran docker (e.g. `$ docker run -it -p 8888:8888 -p 6006:6006 -v /$(pwd)/tensorflow:/notebooks --name tf pkmital/cadl`) it included as part of its command: `-v /$(pwd)/tensorflow:/notebooks`. What that was doing is "mirroring" a directory on your Windows machine inside your Virtual machine. So whatever was in your Windows machine under the directory `/$(pwd)/tensorflow` would appear in the Virtual machine under `/notebooks`. That Windows directory is likely `/Users/<YOURUSERNAME>/tensorflow`. So _ONLY_ inside that directory, create it if it doesn't exist, should you put files in order to access it on the Virtual machine.

So let's say your Username was "pkmital". Then your home directory would be `/Users/pkmital`, and you would have mirrored `/Users/pkmital/tensorflow` on your Windows Machine to the Virtual machine under `/notebook`. Now let's say I create a directory `/Users/pkmital/tensorflow/images` on my Windows Machine, and then put a bunch of png files in there. I will then see them in my Virtual machine under `/notebook/images`.  If I put the CADL repository inside `/Users/pkmital/tensorflow`, then I should have `/Users/pkmital/tensorflow/CADL/session-1/session-1.ipynb` and on the Virtual machine, it will be in `/notebooks/CADL/session-1/session-1.ipynb` - From this notebook, running on the virtual machine, accessed with Jupyter Notebook, I would access my images like so:

```python
import os
os.listdir('../../images')
```

<a name="navigating-to-notebook"></a>
## Navigating to Notebook

After running "jupyter notebook &", you should see a message similar to:

```shell
root@182bd64f27d2:~# jupyter notebook &
[I 21:15:33.647 NotebookApp] Writing notebook server cookie secret to /root/.local/share/jupyter/runtime/notebook_cookie_secret
[W 21:15:33.712 NotebookApp] WARNING: The notebook server is listening on all IP addresses and not using encryption. This is not recommended.
[W 21:15:33.713 NotebookApp] WARNING: The notebook server is listening on all IP addresses and not using authentication. This is highly insecure and not recommended.
[I 21:15:33.720 NotebookApp] Serving notebooks from local directory: /root
[I 21:15:33.721 NotebookApp] 0 active kernels
[I 21:15:33.721 NotebookApp] The IPython Notebook is running at: http://[all ip addresses on your system]:8888/
[I 21:15:33.721 NotebookApp] Use Control-C to stop this server and shut down all kernels (twice to skip confirmation).
```

Don't worry if the IP address or command prompt look different.  Note where it says: `The IPython Notebook is running at`.  If you are running Docker (Windows users), this is where we need that IP address.  For OSX/Linux users, we'll use "localhost" so don't worry about this.  Now open up Chrome/Safari/Firefox whatever browser you like, and then navigate to:

http://localhost:8888

or for Windows users:

http://ADDRESS:8888

where ADDRESS is the ip address you should have noted down before. For instance, on my machine, I would visit the website:

http://192.168.99.100:8888

This will launch the Jupyter Notebook where you will be able to interact with the homework assignments!

<a name="installing-python-packages"></a>
## Installing Python Packages

Packages are libraries or useful extensions to the standard python libraries.  In this course, we'll be using a few including Tensorflow, NumPy, MatPlotLib, SciPy, SciKit-Image, and SciKit-Learn.  Windows users will already have these libraries since the Docker container includes these.  However, if you needed to, you can install these using "pip", which is the python package manager.  OSX/Linux users should follow these steps just to be sure they have the latest versions of these packages. In Python 3.4 and higher, `pip` comes with any standard python installation.  In order to use `pip`, first make sure you are using the correct version.  One way to do this is check which pip you are running:

```shell
$ which pip
$ which pip3
```

Use which `pip` points to the install path that makes the most sense (e.g. Anaconda for OSX users for some reason does not symlink pip3 to the python3 pip, and instead points to the system version of python3).

Then you'll write:

```shell
$ pip3 install -U pip setuptools
```

To make sure you have an up to date pip, then:

```shell
$ pip3 install some_package
```

To get the necessary libraries:

```shell
$ pip3 install "scikit-image>=0.11.3" "numpy>=1.11.0" "matplotlib>=1.5.1" "scikit-learn>=0.17"
```

This should get you all of the libraries we need for the course, EXCEPT for tensorflow.  Tensorflow is a special case, but can be `pip` installed in much the same way by pointing pip to the github repo corresponding to your OS like so.

<a name="ubuntulinux-64-bit-for-python-34"></a>
### Ubuntu/Linux 64-bit for Python 3.4

```shell
$ pip3 install --upgrade https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-0.11.0rc1-cp34-cp34m-linux_x86_64.whl
```

<a name="ubuntulinux-64-bit-for-python-35"></a>
### Ubuntu/Linux 64-bit for Python 3.5

```shell
$ pip3 install --upgrade https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-0.11.0rc1-cp35-cp35m-linux_x86_64.whl
```

<a name="osx-for-python-34-or-python-35"></a>
### OSX for Python 3.4 or Python 3.5

```shell
$ pip3 install --upgrade https://storage.googleapis.com/tensorflow/mac/cpu/tensorflow-0.11.0rc1-py3-none-any.whl
```

<a name="other-linuxosx-varieties"></a>
### Other Linux/OSX varieties

You can pip install Tensorflow for most OSX/Linux setups including those that are making use of NVIDIA GPUs and CUDA using one the packages listed on this link:
https://github.com/tensorflow/tensorflow/blob/master/tensorflow/g3doc/get_started/os_setup.md#pip-installation

If you are having trouble with pip installation, try looking here first: [Common Installation Problems](https://github.com/tensorflow/tensorflow/blob/37451589519d15207448dc2d9b1c0309de15d8db/tensorflow/g3doc/get_started/os_setup.md#common-problems).  Failing that, reach out to us on the forums, or else you may want to instead run a Docker instance as outlined in the Windows instructions above: [Setting up a Docker Container](#docker-toolbox).

<a name="cudagpu-instructions"></a>
## CUDA/GPU instructions

Note that I have not provided instructions on getting setup w/ CUDA as it is beyond the scope of this course!  If you are interested in using GPU acceleration, I highly recommend using Ubuntu Linux and setting up a machine on [Nimbix](https://www.nimbix.net) or [Amazon EC2](https://aws.amazon.com/ec2/
) using the instructions here: https://github.com/tensorflow/tensorflow/blob/master/tensorflow/g3doc/get_started/os_setup.md#optional-install-cuda-gpus-on-linux.  If you're using Nimbix, you can skip the install process as there is already a machine pre-installed w/ Tensorflow.  Similarly, for Amazon EC2, there are many existing "images" of machines that have Tensorflow already installed.


<a name="testing-it"></a>
## Testing it

To confirm it worked, try running:

```shell
$ python3 -c 'import tensorflow as tf; print(tf.__version__)'
```

You should see 0.9.0 or 0.10.0 or 0.11.0rc1 printed, depending on which version you have installed.


<a name="cudagpu-instructions-for-macos"></a>
## CUDA/GPU instructions for MacOS

When your Mac is equipped with a NVidia graphics card, you can use the GPU for computing with Tensorflow. GPU enabled computing is not supported for Macs with ATI or Intel graphics cards. 

If you have a previous cpu installation of tensorflow, uninstall it first:

```
$ pip3 uninstall tensorflow
```

Using homebrew, install the following packages:

```
$ brew install coreutils
$ brew tap caskroom/cask
$ brew cask install cuda
```
Once you have the CUDA Toolkit installed you will need to setup the required environment variables by adding the 
following to your `~/.profile`:
```
export CUDA_HOME=/usr/local/cuda
export DYLD_LIBRARY_PATH="$DYLD_LIBRARY_PATH:$CUDA_HOME/lib"
export PATH="$CUDA_HOME/bin:$PATH"
```
Tensorflow needs the library libcuda.1.dylib, so we have to create an additional symbolic link:
```
sudo ln -sf /usr/local/cuda/lib/libcuda.dylib /usr/local/cuda/lib/libcuda.1.dylib
```
Finally, you will also want to install the **CUDA Deep Neural Network** (cuDNN v5) library which currently requires an 
[_Accelerated Computing Developer Program_](https://developer.nvidia.com/cudnn) account. Once you have it downloaded 
locally, you can unzip and move the header and libraries to your local CUDA Toolkit folder:
```
$ sudo mv include/cudnn.h /Developer/NVIDIA/CUDA-8.0/include/
$ sudo mv lib/libcudnn* /Developer/NVIDIA/CUDA-8.0/lib
$ sudo ln -s /Developer/NVIDIA/CUDA-8.0/lib/libcudnn* /usr/local/cuda/lib/
```
Then, finally, install tensorflow with GPU support with:
```
$ export TF_BINARY_URL=https://storage.googleapis.com/tensorflow/mac/gpu/tensorflow_gpu-0.12.0rc0-py3-none-any.whl`
$ pip3 install --ignore-installed --upgrade $TF_BINARY_URL
```

According to the instructions of the TensorFlow website, this should work. However, on MacOS 10.11 (El Capitan) and 
above, the environment variable `DYLD_LIBRARY_PATH` is ignored, resulting in an error in the interactive python console 
and JetBrains PyCharm IDE. The dynamic library `libcudart.8.0.dylib` fails to load. This
is due to a new protection meganism in MacOS 10.11 and higher. El Capitan ships with a new OS X feature: System 
Integrity Protection (SIP), also known as “rootless” mode. This reduces the attack surface for malware that relies on 
modifying system files by preventing any user, whether with system administrator (“root”) privileges or not from 
modifying a number of operating system directories and files.

**Warning:** The point of SIP is to prevent malware and other unwanted modifications into system files. Consider whether 
or not you want to dispense with this protection.
Follow these steps to disable SIP:

* Restart your Mac.
* Before OS X starts up, hold down Command-R and keep it held down until you see an Apple icon and a progress bar. Release. This boots you into Recovery.
* From the Utilities menu, select Terminal.
* At the prompt type exactly the following and then press Return: `csrutil disable`
* Terminal should display a message that SIP was disabled.
* From the  menu, select Restart.

You can re-enable SIP by following the above steps, but using `csrutil enable` instead.



<a name="troubleshooting"></a>
## Troubleshooting

<a name="importerror-no-module-named-tensorflow"></a>
### ImportError: No module named 'tensorflow'

You may have different versions of Python installed.  You can troubleshoot this by looking at the output of:

```shell
$ which python3
$ which pip3
$ python3 --version
$ pip3 --version
$ which python
$ which pip
$ python --version
$ pip --version
```

You may simply need to install tensorflow using `pip` instead of `pip3` and/or use `python` instead of `python3`, assuming they point to a version of python which is Python 3 or higher.

<a name="attributeerror-module-tensorflow-has-no-attribute-\\version\\"></a>
### AttributeError: module 'tensorflow' has no attribute '\_\_version\_\_'

You could be running python inside a directory that contains the folder "tensorflow".  Try running python inside a different directory.


<a name="gpu-related-issues"></a>
### GPU-related issues

If you encounter the following when trying to run a TensorFlow program:

```python
ImportError: libcudart.so.7.0: cannot open shared object file: No such file or directory
```

Make sure you followed the GPU installation [instructions](#optional-install-cuda-gpus-on-linux).
If you built from source, and you left the Cuda or cuDNN version empty, try specifying them
explicitly.

<a name="protobuf-library-related-issues"></a>
### Protobuf library related issues

TensorFlow pip package depends on protobuf pip package version
3.0.0b2. Protobuf's pip package downloaded from [PyPI](https://pypi.python.org)
(when running `pip install protobuf`) is a Python only library, that has
Python implementations of proto serialization/deserialization which can be 10x-50x
slower than the C++ implementation. Protobuf also supports a binary extension
for the Python package that contains fast C++ based proto parsing. This
extension is not available in the standard Python only PIP package. We have
created a custom binary pip package for protobuf that contains the binary
extension. Follow these instructions to install the custom binary protobuf pip
package :

```bash
# Ubuntu/Linux 64-bit:
$ pip install --upgrade https://storage.googleapis.com/tensorflow/linux/cpu/protobuf-3.0.0b2.post2-cp27-none-linux_x86_64.whl

# Mac OS X:
$ pip install --upgrade https://storage.googleapis.com/tensorflow/mac/protobuf-3.0.0b2.post2-cp27-none-any.whl
```

and for Python 3 :

```bash
# Ubuntu/Linux 64-bit:
$ pip3 install --upgrade https://storage.googleapis.com/tensorflow/linux/cpu/protobuf-3.0.0b2.post2-cp34-none-linux_x86_64.whl

# Mac OS X:
$ pip3 install --upgrade https://storage.googleapis.com/tensorflow/mac/protobuf-3.0.0b2.post2-cp35-none-any.whl
```

Install the above package _after_ you have installed TensorFlow via pip, as the
standard `pip install tensorflow` would install the python only pip package. The
above pip package will over-write the existing protobuf package.
Note that the binary pip package already has support for protobuf larger than
64MB, that should fix errors such as these :

```bash
[libprotobuf ERROR google/protobuf/src/google/protobuf/io/coded_stream.cc:207] A
protocol message was rejected because it was too big (more than 67108864 bytes).
To increase the limit (or to disable these warnings), see
CodedInputStream::SetTotalBytesLimit() in google/protobuf/io/coded_stream.h.

```

<a name="cannot-import-name-descriptor"></a>
### Cannot import name 'descriptor'

```python
ImportError: Traceback (most recent call last):
  File "/usr/local/lib/python3.4/dist-packages/tensorflow/core/framework/graph_pb2.py", line 6, in <module>
    from google.protobuf import descriptor as _descriptor
ImportError: cannot import name 'descriptor'
```

If you the above error when upgrading to a newer version of TensorFlow, try
uninstalling both TensorFlow and protobuf (if installed) and re-installing
TensorFlow (which will also install the correct protobuf dependency).

<a name="cant-find-setuppy"></a>
### Can't find setup.py

If, during `pip install`, you encounter an error like:

```bash
...
IOError: [Errno 2] No such file or directory: '/tmp/pip-o6Tpui-build/setup.py'
```

Solution: upgrade your version of pip:

```bash
pip install --upgrade pip
```

This may require `sudo`, depending on how pip is installed.

<a name="sslerror-sslverifyfailed"></a>
### SSLError: SSL_VERIFY_FAILED

If, during pip install from a URL, you encounter an error like:

```bash
...
SSLError: [SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed
```

Solution: Download the wheel manually via curl or wget, and pip install locally.

<a name="jupyter-notebook-kernel-is-always-busy-windows"></a>
### Jupyter Notebook Kernel is always busy (Windows)
If your have installed Docker Toolbox on Windows but your jupyter notebook doesn't run properly (the notebook kernel keeps busy all the time when you open any file) then you might need to try different browsers (One guy tried Edge and it solved his problem after struggling for long time on Chrome/Firefox).

And you should also enable port forwarding by:

1. Open VirtualBox
2. Click on your default docker image.
3. Click Settings.
4. Click Network.
5. Click forward port.
6. Add a new rule named jupyter with host ip=127.0.0.1 and host/guess port=8888
7. Now you should be able to browse your notebook app via localhost:8888 (instead of having to browse 192.168.xx.xx:8888)

<a name="something-else"></a>
### Something Else!

Post on the [Forums](https://www.kadenze.com/courses/creative-applications-of-deep-learning-with-tensorflow-i/forums?sort=recent_activity) or check on the Tensorflow [README](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/g3doc/get_started/os_setup.md#pip-installation)
