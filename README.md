# <a href="https://www.kadenze.com/courses/creative-applications-of-deep-learning-with-tensorflow/info">Creative Applications of Deep Learning w/ Tensorflow</a>

[![coursecard](imgs/cadl-coursecard.png)](https://www.kadenze.com/courses/creative-applications-of-deep-learning-with-tensorflow/info)

[![Slack Channel](https://cadl.herokuapp.com/badge.svg)](https://cadl.herokuapp.com)

This repository contains homework assignments for the <a href="https://www.kadenze.com/partners/kadenze-academy">Kadenze Academy</a> course on <a href="https://www.kadenze.com/courses/creative-applications-of-deep-learning-with-tensorflow/info">Creative Applications of Deep Learning w/ Tensorflow</a>.


# Course Description

This course introduces you to deep learning: the state-of-the-art approach to building artificial intelligence algorithms. We cover the basic components of deep learning, what it means, how it works, and develop code necessary to build various algorithms such as deep convolutional networks, variational autoencoders, generative adversarial networks, and recurrent neural networks. A major focus of this course will be to not only understand how to build the necessary components of these algorithms, but also how to apply them for exploring creative applications. We'll see how to train a computer to recognize objects in an image and use this knowledge to drive new and interesting behaviors, from understanding the similarities and differences in large datasets and using them to self-organize, to understanding how to infinitely generate entirely new content or match the aesthetics or contents of another image. Deep learning offers enormous potential for creative applications and in this course we interrogate what's possible. Through practical applications and guided homework assignments, you'll be expected to create datasets, develop and train neural networks, explore your own media collections using existing state-of-the-art deep nets, synthesize new content from generative algorithms, and understand deep learning's potential for creating entirely new aesthetics and new ways of interacting with large amounts of data.

# Schedule

Course runs July 21, 2016 - December 31, 2016

## Session 1: Introduction To Tensorflow (July 21, 2016) 
We'll cover the importance of data with machine and deep learning algorithms, the basics of creating a dataset, how to preprocess datasets, then jump into Tensorflow, a library for creating computational graphs built by Google Research. We'll learn the basic components of Tensorflow and see how to use it to filter images.

## Session 2: Training A Network W/ Tensorflow (August 2, 2016) 
We'll see how neural networks work, how they are "trained", and see the basic components of training a neural network. We'll then build our first neural network and use it for a fun application of teaching a neural network how to paint an image, and explore such a network can be extended to produce different aesthetics.

## Session 3: Unsupervised And Supervised Learning (August 18, 2016) 
We explore deep neural networks capable of encoding a large dataset, and see how we can use this encoding to explore "latent" dimensions of a dataset or for generating entirely new content. We'll see what this means, how "autoencoders" can be built, and learn a lot of state-of-the-art extensions that make them incredibly powerful. We'll also learn about another type of model that performs discriminative learning and see how this can be used to predict labels of an image.

## Session 4: Visualizing And Hallucinating Representations (August 30, 2016) 
This sessions works with state of the art networks and sees how to understand what "representations" they learn. We'll see how this process actually allows us to perform some really fun visualizations including "Deep Dream" which can produce infinite generative fractals, or "Style Net" which allows us to combine the content of one image and the style of another to produce widely different painterly aesthetics automatically.

## Session 5: Generative Models (September 13, 2016) 
The last session offers a teaser into some of the future directions of generative modeling, including some state of the art models such as the "generative adversarial network", and its implementation within a "variational autoencoder", which allows for some of the best encodings and generative modeling of datasets that currently exist. We also see how to begin to model time, and give neural networks memory by creating "recurrent neural networks" and see how to use such networks to create entirely generative text.

# Github Contents Overview

This github contains lecture transcripts from the Kadenze videos and homeworks contained in Jupyter Notebooks in the following folders:

| | Session | Description |
| --- | --- | --- |
|Installation| **[Installation](#installation-preliminaries)** | Setting up Python/Notebook and necessary Libraries. |
|Preliminaries| **[Preliminaries with Python](session-0)** | Basics of working with Python and images. |
|1| **[Computing with Tensorflow](session-1)** | Working with a small dataset of images.  Dataset preprocessing.  Tensorflow basics.  Sorting/organizing a dataset. |
|2| **[Basics of Neural Networks](session-2)** | Learn how to create a Neural Network.  Learn to use a neural network to paint an image.  Apply creative thinking to the inputs, outputs, and definition of a network. |
|3| **[Unsupervised and Supervised Learning](session-3)** | Build an autoencoder.  Extend it with convolution, denoising, and variational layers.  Build a deep classification network.  Apply softmax and onehot encodings to classify audio using a Deep Convolutional Network. |
|4| **[Visualizing Representations](session-4)** | Visualize backpropped gradients, use them to create Deep Dream, extend Deep Dream w/ regularization.  Stylize images or synthesize new images with painterly or hallucinated aesthetics of another image. |
|5| **[TBA](session-5)** | TBA. |

<a name="installation-preliminaries"></a>
# Installation Preliminaries

<!-- MarkdownTOC autolink=true autoanchor=true bracket=round -->

- [What is Notebook?](#what-is-notebook)
- [Docker Toolbox](#docker-toolbox)
- [Jupyter Notebook](#jupyter-notebook)
- [Navigating to Notebook](#navigating-to-notebook)
- [Installing Python Packages](#installing-python-packages)
- [CUDA/GPU instructions](#cudagpu-instructions)
- [Testing it](#testing-it)
- [Troubleshooting](#troubleshooting)

<!-- /MarkdownTOC -->

We will be using Jupyter Notebook.  This will be necessary for submitting the homeworks and interacting with the guided session notebooks I will provide for each assignment.  Follow along this guide and we'll see how to obtain all of the necessary libraries that we'll be using.  By the end of this, you'll have installed Jupyter Notebook, NumPy, SciPy, and Matplotlib.  While many of these libraries aren't necessary for performing the Deep Learning which we'll get to in later lectures, they are incredibly useful for manipulating data on your computer, preparing data for learning, and exploring results.

<a name="what-is-notebook"></a>
## What is Notebook?

Jupyter Notebook, previously called "iPython Notebook" prior to version 4.0, is a way of interacting with Python code using a web browser.  It is a very useful instructional tool that we will be using for all of our homework assignments.  Notebooks have the file extensions "ipynb" which are abbreviations of "iPython Notebook".  Some websites such as [nbviewer.ipython.org](http://nbviewer.ipython.org) or [www.github.com](http://www.github.com) can view `.ipynb` files directly as rendered HTML.  However, these are not *interactive* versions of the notebook, meaning, they are not running the python kernel which evaluates/interacts with the code.  So the notebook is just a static version of the code contained inside of it.

In order to interact with notebook and start coding, you will need to launch Terminal (for Mac and Linux users).  For Windows users, or for anyone having any problems with the Linux/Mac instructions, please follow the next section on [Docker Toolbox](#docker-toolbox) very closely!  If you are not a Windows user, please first try skipping over the next section and use the installation instructions in [Jupyter Notebook](#jupyter-notebook) before trying Docker as this solution will be much faster than running Docker.

<a name="docker-toolbox"></a>
## Docker Toolbox

Unforunately, at the time of this writing (July 2016), there are no binaries for Tensorflow available for Windows users.  The easiest way to get up an running is to use Docker.  Docker is a way of managing a "virtual" Linux machine on your computer which will aid the creation a machine capable of running Tensorflow.  First, please download and install the Docker Toolbox:

https://www.docker.com/products/docker-toolbox

With this installed, you'll then need to run the "Docker Quickstart Terminal" which will launch a Terminal environment running on a virtual Linux machine on your computer. A virtual machine is basically an emulation of another machine. This is important because we'll use this machine to run Linux and install all of the necessary libraries for running Tensorflow.

Note, if you have trouble launching the Docker Quickstart Terminal because you have "Hyper-V", try one of the following, as suggested by Danilo Gasques:

1) [Setting up a Windows boot option to run without Hyper-V](http://www.hanselman.com/blog/SwitchEasilyBetweenVirtualBoxAndHyperVWithABCDEditBootEntryInWindows81.aspx)

2) [Running Docker on Windows with Hyper-V installed](http://jayvilalta.com/blog/2016/04/28/installing-docker-toolbox-on-windows-with-hyper-v-installed/)

Once the Docker Quickstart Terminal is launched, run the following command (ignoring the `$` sign at the beginning of each line, which just denote that each line is a terminal command that you should type out exactly and then hit ENTER afterwards):

```shell
$ cd
$ docker-machine ip
```

You should see your virtual machine's IP address as a result of the last command.  This is the location of your virtual machine.  <b>NOTE THIS IP ADDRESS</b>, as we'll need it in a second.  

This next command will move to your Windows home directory, then create a new directory called "tensorflow", and then print out what the full path to that directory is.  PLEASE NOTE DOWN THIS DIRECTORY.  This is where everything will happen, and I'll explain that in a minute.

```shell
$ cd
$ mkdir tensorflow
$ echo /$(pwd)/tensorflow
```

Now run the following command, which will download about ~530 MB containing everything we need to run tensorflow, python, and jupyter notebook (again, ignore the "$" at the beginning of the line only)!

```shell
$ docker run -it -p 8888:8888 -p 6006:6006 -v /$(pwd)/tensorflow:/notebooks --name tf pkmital/tf.0.9.0-py.3.4
```

What this is doing is first creating a directory called tensorflow in the home directory, wherever that may be for your computer.  The echo command that we just ran, and I asked you note down, is showing you exactly where that directory is.  So on your Windows machine, you will want to put files inside this directory only when coding w/ Tensorflow.  We will use Docker to mirror that directory on a virutal machine which has everything necessary for us to code in Python and Tensorflow.  _Whatever is in that directory will be mirrored on the virtual machine's directory under `/notebooks`._

You can also try running the docker run command with any other directory. For instance:

```shell
$ docker run -it -p 8888:8888 -p 6006:6006 -v /Users/YOURUSERNAME/Desktop:/notebooks --name tf pkmital/tf.0.9.0-py.3.4
```

Which would mean that your Desktop is where you can move files around so that on the virtual machine, you can interact with them under the `/notebooks`directory.

For OSX users, if you are installing Docker because you had installation problems using Anaconda and pip, you would instead write the following command:

```shell
$ docker run -it -p 8888:8888 -p 6006:6006 -v $(pwd)/Desktop/tensorflow:/notebooks --name tf pkmital/tf.0.9.0-py.3.4
```

This command will download everything you need to run Tensorflow on your virtual machine.

When you want to start this machine, you will launch the Docker Quickstart Terminal and then write:

```shell
$ cd
$ docker start -i tf
```

Notice that the command prompt will now be `#` instead of `$`.  You should have a new folder "tensorflow" inside your Home directory.  This directory will be empty to begin with.  Please make sure you do everything inside this directory only or else any files you make on your virtual machine WILL BE ERASED once it is shutdown!  When you clone the CADL repository, or expand the zip file downloads contents inside this directory via your Windows machine (it will be in your Home directory under a folder "tensorflow"), then you will be able to access it via your Docker instance.

For instance, after running the `docker start -i tf` command, try going into the directory `/notebooks`:

```shell
# cd /notebooks
```

And then git cloning this repo:

```shell
# git clone https://github.com/pkmital/CADL
```

Now, inside the directory `/notebooks/CADL`, you will have this entire repo.  Alternatively, you could download a zip file of this repo and use Windows to place it in the directory you noted down before.

<a name="jupyter-notebook"></a>
## Jupyter Notebook

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
### Windows/Docker Containers

For users running firewalls, you must make sure you have an exception as per [Jupyter Notebooks Firewall Instructions](http://jupyter-notebook.readthedocs.io/en/latest/public_server.html#firewall-setup) otherwise you may not be able to interact with the notebook.  Namely, you will need to allow connections from 127.0.0.1 (localhost) on ports from 49152 to 65535.  Once inside your Docker container as outlined above, you can now launch notebook like so:

```shell
$ cd /notebooks
$ jupyter notebook &
```

Note on Virtual versus Windows Directories:

This is tricky to grasp, mostly because I didn't explain it. Docker is "virtual" computer running inside your computer. It has its own filesystem and its own directories. So you can't reference your Windows machine's directories inside this machine. When you first ran docker (e.g. `$ docker run -it -p 8888:8888 -p 6006:6006 -v /$(pwd)/tensorflow:/notebooks --name tf pkmital/tf.0.9.0-py.3.4`) it included as part of its command: `-v /$(pwd)/tensorflow:/notebooks`. What that was doing is "mirroring" a directory on your Windows machine inside your Virtual machine. So whatever was in your Windows machine under the directory `/$(pwd)/tensorflow` would appear in the Virtual machine under `/notebooks`. That Windows directory is likely `/Users/<YOURUSERNAME>/tensorflow`. So _ONLY_ inside that directory, create it if it doesn't exist, should you put files in order to access it on the Virtual machine.

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

### Ubuntu/Linux 64-bit for Python 3.4

```shell
$ pip3 install --upgrade https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-0.9.0-cp34-cp34m-linux_x86_64.whl
```

### Ubuntu/Linux 64-bit for Python 3.5

```shell
$ pip3 install --upgrade https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-0.9.0-cp35-cp35m-linux_x86_64.whl
```

### OSX for Python 3.4 or Python 3.5

```shell
$ pip3 install --upgrade https://storage.googleapis.com/tensorflow/mac/tensorflow-0.9.0-py3-none-any.whl
```

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

You should see 0.9.0 be printed.

<a name="troubleshooting"></a>
## Troubleshooting

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

### AttributeError: module 'tensorflow' has no attribute '\_\_version\_\_'

You could be running python inside a directory that contains the folder "tensorflow".  Try running python inside a different directory.


### GPU-related issues

If you encounter the following when trying to run a TensorFlow program:

```python
ImportError: libcudart.so.7.0: cannot open shared object file: No such file or directory
```

Make sure you followed the GPU installation [instructions](#optional-install-cuda-gpus-on-linux).
If you built from source, and you left the Cuda or cuDNN version empty, try specifying them
explicitly.

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

### SSLError: SSL_VERIFY_FAILED

If, during pip install from a URL, you encounter an error like:

```bash
...
SSLError: [SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed
```

Solution: Download the wheel manually via curl or wget, and pip install locally.

### Something Else!

Post on the [Forums](https://www.kadenze.com/courses/creative-applications-of-deep-learning-with-tensorflow-i/forums?sort=recent_activity) or check on the Tensorflow [README](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/g3doc/get_started/os_setup.md#pip-installation)
