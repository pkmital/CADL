# Session 1 - Introduction to Tensorflow
<p class="lead">
Assignment: Creating a Dataset/Computing with Tensorflow
</p>

<p class="lead">
Parag K. Mital
[Creative Applications of Deep Learning w/ Tensorflow](https://www.kadenze.com/courses/creative-applications-of-deep-learning-with-tensorflow/info)
Kadenze Academy
\#CADL</p>

# Learning Goals

<!-- MarkdownTOC autolink=true autoanchor=true bracket=round -->

- [Assignment Synopsis](#assignment-synopsis)
- [Part One - Create a Small Dataset](#part-one---create-a-small-dataset)
  - [Instructions](#instructions)
  - [Code](#code)
- [Part Two - Compute the Mean](#part-two---compute-the-mean)
  - [Instructions](#instructions-1)
  - [Code](#code-1)
- [Part Three - Compute the Standard Deviation](#part-three---compute-the-standard-deviation)
  - [Instructions](#instructions-2)
  - [Code](#code-2)
- [Part Four - Sort the Dataset](#part-four---sort-the-dataset)
  - [Instructions](#instructions-3)
  - [Code](#code-3)
- [Assignment Submission](#assignment-submission)

<!-- /MarkdownTOC -->


<h1>Notebook</h1>

Everything you will need to do will be inside of this notebook, and I've marked which cells you will need to edit by saying <b><font color='red'>"TODO! COMPLETE THIS SECTION!"</font></b>.  For you to work with this notebook, you'll either download the zip file from the resources section on Kadenze or clone the github repo (whichever you are more comfortable with), and then run notebook inside the same directory as wherever this file is located using the command line "jupyter notebook" or "ipython notbeook" (using Terminal on Unix/Linux/OSX, or Command Line/Shell/Powershell on Windows).  If you are unfamiliar with jupyter notebook, please look at the session-0 notebook and/or html file to be sure you have all the necessary libraries and notebook installed.

Once you have launched notebook, this will launch a web browser with the contents of the zip files listed.  Click the file "session-1.ipynb" and this document will open in an interactive notebook, allowing you to "run" the cells, computing them using python, and edit the text inside the cells.

<a name="assignment-synopsis"></a>
# Assignment Synopsis

This first homework assignment will guide you through working with a small dataset of images.  For Part 1, you'll need to find 100 images and use the function I've provided to create a montage of your images, saving it to the file "dataset.png" (template code provided below).  You can load an existing dataset of images, find your own images, or perhaps create your own images using a creative process such as painting, photography, or something along those lines.  Each image will be reshaped to 100 x 100 pixels.  There needs to be at least 100 images.  For Parts 2 and 3, you'll then calculate the mean and deviation of it using a tensorflow session.  For Part 4, you'll need to sort the entire dataset based on its color values or find another method of sorting them.  Finally, the last part will package everything for you in a zip file which you can upload to Kadenze to get assessed (only if you are a Kadenze Premium member, $10 p/m, free for the first month).  If you have any questions, be sure to enroll in the course and ask your peers in the \#CADL community or me on the forums!

https://www.kadenze.com/courses/creative-applications-of-deep-learning-with-tensorflow/info

The following assignment breakdown gives more detailed instructions and includes template code for you to fill out.  Good luck!

<a name="part-one---create-a-small-dataset"></a>
# Part One - Create a Small Dataset

<a name="instructions"></a>
## Instructions

Use Python, Numpy, and Matplotlib to load a dataset of 100 images and create a montage of the dataset as a 10 x 10 image using the function below. You'll need to make sure you call the function using a 4-d array of `N x H x W x C` dimensions, meaning every image will need to be the same size! You can load an existing dataset of images, find your own images, or perhaps create your own images using a creative process such as painting, photography, or something along those lines. The code below will show you how to resize and/or crop your images so that they are 100 pixels x 100 pixels in height and width. Finally, make sure you only use 100 images of any dataset you create or use!  Once you have your 100 images loaded, make sure you use `montage` function to draw and save your dataset to the file <b>dataset.png</b>.

<a name="code"></a>
## Code

This next section will just make sure you have the right version of python and the libraries that we'll be using.  Don't change the code here but make sure you "run" it (use "shift+enter")!


```python
# First check the Python version
import sys
if sys.version_info < (3,4):
    print('You are running an older version of Python!\n\n' \
          'You should consider updating to Python 3.4.0 or ' \
          'higher as the libraries built for this course ' \
          'have only been tested in Python 3.4 and higher.\n')
    print('Try installing the Python 3.5 version of anaconda '
          'and then restart `jupyter notebook`:\n' \
          'https://www.continuum.io/downloads\n\n')

# Now get necessary libraries
try:
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    from skimage.transform import resize
except ImportError:
    print('You are missing some packages! ' \
          'We will try installing them before continuing!')
    !pip install "numpy>=1.11.0" "matplotlib>=1.5.1" "scikit-image>=0.11.3" "scikit-learn>=0.17"
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    from skimage.transform import resize
    print('Done!')
```


```python
# This cell includes the provided libraries from the zip file
try:
    from libs.utils import montage, imcrop_tosquare, build_submission
except ImportError:
    print("Make sure you have started notebook in the same directory" +
          " as the provided zip file which includes the 'libs' folder" +
          " and the file 'utils.py' inside of it.  You will NOT be able"
          " to complete this assignment unless you restart jupyter"
          " notebook inside the directory created by extracting"
          " the zip file or cloning the github repo.")
```


```python
# We'll tell matplotlib to inline any drawn figures like so:
%matplotlib inline
```

Places your images in a folder such as `dirname = '/Users/Someone/Desktop/ImagesFromTheInternet'`.  We'll then use the `os` package to load them and crop/resize them to a standard size of 100 x 100 pixels.

<h3><font color='red'>TODO! COMPLETE THIS SECTION!</font></h3>


```python
# You need to find 100 images from the web/create them yourself
# or find a dataset that interests you (e.g. I used celeb faces
# in the course lecture...)
# then store them all in a single directory.
# With all the images in a single directory, you can then
# perform the following steps to create a 4-d array of:
# N x H x W x C dimensions as 100 x 100 x 100 x 3.

dirname = ...

# Load every file in the provided directory
filenames = [os.path.join(dirname, fname)
             for fname in os.listdir(dirname)
             if os.path.join(dirname, fname).endswith('.png')]

# Make sure we have exactly 100 image files!
filenames = filenames[:100]
assert(len(filenames) == 100)

# Read every filename as an RGB image
imgs = [plt.imread(fname)[..., :3] for fname in filenames]

# Crop every image to a square
imgs = [imcrop_tosquare(img_i) for img_i in imgs]

# Then resize the square image to 100 x 100 pixels
imgs = [resize(img_i, (100, 100)) for img_i in imgs]

# Finally make our list of 3-D images a 4-D array with the first dimension the number of images:
imgs = np.array(imgs)
```

Don't change this section!


```python
# Plot the resulting dataset:
# Make sure you "run" this cell after you create your `imgs` variable as a 4-D array!
# Make sure we have a 100 x 100 x 100 x 3 dimension array
assert(imgs.shape == (100, 100, 100, 3))
plt.figure(figsize=(10, 10))
plt.imshow(montage(imgs, saveto='dataset.png'))
```

<a name="part-two---compute-the-mean"></a>
# Part Two - Compute the Mean

<a name="instructions-1"></a>
## Instructions

First use Tensorflow to define a session.  Then use Tensorflow to create an operation which takes your 4-d array and calculates the mean color image (100 x 100 x 3).  You'll then calculate the mean image by running the operation with your session (e.g. <code>sess.run(...)</code>).  Finally, plot the mean image, save it, and then include this image in your zip file as <b>mean.png</b>.

<a name="code-1"></a>
## Code


```python
try:
    import tensorflow as tf
except ImportError:
    print("You do not have tensorflow installed!")
    print("Follow the instructions on the following link")
    print("to install tensorflow before continuing:")
    print("")
    print("https://github.com/tensorflow/tensorflow/blob/master/tensorflow/g3doc/get_started/os_setup.md#pip-installation")
```

<h3><font color='red'>TODO! COMPLETE THIS SECTION!</font></h3>


```python
# First create a tensorflow session
sess = ...

# Now create an operation that will calculate the mean of your images
mean_img_op = ...

# And then run that operation using your session
mean_img = sess.run(mean_img_op)
```

Don't change this section!


```python
# Then plot the resulting mean image:
# Make sure the mean image is the right size!
assert(mean_img.shape == (100, 100, 3))
plt.figure(figsize=(10, 10))
plt.imshow(mean_img)
plt.imsave(arr=mean_img, fname='mean.png')
```

<a name="part-three---compute-the-standard-deviation"></a>
# Part Three - Compute the Standard Deviation

<a name="instructions-2"></a>
## Instructions

Now use tensorflow to calculate the standard deviation and upload the standard deviation image averaged across color channels as a "jet" heatmap of the 100 images.  This will be a little more involved as there is no operation in tensorflow to do this for you.  However, you can do this by calculating the mean image of your dataset as a 4-D array.  To do this, you could write e.g.  `mean_img = tf.reduce_mean(images, reduction_indices=0, keep_dims=True)` to give you a `1 x H x W x C` dimension array calculated on the `N x H x W x C` images variable.  The reduction_indices parameter is saying to keep the 0th dimension.  This way, you can write `images - mean_img` to give you a `N x H x W x C` dimension variable, with every image in your images array having been subtracted by the `mean_img`.  If you calculate the square root of the sum of the squared differences of this resulting operation, you have your standard deviation!

In summary, you'll need to write something like: `subtraction = images - tf.reduce_mean(images, reduction_indices=0, keep_dims=True)`, then reduce this operation using `tf.sqrt(tf.reduce_sum(subtraction * subtraction, reduction_indices=0))` to get your standard deviation then include this image in your zip file as <b>std.png</b>

<a name="code-2"></a>
## Code

<h3><font color='red'>TODO! COMPLETE THIS SECTION!</font></h3>


```python
# Create a tensorflow operation to give you the standard deviation
# of your images (you may need to use more than one line of code)
std_img_op = ...

# Now calculate the standard deviation using your session
std_img = sess.run(std_img_op)
```

Don't change this section!


```python
# Then plot the resulting standard deviation image:
# Make sure the std image is the right size!
assert(std_img.shape == (100, 100) or std_img.shape == (100, 100, 3))
plt.figure(figsize=(10, 10))
plt.imshow(std_img / np.max(std_img))
plt.imsave(arr=std_img, fname='std.png')
```

<a name="part-four---sort-the-dataset"></a>
# Part Four - Sort the Dataset

<a name="instructions-3"></a>
## Instructions
Using tensorflow, apply some type of organization to the dataset. You may want to sort your images based on their average value. To do this, you could calculate either the sum value or the mean value of each image in your dataset and then use those values, e.g. stored inside a variable `values` to sort your images using something like `tf.nn.top_k` and `sorted_imgs = np.array([imgs[idx_i] for idx_i in idxs])` prior to creating the montage image, `m = montage(sorted_imgs, "sorted.png")` and then include this image in your zip file as <b>sorted.png</b>

<a name="code-3"></a>
## Code

<h3><font color='red'>TODO! COMPLETE THIS SECTION!</font></h3>


```python
# Create an operation using tensorflow which could
# provide you for instance the sum or mean value
# of every image in your dataset:
values = ...

# Then create another operation which sorts those values
# and then calculate the result:
idxs_op = ...
idxs = sess.run(idxs_op)

# Then finally use the sorted indices to sort your images:
sorted_imgs = np.array([imgs[idx_i] for idx_i in idxs])
```

Don't change this section!


```python
# Then plot the resulting sorted dataset montage:
# Make sure we have a 100 x 100 x 100 x 3 dimension array
assert(sorted_imgs.shape == (100, 100, 100, 3))
plt.figure(figsize=(10, 10))
plt.imshow(montage(sorted_imgs, 'sorted.png'))
```

<a name="assignment-submission"></a>
# Assignment Submission

After you've completed all 4 parts, create a zip file of the current directory using the code below.  This code will make sure you have included this completed ipython notebook and the following files named exactly as:

<pre>
    session-1/
      session-1.ipynb
      dataset.png
      mean.png
      std.png
      sorted.png
      libs/
        utils.py
</pre>

You'll then submit this zip file for your first assignment on Kadenze for "Assignment 1: Datasets/Computing with Tensorflow"!  If you have any questions, remember to reach out on the forums and connect with your peers or with me.

To get assessed, you'll need to be a premium student which is free for a month!  If you aren't already enrolled as a student, register now at http://www.kadenze.com/ and join the #CADL community to see what your peers are doing! https://www.kadenze.com/courses/creative-applications-of-deep-learning-with-tensorflow/info

Then remember to complete the remaining parts of Assignemnt 1 on Kadenze!:

    * Participation Assessment: Comment on another student's open-ended arrangement (Part 4).  Think about what images they've used in their dataset and how the arrangement reflects what could be represented by that data.
    * (Extra Credit): Forum Post - Find artists making use of machine learning to organize data or finding representations within large datasets.


```python
build_submission('session-1.zip',
                 ('dataset.png',
                  'mean.png',
                  'std.png',
                  'sorted.png',
                  'session-1.ipynb'))
```
