
# Session 4: Visualizing Representations
<p class='lead'>
Creative Applications of Deep Learning with Google's Tensorflow  
Parag K. Mital  
Kadenze, Inc.
</p>

<a name="learning-goals"></a>
## Learning Goals

* Learn how to inspect deep networks by visualizing their gradients
* Learn how to "deep dream" with different objective functions and regularization techniques
* Learn how to "stylize" an image using content and style losses from different images

<!-- MarkdownTOC autolink=true autoanchor=true bracket=round -->

- [Introduction](#introduction)
- [Deep Convolutional Networks](#deep-convolutional-networks)
    - [Loading a Pretrained Network](#loading-a-pretrained-network)
    - [Predicting with the Inception Network](#predicting-with-the-inception-network)
    - [Visualizing Filters](#visualizing-filters)
    - [Visualizing the Gradient](#visualizing-the-gradient)
- [Deep Dreaming](#deep-dreaming)
    - [Simplest Approach](#simplest-approach)
    - [Specifying the Objective](#specifying-the-objective)
    - [Decaying the Gradient](#decaying-the-gradient)
    - [Blurring the Gradient](#blurring-the-gradient)
    - [Clipping the Gradient](#clipping-the-gradient)
    - [Infinite Zoom / Fractal](#infinite-zoom--fractal)
    - [Laplacian Pyramid](#laplacian-pyramid)
- [Style Net](#style-net)
    - [VGG Network](#vgg-network)
    - [Dropout](#dropout)
    - [Defining the Content Features](#defining-the-content-features)
    - [Defining the Style Features](#defining-the-style-features)
    - [Remapping the Input](#remapping-the-input)
    - [Defining the Content Loss](#defining-the-content-loss)
    - [Defining the Style Loss](#defining-the-style-loss)
    - [Defining the Total Variation Loss](#defining-the-total-variation-loss)
    - [Training](#training)
- [Homework](#homework)
- [Reading](#reading)

<!-- /MarkdownTOC -->

<a name="introduction"></a>
# Introduction

So far, we've seen that a deep convolutional network can get very high accuracy in classifying the MNIST dataset, a dataset of handwritten digits numbered 0 - 9.  What happens when the number of classes grows higher than 10 possibilities?  Or the images get much larger?  We're going to explore a few new datasets and bigger and better models to try and find out.  We'll then explore a few interesting visualization tehcniques to help us understand what the networks are representing in its deeper layers and how these techniques can be used for some very interesting creative applications.

<a name="deep-convolutional-networks"></a>
# Deep Convolutional Networks

Almost 30 years of computer vision and machine learning research based on images takes an approach to processing images like what we saw at the end of Session 1: you take an image, convolve it with a set of edge detectors like the gabor filter we created, and then find some thresholding of this image to find more interesting features, such as corners, or look at histograms of the number of some orientation of edges in a particular window.  In the previous session, we started to see how Deep Learning has allowed us to move away from hand crafted features such as Gabor-like filters to letting data discover representations.  Though, how well does it scale?

A seminal shift in the perceived capabilities of deep neural networks occurred in 2012.  A network dubbed AlexNet, after its primary author, Alex Krizevsky, achieved remarkable performance on one of the most difficult computer vision datasets at the time, ImageNet.  <TODO: Insert montage of ImageNet>.  ImageNet is a dataset used in a yearly challenge called the ImageNet Large Scale Visual Recognition Challenge (ILSVRC), started in 2010.  The dataset contains nearly 1.2 million images composed of 1000 different types of objects.  Each object has anywhere between 600 - 1200 different images.  <TODO: Histogram of object labels>

Up until now, the most number of labels we've considered is 10!  The image sizes were also very small, only 28 x 28 pixels, and it didn't even have color.

Let's look at a state-of-the-art network that has already been trained on ImageNet.

<a name="loading-a-pretrained-network"></a>
## Loading a Pretrained Network

We can use an existing network that has been trained by loading the model's weights into a network definition.  The network definition is basically saying what are the set of operations in the tensorflow graph.  So how is the image manipulated, filtered, in order to get from an input image to a probability saying which 1 of 1000 possible objects is the image describing?  It also restores the model's weights.  Those are the values of every parameter in the network learned through gradient descent.  Luckily, many researchers are releasing their model definitions and weights so we don't have to train them!  We just have to load them up and then we can use the model straight away.  That's very lucky for us because these models take a lot of time, cpu, memory, and money to train.

To get the files required for these models, you'll need to download them from the resources page.

First, let's import some necessary libraries.


```python
%matplotlib inline
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import IPython.display as ipyd
from libs import gif, nb_utils
```


```python
# Bit of formatting because I don't like the default inline code style:
from IPython.core.display import HTML
HTML("""<style> .rendered_html code { 
    padding: 2px 4px;
    color: #c7254e;
    background-color: #f9f2f4;
    border-radius: 4px;
} </style>""")
```




<style> .rendered_html code { 
    padding: 2px 4px;
    color: #c7254e;
    background-color: #f9f2f4;
    border-radius: 4px;
} </style>



Start an interactive session:


```python
sess = tf.InteractiveSession()
```

Now we'll load Google's Inception model, which is a pretrained network for classification built using the ImageNet database.  I've included some helper functions for getting this model loaded and setup w/ Tensorflow.


```python
from libs import inception
net = inception.get_inception_model()
```

Here's a little extra that wasn't in the lecture.  We can visualize the graph definition using the `nb_utils` module's `show_graph` function.  This function is taken from an example in the Tensorflow repo so I can't take credit for it!  It uses Tensorboard, which we didn't get a chance to discuss, Tensorflow's web interface for visualizing graphs and training performance.  It is very useful but we sadly did not have enough time to discuss this!


```python
nb_utils.show_graph(net['graph_def'])
```



            <iframe seamless style="width:800px;height:620px;border:0" srcdoc="
            <script>
              function load() {
                document.getElementById(&quot;graph0.7352770356060288&quot;).pbtxt = 'node {\n  name: &quot;input&quot;\n  op: &quot;Placeholder&quot;\n  device: &quot;/cpu:0&quot;\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;shape&quot;\n    value {\n      shape {\n      }\n    }\n  }\n}\nnode {\n  name: &quot;conv2d0/w&quot;\n  op: &quot;Const&quot;\n  device: &quot;/cpu:0&quot;\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;value&quot;\n    value {\n      tensor {\n        dtype: DT_FLOAT\n        tensor_shape {\n          dim {\n            size: 7\n          }\n          dim {\n            size: 7\n          }\n          dim {\n            size: 3\n          }\n          dim {\n            size: 64\n          }\n        }\n        tensor_content: &quot;<stripped 37632 bytes>&quot;\n      }\n    }\n  }\n}\nnode {\n  name: &quot;conv2d0/b&quot;\n  op: &quot;Const&quot;\n  device: &quot;/cpu:0&quot;\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;value&quot;\n    value {\n      tensor {\n        dtype: DT_FLOAT\n        tensor_shape {\n          dim {\n            size: 64\n          }\n        }\n        tensor_content: &quot;<stripped 256 bytes>&quot;\n      }\n    }\n  }\n}\nnode {\n  name: &quot;conv2d1/w&quot;\n  op: &quot;Const&quot;\n  device: &quot;/cpu:0&quot;\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;value&quot;\n    value {\n      tensor {\n        dtype: DT_FLOAT\n        tensor_shape {\n          dim {\n            size: 1\n          }\n          dim {\n            size: 1\n          }\n          dim {\n            size: 64\n          }\n          dim {\n            size: 64\n          }\n        }\n        tensor_content: &quot;<stripped 16384 bytes>&quot;\n      }\n    }\n  }\n}\nnode {\n  name: &quot;conv2d1/b&quot;\n  op: &quot;Const&quot;\n  device: &quot;/cpu:0&quot;\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;value&quot;\n    value {\n      tensor {\n        dtype: DT_FLOAT\n        tensor_shape {\n          dim {\n            size: 64\n          }\n        }\n        tensor_content: &quot;<stripped 256 bytes>&quot;\n      }\n    }\n  }\n}\nnode {\n  name: &quot;conv2d2/w&quot;\n  op: &quot;Const&quot;\n  device: &quot;/cpu:0&quot;\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;value&quot;\n    value {\n      tensor {\n        dtype: DT_FLOAT\n        tensor_shape {\n          dim {\n            size: 3\n          }\n          dim {\n            size: 3\n          }\n          dim {\n            size: 64\n          }\n          dim {\n            size: 192\n          }\n        }\n        tensor_content: &quot;<stripped 442368 bytes>&quot;\n      }\n    }\n  }\n}\nnode {\n  name: &quot;conv2d2/b&quot;\n  op: &quot;Const&quot;\n  device: &quot;/cpu:0&quot;\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;value&quot;\n    value {\n      tensor {\n        dtype: DT_FLOAT\n        tensor_shape {\n          dim {\n            size: 192\n          }\n        }\n        tensor_content: &quot;<stripped 768 bytes>&quot;\n      }\n    }\n  }\n}\nnode {\n  name: &quot;mixed3a/1x1_w&quot;\n  op: &quot;Const&quot;\n  device: &quot;/cpu:0&quot;\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;value&quot;\n    value {\n      tensor {\n        dtype: DT_FLOAT\n        tensor_shape {\n          dim {\n            size: 1\n          }\n          dim {\n            size: 1\n          }\n          dim {\n            size: 192\n          }\n          dim {\n            size: 64\n          }\n        }\n        tensor_content: &quot;<stripped 49152 bytes>&quot;\n      }\n    }\n  }\n}\nnode {\n  name: &quot;mixed3a/1x1_b&quot;\n  op: &quot;Const&quot;\n  device: &quot;/cpu:0&quot;\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;value&quot;\n    value {\n      tensor {\n        dtype: DT_FLOAT\n        tensor_shape {\n          dim {\n            size: 64\n          }\n        }\n        tensor_content: &quot;<stripped 256 bytes>&quot;\n      }\n    }\n  }\n}\nnode {\n  name: &quot;mixed3a/3x3_bottleneck_w&quot;\n  op: &quot;Const&quot;\n  device: &quot;/cpu:0&quot;\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;value&quot;\n    value {\n      tensor {\n        dtype: DT_FLOAT\n        tensor_shape {\n          dim {\n            size: 1\n          }\n          dim {\n            size: 1\n          }\n          dim {\n            size: 192\n          }\n          dim {\n            size: 96\n          }\n        }\n        tensor_content: &quot;<stripped 73728 bytes>&quot;\n      }\n    }\n  }\n}\nnode {\n  name: &quot;mixed3a/3x3_bottleneck_b&quot;\n  op: &quot;Const&quot;\n  device: &quot;/cpu:0&quot;\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;value&quot;\n    value {\n      tensor {\n        dtype: DT_FLOAT\n        tensor_shape {\n          dim {\n            size: 96\n          }\n        }\n        tensor_content: &quot;<stripped 384 bytes>&quot;\n      }\n    }\n  }\n}\nnode {\n  name: &quot;mixed3a/3x3_w&quot;\n  op: &quot;Const&quot;\n  device: &quot;/cpu:0&quot;\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;value&quot;\n    value {\n      tensor {\n        dtype: DT_FLOAT\n        tensor_shape {\n          dim {\n            size: 3\n          }\n          dim {\n            size: 3\n          }\n          dim {\n            size: 96\n          }\n          dim {\n            size: 128\n          }\n        }\n        tensor_content: &quot;<stripped 442368 bytes>&quot;\n      }\n    }\n  }\n}\nnode {\n  name: &quot;mixed3a/3x3_b&quot;\n  op: &quot;Const&quot;\n  device: &quot;/cpu:0&quot;\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;value&quot;\n    value {\n      tensor {\n        dtype: DT_FLOAT\n        tensor_shape {\n          dim {\n            size: 128\n          }\n        }\n        tensor_content: &quot;<stripped 512 bytes>&quot;\n      }\n    }\n  }\n}\nnode {\n  name: &quot;mixed3a/5x5_bottleneck_w&quot;\n  op: &quot;Const&quot;\n  device: &quot;/cpu:0&quot;\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;value&quot;\n    value {\n      tensor {\n        dtype: DT_FLOAT\n        tensor_shape {\n          dim {\n            size: 1\n          }\n          dim {\n            size: 1\n          }\n          dim {\n            size: 192\n          }\n          dim {\n            size: 16\n          }\n        }\n        tensor_content: &quot;<stripped 12288 bytes>&quot;\n      }\n    }\n  }\n}\nnode {\n  name: &quot;mixed3a/5x5_bottleneck_b&quot;\n  op: &quot;Const&quot;\n  device: &quot;/cpu:0&quot;\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;value&quot;\n    value {\n      tensor {\n        dtype: DT_FLOAT\n        tensor_shape {\n          dim {\n            size: 16\n          }\n        }\n        tensor_content: &quot;<stripped 64 bytes>&quot;\n      }\n    }\n  }\n}\nnode {\n  name: &quot;mixed3a/5x5_w&quot;\n  op: &quot;Const&quot;\n  device: &quot;/cpu:0&quot;\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;value&quot;\n    value {\n      tensor {\n        dtype: DT_FLOAT\n        tensor_shape {\n          dim {\n            size: 5\n          }\n          dim {\n            size: 5\n          }\n          dim {\n            size: 16\n          }\n          dim {\n            size: 32\n          }\n        }\n        tensor_content: &quot;<stripped 51200 bytes>&quot;\n      }\n    }\n  }\n}\nnode {\n  name: &quot;mixed3a/5x5_b&quot;\n  op: &quot;Const&quot;\n  device: &quot;/cpu:0&quot;\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;value&quot;\n    value {\n      tensor {\n        dtype: DT_FLOAT\n        tensor_shape {\n          dim {\n            size: 32\n          }\n        }\n        tensor_content: &quot;<stripped 128 bytes>&quot;\n      }\n    }\n  }\n}\nnode {\n  name: &quot;mixed3a/pool_reduce_w&quot;\n  op: &quot;Const&quot;\n  device: &quot;/cpu:0&quot;\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;value&quot;\n    value {\n      tensor {\n        dtype: DT_FLOAT\n        tensor_shape {\n          dim {\n            size: 1\n          }\n          dim {\n            size: 1\n          }\n          dim {\n            size: 192\n          }\n          dim {\n            size: 32\n          }\n        }\n        tensor_content: &quot;<stripped 24576 bytes>&quot;\n      }\n    }\n  }\n}\nnode {\n  name: &quot;mixed3a/pool_reduce_b&quot;\n  op: &quot;Const&quot;\n  device: &quot;/cpu:0&quot;\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;value&quot;\n    value {\n      tensor {\n        dtype: DT_FLOAT\n        tensor_shape {\n          dim {\n            size: 32\n          }\n        }\n        tensor_content: &quot;<stripped 128 bytes>&quot;\n      }\n    }\n  }\n}\nnode {\n  name: &quot;mixed3b/1x1_w&quot;\n  op: &quot;Const&quot;\n  device: &quot;/cpu:0&quot;\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;value&quot;\n    value {\n      tensor {\n        dtype: DT_FLOAT\n        tensor_shape {\n          dim {\n            size: 1\n          }\n          dim {\n            size: 1\n          }\n          dim {\n            size: 256\n          }\n          dim {\n            size: 128\n          }\n        }\n        tensor_content: &quot;<stripped 131072 bytes>&quot;\n      }\n    }\n  }\n}\nnode {\n  name: &quot;mixed3b/1x1_b&quot;\n  op: &quot;Const&quot;\n  device: &quot;/cpu:0&quot;\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;value&quot;\n    value {\n      tensor {\n        dtype: DT_FLOAT\n        tensor_shape {\n          dim {\n            size: 128\n          }\n        }\n        tensor_content: &quot;<stripped 512 bytes>&quot;\n      }\n    }\n  }\n}\nnode {\n  name: &quot;mixed3b/3x3_bottleneck_w&quot;\n  op: &quot;Const&quot;\n  device: &quot;/cpu:0&quot;\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;value&quot;\n    value {\n      tensor {\n        dtype: DT_FLOAT\n        tensor_shape {\n          dim {\n            size: 1\n          }\n          dim {\n            size: 1\n          }\n          dim {\n            size: 256\n          }\n          dim {\n            size: 128\n          }\n        }\n        tensor_content: &quot;<stripped 131072 bytes>&quot;\n      }\n    }\n  }\n}\nnode {\n  name: &quot;mixed3b/3x3_bottleneck_b&quot;\n  op: &quot;Const&quot;\n  device: &quot;/cpu:0&quot;\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;value&quot;\n    value {\n      tensor {\n        dtype: DT_FLOAT\n        tensor_shape {\n          dim {\n            size: 128\n          }\n        }\n        tensor_content: &quot;<stripped 512 bytes>&quot;\n      }\n    }\n  }\n}\nnode {\n  name: &quot;mixed3b/3x3_w&quot;\n  op: &quot;Const&quot;\n  device: &quot;/cpu:0&quot;\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;value&quot;\n    value {\n      tensor {\n        dtype: DT_FLOAT\n        tensor_shape {\n          dim {\n            size: 3\n          }\n          dim {\n            size: 3\n          }\n          dim {\n            size: 128\n          }\n          dim {\n            size: 192\n          }\n        }\n        tensor_content: &quot;<stripped 884736 bytes>&quot;\n      }\n    }\n  }\n}\nnode {\n  name: &quot;mixed3b/3x3_b&quot;\n  op: &quot;Const&quot;\n  device: &quot;/cpu:0&quot;\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;value&quot;\n    value {\n      tensor {\n        dtype: DT_FLOAT\n        tensor_shape {\n          dim {\n            size: 192\n          }\n        }\n        tensor_content: &quot;<stripped 768 bytes>&quot;\n      }\n    }\n  }\n}\nnode {\n  name: &quot;mixed3b/5x5_bottleneck_w&quot;\n  op: &quot;Const&quot;\n  device: &quot;/cpu:0&quot;\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;value&quot;\n    value {\n      tensor {\n        dtype: DT_FLOAT\n        tensor_shape {\n          dim {\n            size: 1\n          }\n          dim {\n            size: 1\n          }\n          dim {\n            size: 256\n          }\n          dim {\n            size: 32\n          }\n        }\n        tensor_content: &quot;<stripped 32768 bytes>&quot;\n      }\n    }\n  }\n}\nnode {\n  name: &quot;mixed3b/5x5_bottleneck_b&quot;\n  op: &quot;Const&quot;\n  device: &quot;/cpu:0&quot;\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;value&quot;\n    value {\n      tensor {\n        dtype: DT_FLOAT\n        tensor_shape {\n          dim {\n            size: 32\n          }\n        }\n        tensor_content: &quot;<stripped 128 bytes>&quot;\n      }\n    }\n  }\n}\nnode {\n  name: &quot;mixed3b/5x5_w&quot;\n  op: &quot;Const&quot;\n  device: &quot;/cpu:0&quot;\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;value&quot;\n    value {\n      tensor {\n        dtype: DT_FLOAT\n        tensor_shape {\n          dim {\n            size: 5\n          }\n          dim {\n            size: 5\n          }\n          dim {\n            size: 32\n          }\n          dim {\n            size: 96\n          }\n        }\n        tensor_content: &quot;<stripped 307200 bytes>&quot;\n      }\n    }\n  }\n}\nnode {\n  name: &quot;mixed3b/5x5_b&quot;\n  op: &quot;Const&quot;\n  device: &quot;/cpu:0&quot;\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;value&quot;\n    value {\n      tensor {\n        dtype: DT_FLOAT\n        tensor_shape {\n          dim {\n            size: 96\n          }\n        }\n        tensor_content: &quot;<stripped 384 bytes>&quot;\n      }\n    }\n  }\n}\nnode {\n  name: &quot;mixed3b/pool_reduce_w&quot;\n  op: &quot;Const&quot;\n  device: &quot;/cpu:0&quot;\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;value&quot;\n    value {\n      tensor {\n        dtype: DT_FLOAT\n        tensor_shape {\n          dim {\n            size: 1\n          }\n          dim {\n            size: 1\n          }\n          dim {\n            size: 256\n          }\n          dim {\n            size: 64\n          }\n        }\n        tensor_content: &quot;<stripped 65536 bytes>&quot;\n      }\n    }\n  }\n}\nnode {\n  name: &quot;mixed3b/pool_reduce_b&quot;\n  op: &quot;Const&quot;\n  device: &quot;/cpu:0&quot;\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;value&quot;\n    value {\n      tensor {\n        dtype: DT_FLOAT\n        tensor_shape {\n          dim {\n            size: 64\n          }\n        }\n        tensor_content: &quot;<stripped 256 bytes>&quot;\n      }\n    }\n  }\n}\nnode {\n  name: &quot;mixed4a/1x1_w&quot;\n  op: &quot;Const&quot;\n  device: &quot;/cpu:0&quot;\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;value&quot;\n    value {\n      tensor {\n        dtype: DT_FLOAT\n        tensor_shape {\n          dim {\n            size: 1\n          }\n          dim {\n            size: 1\n          }\n          dim {\n            size: 480\n          }\n          dim {\n            size: 192\n          }\n        }\n        tensor_content: &quot;<stripped 368640 bytes>&quot;\n      }\n    }\n  }\n}\nnode {\n  name: &quot;mixed4a/1x1_b&quot;\n  op: &quot;Const&quot;\n  device: &quot;/cpu:0&quot;\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;value&quot;\n    value {\n      tensor {\n        dtype: DT_FLOAT\n        tensor_shape {\n          dim {\n            size: 192\n          }\n        }\n        tensor_content: &quot;<stripped 768 bytes>&quot;\n      }\n    }\n  }\n}\nnode {\n  name: &quot;mixed4a/3x3_bottleneck_w&quot;\n  op: &quot;Const&quot;\n  device: &quot;/cpu:0&quot;\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;value&quot;\n    value {\n      tensor {\n        dtype: DT_FLOAT\n        tensor_shape {\n          dim {\n            size: 1\n          }\n          dim {\n            size: 1\n          }\n          dim {\n            size: 480\n          }\n          dim {\n            size: 96\n          }\n        }\n        tensor_content: &quot;<stripped 184320 bytes>&quot;\n      }\n    }\n  }\n}\nnode {\n  name: &quot;mixed4a/3x3_bottleneck_b&quot;\n  op: &quot;Const&quot;\n  device: &quot;/cpu:0&quot;\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;value&quot;\n    value {\n      tensor {\n        dtype: DT_FLOAT\n        tensor_shape {\n          dim {\n            size: 96\n          }\n        }\n        tensor_content: &quot;<stripped 384 bytes>&quot;\n      }\n    }\n  }\n}\nnode {\n  name: &quot;mixed4a/3x3_w&quot;\n  op: &quot;Const&quot;\n  device: &quot;/cpu:0&quot;\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;value&quot;\n    value {\n      tensor {\n        dtype: DT_FLOAT\n        tensor_shape {\n          dim {\n            size: 3\n          }\n          dim {\n            size: 3\n          }\n          dim {\n            size: 96\n          }\n          dim {\n            size: 204\n          }\n        }\n        tensor_content: &quot;<stripped 705024 bytes>&quot;\n      }\n    }\n  }\n}\nnode {\n  name: &quot;mixed4a/3x3_b&quot;\n  op: &quot;Const&quot;\n  device: &quot;/cpu:0&quot;\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;value&quot;\n    value {\n      tensor {\n        dtype: DT_FLOAT\n        tensor_shape {\n          dim {\n            size: 204\n          }\n        }\n        tensor_content: &quot;<stripped 816 bytes>&quot;\n      }\n    }\n  }\n}\nnode {\n  name: &quot;mixed4a/5x5_bottleneck_w&quot;\n  op: &quot;Const&quot;\n  device: &quot;/cpu:0&quot;\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;value&quot;\n    value {\n      tensor {\n        dtype: DT_FLOAT\n        tensor_shape {\n          dim {\n            size: 1\n          }\n          dim {\n            size: 1\n          }\n          dim {\n            size: 480\n          }\n          dim {\n            size: 16\n          }\n        }\n        tensor_content: &quot;<stripped 30720 bytes>&quot;\n      }\n    }\n  }\n}\nnode {\n  name: &quot;mixed4a/5x5_bottleneck_b&quot;\n  op: &quot;Const&quot;\n  device: &quot;/cpu:0&quot;\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;value&quot;\n    value {\n      tensor {\n        dtype: DT_FLOAT\n        tensor_shape {\n          dim {\n            size: 16\n          }\n        }\n        tensor_content: &quot;<stripped 64 bytes>&quot;\n      }\n    }\n  }\n}\nnode {\n  name: &quot;mixed4a/5x5_w&quot;\n  op: &quot;Const&quot;\n  device: &quot;/cpu:0&quot;\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;value&quot;\n    value {\n      tensor {\n        dtype: DT_FLOAT\n        tensor_shape {\n          dim {\n            size: 5\n          }\n          dim {\n            size: 5\n          }\n          dim {\n            size: 16\n          }\n          dim {\n            size: 48\n          }\n        }\n        tensor_content: &quot;<stripped 76800 bytes>&quot;\n      }\n    }\n  }\n}\nnode {\n  name: &quot;mixed4a/5x5_b&quot;\n  op: &quot;Const&quot;\n  device: &quot;/cpu:0&quot;\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;value&quot;\n    value {\n      tensor {\n        dtype: DT_FLOAT\n        tensor_shape {\n          dim {\n            size: 48\n          }\n        }\n        tensor_content: &quot;<stripped 192 bytes>&quot;\n      }\n    }\n  }\n}\nnode {\n  name: &quot;mixed4a/pool_reduce_w&quot;\n  op: &quot;Const&quot;\n  device: &quot;/cpu:0&quot;\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;value&quot;\n    value {\n      tensor {\n        dtype: DT_FLOAT\n        tensor_shape {\n          dim {\n            size: 1\n          }\n          dim {\n            size: 1\n          }\n          dim {\n            size: 480\n          }\n          dim {\n            size: 64\n          }\n        }\n        tensor_content: &quot;<stripped 122880 bytes>&quot;\n      }\n    }\n  }\n}\nnode {\n  name: &quot;mixed4a/pool_reduce_b&quot;\n  op: &quot;Const&quot;\n  device: &quot;/cpu:0&quot;\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;value&quot;\n    value {\n      tensor {\n        dtype: DT_FLOAT\n        tensor_shape {\n          dim {\n            size: 64\n          }\n        }\n        tensor_content: &quot;<stripped 256 bytes>&quot;\n      }\n    }\n  }\n}\nnode {\n  name: &quot;mixed4b/1x1_w&quot;\n  op: &quot;Const&quot;\n  device: &quot;/cpu:0&quot;\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;value&quot;\n    value {\n      tensor {\n        dtype: DT_FLOAT\n        tensor_shape {\n          dim {\n            size: 1\n          }\n          dim {\n            size: 1\n          }\n          dim {\n            size: 508\n          }\n          dim {\n            size: 160\n          }\n        }\n        tensor_content: &quot;<stripped 325120 bytes>&quot;\n      }\n    }\n  }\n}\nnode {\n  name: &quot;mixed4b/1x1_b&quot;\n  op: &quot;Const&quot;\n  device: &quot;/cpu:0&quot;\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;value&quot;\n    value {\n      tensor {\n        dtype: DT_FLOAT\n        tensor_shape {\n          dim {\n            size: 160\n          }\n        }\n        tensor_content: &quot;<stripped 640 bytes>&quot;\n      }\n    }\n  }\n}\nnode {\n  name: &quot;mixed4b/3x3_bottleneck_w&quot;\n  op: &quot;Const&quot;\n  device: &quot;/cpu:0&quot;\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;value&quot;\n    value {\n      tensor {\n        dtype: DT_FLOAT\n        tensor_shape {\n          dim {\n            size: 1\n          }\n          dim {\n            size: 1\n          }\n          dim {\n            size: 508\n          }\n          dim {\n            size: 112\n          }\n        }\n        tensor_content: &quot;<stripped 227584 bytes>&quot;\n      }\n    }\n  }\n}\nnode {\n  name: &quot;mixed4b/3x3_bottleneck_b&quot;\n  op: &quot;Const&quot;\n  device: &quot;/cpu:0&quot;\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;value&quot;\n    value {\n      tensor {\n        dtype: DT_FLOAT\n        tensor_shape {\n          dim {\n            size: 112\n          }\n        }\n        tensor_content: &quot;<stripped 448 bytes>&quot;\n      }\n    }\n  }\n}\nnode {\n  name: &quot;mixed4b/3x3_w&quot;\n  op: &quot;Const&quot;\n  device: &quot;/cpu:0&quot;\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;value&quot;\n    value {\n      tensor {\n        dtype: DT_FLOAT\n        tensor_shape {\n          dim {\n            size: 3\n          }\n          dim {\n            size: 3\n          }\n          dim {\n            size: 112\n          }\n          dim {\n            size: 224\n          }\n        }\n        tensor_content: &quot;<stripped 903168 bytes>&quot;\n      }\n    }\n  }\n}\nnode {\n  name: &quot;mixed4b/3x3_b&quot;\n  op: &quot;Const&quot;\n  device: &quot;/cpu:0&quot;\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;value&quot;\n    value {\n      tensor {\n        dtype: DT_FLOAT\n        tensor_shape {\n          dim {\n            size: 224\n          }\n        }\n        tensor_content: &quot;<stripped 896 bytes>&quot;\n      }\n    }\n  }\n}\nnode {\n  name: &quot;mixed4b/5x5_bottleneck_w&quot;\n  op: &quot;Const&quot;\n  device: &quot;/cpu:0&quot;\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;value&quot;\n    value {\n      tensor {\n        dtype: DT_FLOAT\n        tensor_shape {\n          dim {\n            size: 1\n          }\n          dim {\n            size: 1\n          }\n          dim {\n            size: 508\n          }\n          dim {\n            size: 24\n          }\n        }\n        tensor_content: &quot;<stripped 48768 bytes>&quot;\n      }\n    }\n  }\n}\nnode {\n  name: &quot;mixed4b/5x5_bottleneck_b&quot;\n  op: &quot;Const&quot;\n  device: &quot;/cpu:0&quot;\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;value&quot;\n    value {\n      tensor {\n        dtype: DT_FLOAT\n        tensor_shape {\n          dim {\n            size: 24\n          }\n        }\n        tensor_content: &quot;<stripped 96 bytes>&quot;\n      }\n    }\n  }\n}\nnode {\n  name: &quot;mixed4b/5x5_w&quot;\n  op: &quot;Const&quot;\n  device: &quot;/cpu:0&quot;\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;value&quot;\n    value {\n      tensor {\n        dtype: DT_FLOAT\n        tensor_shape {\n          dim {\n            size: 5\n          }\n          dim {\n            size: 5\n          }\n          dim {\n            size: 24\n          }\n          dim {\n            size: 64\n          }\n        }\n        tensor_content: &quot;<stripped 153600 bytes>&quot;\n      }\n    }\n  }\n}\nnode {\n  name: &quot;mixed4b/5x5_b&quot;\n  op: &quot;Const&quot;\n  device: &quot;/cpu:0&quot;\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;value&quot;\n    value {\n      tensor {\n        dtype: DT_FLOAT\n        tensor_shape {\n          dim {\n            size: 64\n          }\n        }\n        tensor_content: &quot;<stripped 256 bytes>&quot;\n      }\n    }\n  }\n}\nnode {\n  name: &quot;mixed4b/pool_reduce_w&quot;\n  op: &quot;Const&quot;\n  device: &quot;/cpu:0&quot;\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;value&quot;\n    value {\n      tensor {\n        dtype: DT_FLOAT\n        tensor_shape {\n          dim {\n            size: 1\n          }\n          dim {\n            size: 1\n          }\n          dim {\n            size: 508\n          }\n          dim {\n            size: 64\n          }\n        }\n        tensor_content: &quot;<stripped 130048 bytes>&quot;\n      }\n    }\n  }\n}\nnode {\n  name: &quot;mixed4b/pool_reduce_b&quot;\n  op: &quot;Const&quot;\n  device: &quot;/cpu:0&quot;\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;value&quot;\n    value {\n      tensor {\n        dtype: DT_FLOAT\n        tensor_shape {\n          dim {\n            size: 64\n          }\n        }\n        tensor_content: &quot;<stripped 256 bytes>&quot;\n      }\n    }\n  }\n}\nnode {\n  name: &quot;mixed4c/1x1_w&quot;\n  op: &quot;Const&quot;\n  device: &quot;/cpu:0&quot;\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;value&quot;\n    value {\n      tensor {\n        dtype: DT_FLOAT\n        tensor_shape {\n          dim {\n            size: 1\n          }\n          dim {\n            size: 1\n          }\n          dim {\n            size: 512\n          }\n          dim {\n            size: 128\n          }\n        }\n        tensor_content: &quot;<stripped 262144 bytes>&quot;\n      }\n    }\n  }\n}\nnode {\n  name: &quot;mixed4c/1x1_b&quot;\n  op: &quot;Const&quot;\n  device: &quot;/cpu:0&quot;\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;value&quot;\n    value {\n      tensor {\n        dtype: DT_FLOAT\n        tensor_shape {\n          dim {\n            size: 128\n          }\n        }\n        tensor_content: &quot;<stripped 512 bytes>&quot;\n      }\n    }\n  }\n}\nnode {\n  name: &quot;mixed4c/3x3_bottleneck_w&quot;\n  op: &quot;Const&quot;\n  device: &quot;/cpu:0&quot;\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;value&quot;\n    value {\n      tensor {\n        dtype: DT_FLOAT\n        tensor_shape {\n          dim {\n            size: 1\n          }\n          dim {\n            size: 1\n          }\n          dim {\n            size: 512\n          }\n          dim {\n            size: 128\n          }\n        }\n        tensor_content: &quot;<stripped 262144 bytes>&quot;\n      }\n    }\n  }\n}\nnode {\n  name: &quot;mixed4c/3x3_bottleneck_b&quot;\n  op: &quot;Const&quot;\n  device: &quot;/cpu:0&quot;\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;value&quot;\n    value {\n      tensor {\n        dtype: DT_FLOAT\n        tensor_shape {\n          dim {\n            size: 128\n          }\n        }\n        tensor_content: &quot;<stripped 512 bytes>&quot;\n      }\n    }\n  }\n}\nnode {\n  name: &quot;mixed4c/3x3_w&quot;\n  op: &quot;Const&quot;\n  device: &quot;/cpu:0&quot;\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;value&quot;\n    value {\n      tensor {\n        dtype: DT_FLOAT\n        tensor_shape {\n          dim {\n            size: 3\n          }\n          dim {\n            size: 3\n          }\n          dim {\n            size: 128\n          }\n          dim {\n            size: 256\n          }\n        }\n        tensor_content: &quot;<stripped 1179648 bytes>&quot;\n      }\n    }\n  }\n}\nnode {\n  name: &quot;mixed4c/3x3_b&quot;\n  op: &quot;Const&quot;\n  device: &quot;/cpu:0&quot;\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;value&quot;\n    value {\n      tensor {\n        dtype: DT_FLOAT\n        tensor_shape {\n          dim {\n            size: 256\n          }\n        }\n        tensor_content: &quot;<stripped 1024 bytes>&quot;\n      }\n    }\n  }\n}\nnode {\n  name: &quot;mixed4c/5x5_bottleneck_w&quot;\n  op: &quot;Const&quot;\n  device: &quot;/cpu:0&quot;\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;value&quot;\n    value {\n      tensor {\n        dtype: DT_FLOAT\n        tensor_shape {\n          dim {\n            size: 1\n          }\n          dim {\n            size: 1\n          }\n          dim {\n            size: 512\n          }\n          dim {\n            size: 24\n          }\n        }\n        tensor_content: &quot;<stripped 49152 bytes>&quot;\n      }\n    }\n  }\n}\nnode {\n  name: &quot;mixed4c/5x5_bottleneck_b&quot;\n  op: &quot;Const&quot;\n  device: &quot;/cpu:0&quot;\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;value&quot;\n    value {\n      tensor {\n        dtype: DT_FLOAT\n        tensor_shape {\n          dim {\n            size: 24\n          }\n        }\n        tensor_content: &quot;<stripped 96 bytes>&quot;\n      }\n    }\n  }\n}\nnode {\n  name: &quot;mixed4c/5x5_w&quot;\n  op: &quot;Const&quot;\n  device: &quot;/cpu:0&quot;\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;value&quot;\n    value {\n      tensor {\n        dtype: DT_FLOAT\n        tensor_shape {\n          dim {\n            size: 5\n          }\n          dim {\n            size: 5\n          }\n          dim {\n            size: 24\n          }\n          dim {\n            size: 64\n          }\n        }\n        tensor_content: &quot;<stripped 153600 bytes>&quot;\n      }\n    }\n  }\n}\nnode {\n  name: &quot;mixed4c/5x5_b&quot;\n  op: &quot;Const&quot;\n  device: &quot;/cpu:0&quot;\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;value&quot;\n    value {\n      tensor {\n        dtype: DT_FLOAT\n        tensor_shape {\n          dim {\n            size: 64\n          }\n        }\n        tensor_content: &quot;<stripped 256 bytes>&quot;\n      }\n    }\n  }\n}\nnode {\n  name: &quot;mixed4c/pool_reduce_w&quot;\n  op: &quot;Const&quot;\n  device: &quot;/cpu:0&quot;\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;value&quot;\n    value {\n      tensor {\n        dtype: DT_FLOAT\n        tensor_shape {\n          dim {\n            size: 1\n          }\n          dim {\n            size: 1\n          }\n          dim {\n            size: 512\n          }\n          dim {\n            size: 64\n          }\n        }\n        tensor_content: &quot;<stripped 131072 bytes>&quot;\n      }\n    }\n  }\n}\nnode {\n  name: &quot;mixed4c/pool_reduce_b&quot;\n  op: &quot;Const&quot;\n  device: &quot;/cpu:0&quot;\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;value&quot;\n    value {\n      tensor {\n        dtype: DT_FLOAT\n        tensor_shape {\n          dim {\n            size: 64\n          }\n        }\n        tensor_content: &quot;<stripped 256 bytes>&quot;\n      }\n    }\n  }\n}\nnode {\n  name: &quot;mixed4d/1x1_w&quot;\n  op: &quot;Const&quot;\n  device: &quot;/cpu:0&quot;\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;value&quot;\n    value {\n      tensor {\n        dtype: DT_FLOAT\n        tensor_shape {\n          dim {\n            size: 1\n          }\n          dim {\n            size: 1\n          }\n          dim {\n            size: 512\n          }\n          dim {\n            size: 112\n          }\n        }\n        tensor_content: &quot;<stripped 229376 bytes>&quot;\n      }\n    }\n  }\n}\nnode {\n  name: &quot;mixed4d/1x1_b&quot;\n  op: &quot;Const&quot;\n  device: &quot;/cpu:0&quot;\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;value&quot;\n    value {\n      tensor {\n        dtype: DT_FLOAT\n        tensor_shape {\n          dim {\n            size: 112\n          }\n        }\n        tensor_content: &quot;<stripped 448 bytes>&quot;\n      }\n    }\n  }\n}\nnode {\n  name: &quot;mixed4d/3x3_bottleneck_w&quot;\n  op: &quot;Const&quot;\n  device: &quot;/cpu:0&quot;\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;value&quot;\n    value {\n      tensor {\n        dtype: DT_FLOAT\n        tensor_shape {\n          dim {\n            size: 1\n          }\n          dim {\n            size: 1\n          }\n          dim {\n            size: 512\n          }\n          dim {\n            size: 144\n          }\n        }\n        tensor_content: &quot;<stripped 294912 bytes>&quot;\n      }\n    }\n  }\n}\nnode {\n  name: &quot;mixed4d/3x3_bottleneck_b&quot;\n  op: &quot;Const&quot;\n  device: &quot;/cpu:0&quot;\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;value&quot;\n    value {\n      tensor {\n        dtype: DT_FLOAT\n        tensor_shape {\n          dim {\n            size: 144\n          }\n        }\n        tensor_content: &quot;<stripped 576 bytes>&quot;\n      }\n    }\n  }\n}\nnode {\n  name: &quot;mixed4d/3x3_w&quot;\n  op: &quot;Const&quot;\n  device: &quot;/cpu:0&quot;\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;value&quot;\n    value {\n      tensor {\n        dtype: DT_FLOAT\n        tensor_shape {\n          dim {\n            size: 3\n          }\n          dim {\n            size: 3\n          }\n          dim {\n            size: 144\n          }\n          dim {\n            size: 288\n          }\n        }\n        tensor_content: &quot;<stripped 1492992 bytes>&quot;\n      }\n    }\n  }\n}\nnode {\n  name: &quot;mixed4d/3x3_b&quot;\n  op: &quot;Const&quot;\n  device: &quot;/cpu:0&quot;\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;value&quot;\n    value {\n      tensor {\n        dtype: DT_FLOAT\n        tensor_shape {\n          dim {\n            size: 288\n          }\n        }\n        tensor_content: &quot;<stripped 1152 bytes>&quot;\n      }\n    }\n  }\n}\nnode {\n  name: &quot;mixed4d/5x5_bottleneck_w&quot;\n  op: &quot;Const&quot;\n  device: &quot;/cpu:0&quot;\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;value&quot;\n    value {\n      tensor {\n        dtype: DT_FLOAT\n        tensor_shape {\n          dim {\n            size: 1\n          }\n          dim {\n            size: 1\n          }\n          dim {\n            size: 512\n          }\n          dim {\n            size: 32\n          }\n        }\n        tensor_content: &quot;<stripped 65536 bytes>&quot;\n      }\n    }\n  }\n}\nnode {\n  name: &quot;mixed4d/5x5_bottleneck_b&quot;\n  op: &quot;Const&quot;\n  device: &quot;/cpu:0&quot;\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;value&quot;\n    value {\n      tensor {\n        dtype: DT_FLOAT\n        tensor_shape {\n          dim {\n            size: 32\n          }\n        }\n        tensor_content: &quot;<stripped 128 bytes>&quot;\n      }\n    }\n  }\n}\nnode {\n  name: &quot;mixed4d/5x5_w&quot;\n  op: &quot;Const&quot;\n  device: &quot;/cpu:0&quot;\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;value&quot;\n    value {\n      tensor {\n        dtype: DT_FLOAT\n        tensor_shape {\n          dim {\n            size: 5\n          }\n          dim {\n            size: 5\n          }\n          dim {\n            size: 32\n          }\n          dim {\n            size: 64\n          }\n        }\n        tensor_content: &quot;<stripped 204800 bytes>&quot;\n      }\n    }\n  }\n}\nnode {\n  name: &quot;mixed4d/5x5_b&quot;\n  op: &quot;Const&quot;\n  device: &quot;/cpu:0&quot;\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;value&quot;\n    value {\n      tensor {\n        dtype: DT_FLOAT\n        tensor_shape {\n          dim {\n            size: 64\n          }\n        }\n        tensor_content: &quot;<stripped 256 bytes>&quot;\n      }\n    }\n  }\n}\nnode {\n  name: &quot;mixed4d/pool_reduce_w&quot;\n  op: &quot;Const&quot;\n  device: &quot;/cpu:0&quot;\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;value&quot;\n    value {\n      tensor {\n        dtype: DT_FLOAT\n        tensor_shape {\n          dim {\n            size: 1\n          }\n          dim {\n            size: 1\n          }\n          dim {\n            size: 512\n          }\n          dim {\n            size: 64\n          }\n        }\n        tensor_content: &quot;<stripped 131072 bytes>&quot;\n      }\n    }\n  }\n}\nnode {\n  name: &quot;mixed4d/pool_reduce_b&quot;\n  op: &quot;Const&quot;\n  device: &quot;/cpu:0&quot;\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;value&quot;\n    value {\n      tensor {\n        dtype: DT_FLOAT\n        tensor_shape {\n          dim {\n            size: 64\n          }\n        }\n        tensor_content: &quot;<stripped 256 bytes>&quot;\n      }\n    }\n  }\n}\nnode {\n  name: &quot;mixed4e/1x1_w&quot;\n  op: &quot;Const&quot;\n  device: &quot;/cpu:0&quot;\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;value&quot;\n    value {\n      tensor {\n        dtype: DT_FLOAT\n        tensor_shape {\n          dim {\n            size: 1\n          }\n          dim {\n            size: 1\n          }\n          dim {\n            size: 528\n          }\n          dim {\n            size: 256\n          }\n        }\n        tensor_content: &quot;<stripped 540672 bytes>&quot;\n      }\n    }\n  }\n}\nnode {\n  name: &quot;mixed4e/1x1_b&quot;\n  op: &quot;Const&quot;\n  device: &quot;/cpu:0&quot;\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;value&quot;\n    value {\n      tensor {\n        dtype: DT_FLOAT\n        tensor_shape {\n          dim {\n            size: 256\n          }\n        }\n        tensor_content: &quot;<stripped 1024 bytes>&quot;\n      }\n    }\n  }\n}\nnode {\n  name: &quot;mixed4e/3x3_bottleneck_w&quot;\n  op: &quot;Const&quot;\n  device: &quot;/cpu:0&quot;\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;value&quot;\n    value {\n      tensor {\n        dtype: DT_FLOAT\n        tensor_shape {\n          dim {\n            size: 1\n          }\n          dim {\n            size: 1\n          }\n          dim {\n            size: 528\n          }\n          dim {\n            size: 160\n          }\n        }\n        tensor_content: &quot;<stripped 337920 bytes>&quot;\n      }\n    }\n  }\n}\nnode {\n  name: &quot;mixed4e/3x3_bottleneck_b&quot;\n  op: &quot;Const&quot;\n  device: &quot;/cpu:0&quot;\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;value&quot;\n    value {\n      tensor {\n        dtype: DT_FLOAT\n        tensor_shape {\n          dim {\n            size: 160\n          }\n        }\n        tensor_content: &quot;<stripped 640 bytes>&quot;\n      }\n    }\n  }\n}\nnode {\n  name: &quot;mixed4e/3x3_w&quot;\n  op: &quot;Const&quot;\n  device: &quot;/cpu:0&quot;\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;value&quot;\n    value {\n      tensor {\n        dtype: DT_FLOAT\n        tensor_shape {\n          dim {\n            size: 3\n          }\n          dim {\n            size: 3\n          }\n          dim {\n            size: 160\n          }\n          dim {\n            size: 320\n          }\n        }\n        tensor_content: &quot;<stripped 1843200 bytes>&quot;\n      }\n    }\n  }\n}\nnode {\n  name: &quot;mixed4e/3x3_b&quot;\n  op: &quot;Const&quot;\n  device: &quot;/cpu:0&quot;\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;value&quot;\n    value {\n      tensor {\n        dtype: DT_FLOAT\n        tensor_shape {\n          dim {\n            size: 320\n          }\n        }\n        tensor_content: &quot;<stripped 1280 bytes>&quot;\n      }\n    }\n  }\n}\nnode {\n  name: &quot;mixed4e/5x5_bottleneck_w&quot;\n  op: &quot;Const&quot;\n  device: &quot;/cpu:0&quot;\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;value&quot;\n    value {\n      tensor {\n        dtype: DT_FLOAT\n        tensor_shape {\n          dim {\n            size: 1\n          }\n          dim {\n            size: 1\n          }\n          dim {\n            size: 528\n          }\n          dim {\n            size: 32\n          }\n        }\n        tensor_content: &quot;<stripped 67584 bytes>&quot;\n      }\n    }\n  }\n}\nnode {\n  name: &quot;mixed4e/5x5_bottleneck_b&quot;\n  op: &quot;Const&quot;\n  device: &quot;/cpu:0&quot;\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;value&quot;\n    value {\n      tensor {\n        dtype: DT_FLOAT\n        tensor_shape {\n          dim {\n            size: 32\n          }\n        }\n        tensor_content: &quot;<stripped 128 bytes>&quot;\n      }\n    }\n  }\n}\nnode {\n  name: &quot;mixed4e/5x5_w&quot;\n  op: &quot;Const&quot;\n  device: &quot;/cpu:0&quot;\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;value&quot;\n    value {\n      tensor {\n        dtype: DT_FLOAT\n        tensor_shape {\n          dim {\n            size: 5\n          }\n          dim {\n            size: 5\n          }\n          dim {\n            size: 32\n          }\n          dim {\n            size: 128\n          }\n        }\n        tensor_content: &quot;<stripped 409600 bytes>&quot;\n      }\n    }\n  }\n}\nnode {\n  name: &quot;mixed4e/5x5_b&quot;\n  op: &quot;Const&quot;\n  device: &quot;/cpu:0&quot;\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;value&quot;\n    value {\n      tensor {\n        dtype: DT_FLOAT\n        tensor_shape {\n          dim {\n            size: 128\n          }\n        }\n        tensor_content: &quot;<stripped 512 bytes>&quot;\n      }\n    }\n  }\n}\nnode {\n  name: &quot;mixed4e/pool_reduce_w&quot;\n  op: &quot;Const&quot;\n  device: &quot;/cpu:0&quot;\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;value&quot;\n    value {\n      tensor {\n        dtype: DT_FLOAT\n        tensor_shape {\n          dim {\n            size: 1\n          }\n          dim {\n            size: 1\n          }\n          dim {\n            size: 528\n          }\n          dim {\n            size: 128\n          }\n        }\n        tensor_content: &quot;<stripped 270336 bytes>&quot;\n      }\n    }\n  }\n}\nnode {\n  name: &quot;mixed4e/pool_reduce_b&quot;\n  op: &quot;Const&quot;\n  device: &quot;/cpu:0&quot;\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;value&quot;\n    value {\n      tensor {\n        dtype: DT_FLOAT\n        tensor_shape {\n          dim {\n            size: 128\n          }\n        }\n        tensor_content: &quot;<stripped 512 bytes>&quot;\n      }\n    }\n  }\n}\nnode {\n  name: &quot;mixed5a/1x1_w&quot;\n  op: &quot;Const&quot;\n  device: &quot;/cpu:0&quot;\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;value&quot;\n    value {\n      tensor {\n        dtype: DT_FLOAT\n        tensor_shape {\n          dim {\n            size: 1\n          }\n          dim {\n            size: 1\n          }\n          dim {\n            size: 832\n          }\n          dim {\n            size: 256\n          }\n        }\n        tensor_content: &quot;<stripped 851968 bytes>&quot;\n      }\n    }\n  }\n}\nnode {\n  name: &quot;mixed5a/1x1_b&quot;\n  op: &quot;Const&quot;\n  device: &quot;/cpu:0&quot;\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;value&quot;\n    value {\n      tensor {\n        dtype: DT_FLOAT\n        tensor_shape {\n          dim {\n            size: 256\n          }\n        }\n        tensor_content: &quot;<stripped 1024 bytes>&quot;\n      }\n    }\n  }\n}\nnode {\n  name: &quot;mixed5a/3x3_bottleneck_w&quot;\n  op: &quot;Const&quot;\n  device: &quot;/cpu:0&quot;\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;value&quot;\n    value {\n      tensor {\n        dtype: DT_FLOAT\n        tensor_shape {\n          dim {\n            size: 1\n          }\n          dim {\n            size: 1\n          }\n          dim {\n            size: 832\n          }\n          dim {\n            size: 160\n          }\n        }\n        tensor_content: &quot;<stripped 532480 bytes>&quot;\n      }\n    }\n  }\n}\nnode {\n  name: &quot;mixed5a/3x3_bottleneck_b&quot;\n  op: &quot;Const&quot;\n  device: &quot;/cpu:0&quot;\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;value&quot;\n    value {\n      tensor {\n        dtype: DT_FLOAT\n        tensor_shape {\n          dim {\n            size: 160\n          }\n        }\n        tensor_content: &quot;<stripped 640 bytes>&quot;\n      }\n    }\n  }\n}\nnode {\n  name: &quot;mixed5a/3x3_w&quot;\n  op: &quot;Const&quot;\n  device: &quot;/cpu:0&quot;\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;value&quot;\n    value {\n      tensor {\n        dtype: DT_FLOAT\n        tensor_shape {\n          dim {\n            size: 3\n          }\n          dim {\n            size: 3\n          }\n          dim {\n            size: 160\n          }\n          dim {\n            size: 320\n          }\n        }\n        tensor_content: &quot;<stripped 1843200 bytes>&quot;\n      }\n    }\n  }\n}\nnode {\n  name: &quot;mixed5a/3x3_b&quot;\n  op: &quot;Const&quot;\n  device: &quot;/cpu:0&quot;\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;value&quot;\n    value {\n      tensor {\n        dtype: DT_FLOAT\n        tensor_shape {\n          dim {\n            size: 320\n          }\n        }\n        tensor_content: &quot;<stripped 1280 bytes>&quot;\n      }\n    }\n  }\n}\nnode {\n  name: &quot;mixed5a/5x5_bottleneck_w&quot;\n  op: &quot;Const&quot;\n  device: &quot;/cpu:0&quot;\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;value&quot;\n    value {\n      tensor {\n        dtype: DT_FLOAT\n        tensor_shape {\n          dim {\n            size: 1\n          }\n          dim {\n            size: 1\n          }\n          dim {\n            size: 832\n          }\n          dim {\n            size: 48\n          }\n        }\n        tensor_content: &quot;<stripped 159744 bytes>&quot;\n      }\n    }\n  }\n}\nnode {\n  name: &quot;mixed5a/5x5_bottleneck_b&quot;\n  op: &quot;Const&quot;\n  device: &quot;/cpu:0&quot;\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;value&quot;\n    value {\n      tensor {\n        dtype: DT_FLOAT\n        tensor_shape {\n          dim {\n            size: 48\n          }\n        }\n        tensor_content: &quot;<stripped 192 bytes>&quot;\n      }\n    }\n  }\n}\nnode {\n  name: &quot;mixed5a/5x5_w&quot;\n  op: &quot;Const&quot;\n  device: &quot;/cpu:0&quot;\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;value&quot;\n    value {\n      tensor {\n        dtype: DT_FLOAT\n        tensor_shape {\n          dim {\n            size: 5\n          }\n          dim {\n            size: 5\n          }\n          dim {\n            size: 48\n          }\n          dim {\n            size: 128\n          }\n        }\n        tensor_content: &quot;<stripped 614400 bytes>&quot;\n      }\n    }\n  }\n}\nnode {\n  name: &quot;mixed5a/5x5_b&quot;\n  op: &quot;Const&quot;\n  device: &quot;/cpu:0&quot;\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;value&quot;\n    value {\n      tensor {\n        dtype: DT_FLOAT\n        tensor_shape {\n          dim {\n            size: 128\n          }\n        }\n        tensor_content: &quot;<stripped 512 bytes>&quot;\n      }\n    }\n  }\n}\nnode {\n  name: &quot;mixed5a/pool_reduce_w&quot;\n  op: &quot;Const&quot;\n  device: &quot;/cpu:0&quot;\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;value&quot;\n    value {\n      tensor {\n        dtype: DT_FLOAT\n        tensor_shape {\n          dim {\n            size: 1\n          }\n          dim {\n            size: 1\n          }\n          dim {\n            size: 832\n          }\n          dim {\n            size: 128\n          }\n        }\n        tensor_content: &quot;<stripped 425984 bytes>&quot;\n      }\n    }\n  }\n}\nnode {\n  name: &quot;mixed5a/pool_reduce_b&quot;\n  op: &quot;Const&quot;\n  device: &quot;/cpu:0&quot;\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;value&quot;\n    value {\n      tensor {\n        dtype: DT_FLOAT\n        tensor_shape {\n          dim {\n            size: 128\n          }\n        }\n        tensor_content: &quot;<stripped 512 bytes>&quot;\n      }\n    }\n  }\n}\nnode {\n  name: &quot;mixed5b/1x1_w&quot;\n  op: &quot;Const&quot;\n  device: &quot;/cpu:0&quot;\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;value&quot;\n    value {\n      tensor {\n        dtype: DT_FLOAT\n        tensor_shape {\n          dim {\n            size: 1\n          }\n          dim {\n            size: 1\n          }\n          dim {\n            size: 832\n          }\n          dim {\n            size: 384\n          }\n        }\n        tensor_content: &quot;<stripped 1277952 bytes>&quot;\n      }\n    }\n  }\n}\nnode {\n  name: &quot;mixed5b/1x1_b&quot;\n  op: &quot;Const&quot;\n  device: &quot;/cpu:0&quot;\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;value&quot;\n    value {\n      tensor {\n        dtype: DT_FLOAT\n        tensor_shape {\n          dim {\n            size: 384\n          }\n        }\n        tensor_content: &quot;<stripped 1536 bytes>&quot;\n      }\n    }\n  }\n}\nnode {\n  name: &quot;mixed5b/3x3_bottleneck_w&quot;\n  op: &quot;Const&quot;\n  device: &quot;/cpu:0&quot;\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;value&quot;\n    value {\n      tensor {\n        dtype: DT_FLOAT\n        tensor_shape {\n          dim {\n            size: 1\n          }\n          dim {\n            size: 1\n          }\n          dim {\n            size: 832\n          }\n          dim {\n            size: 192\n          }\n        }\n        tensor_content: &quot;<stripped 638976 bytes>&quot;\n      }\n    }\n  }\n}\nnode {\n  name: &quot;mixed5b/3x3_bottleneck_b&quot;\n  op: &quot;Const&quot;\n  device: &quot;/cpu:0&quot;\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;value&quot;\n    value {\n      tensor {\n        dtype: DT_FLOAT\n        tensor_shape {\n          dim {\n            size: 192\n          }\n        }\n        tensor_content: &quot;<stripped 768 bytes>&quot;\n      }\n    }\n  }\n}\nnode {\n  name: &quot;mixed5b/3x3_w&quot;\n  op: &quot;Const&quot;\n  device: &quot;/cpu:0&quot;\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;value&quot;\n    value {\n      tensor {\n        dtype: DT_FLOAT\n        tensor_shape {\n          dim {\n            size: 3\n          }\n          dim {\n            size: 3\n          }\n          dim {\n            size: 192\n          }\n          dim {\n            size: 384\n          }\n        }\n        tensor_content: &quot;<stripped 2654208 bytes>&quot;\n      }\n    }\n  }\n}\nnode {\n  name: &quot;mixed5b/3x3_b&quot;\n  op: &quot;Const&quot;\n  device: &quot;/cpu:0&quot;\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;value&quot;\n    value {\n      tensor {\n        dtype: DT_FLOAT\n        tensor_shape {\n          dim {\n            size: 384\n          }\n        }\n        tensor_content: &quot;<stripped 1536 bytes>&quot;\n      }\n    }\n  }\n}\nnode {\n  name: &quot;mixed5b/5x5_bottleneck_w&quot;\n  op: &quot;Const&quot;\n  device: &quot;/cpu:0&quot;\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;value&quot;\n    value {\n      tensor {\n        dtype: DT_FLOAT\n        tensor_shape {\n          dim {\n            size: 1\n          }\n          dim {\n            size: 1\n          }\n          dim {\n            size: 832\n          }\n          dim {\n            size: 48\n          }\n        }\n        tensor_content: &quot;<stripped 159744 bytes>&quot;\n      }\n    }\n  }\n}\nnode {\n  name: &quot;mixed5b/5x5_bottleneck_b&quot;\n  op: &quot;Const&quot;\n  device: &quot;/cpu:0&quot;\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;value&quot;\n    value {\n      tensor {\n        dtype: DT_FLOAT\n        tensor_shape {\n          dim {\n            size: 48\n          }\n        }\n        tensor_content: &quot;<stripped 192 bytes>&quot;\n      }\n    }\n  }\n}\nnode {\n  name: &quot;mixed5b/5x5_w&quot;\n  op: &quot;Const&quot;\n  device: &quot;/cpu:0&quot;\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;value&quot;\n    value {\n      tensor {\n        dtype: DT_FLOAT\n        tensor_shape {\n          dim {\n            size: 5\n          }\n          dim {\n            size: 5\n          }\n          dim {\n            size: 48\n          }\n          dim {\n            size: 128\n          }\n        }\n        tensor_content: &quot;<stripped 614400 bytes>&quot;\n      }\n    }\n  }\n}\nnode {\n  name: &quot;mixed5b/5x5_b&quot;\n  op: &quot;Const&quot;\n  device: &quot;/cpu:0&quot;\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;value&quot;\n    value {\n      tensor {\n        dtype: DT_FLOAT\n        tensor_shape {\n          dim {\n            size: 128\n          }\n        }\n        tensor_content: &quot;<stripped 512 bytes>&quot;\n      }\n    }\n  }\n}\nnode {\n  name: &quot;mixed5b/pool_reduce_w&quot;\n  op: &quot;Const&quot;\n  device: &quot;/cpu:0&quot;\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;value&quot;\n    value {\n      tensor {\n        dtype: DT_FLOAT\n        tensor_shape {\n          dim {\n            size: 1\n          }\n          dim {\n            size: 1\n          }\n          dim {\n            size: 832\n          }\n          dim {\n            size: 128\n          }\n        }\n        tensor_content: &quot;<stripped 425984 bytes>&quot;\n      }\n    }\n  }\n}\nnode {\n  name: &quot;mixed5b/pool_reduce_b&quot;\n  op: &quot;Const&quot;\n  device: &quot;/cpu:0&quot;\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;value&quot;\n    value {\n      tensor {\n        dtype: DT_FLOAT\n        tensor_shape {\n          dim {\n            size: 128\n          }\n        }\n        tensor_content: &quot;<stripped 512 bytes>&quot;\n      }\n    }\n  }\n}\nnode {\n  name: &quot;head0/bottleneck_w&quot;\n  op: &quot;Const&quot;\n  device: &quot;/cpu:0&quot;\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;value&quot;\n    value {\n      tensor {\n        dtype: DT_FLOAT\n        tensor_shape {\n          dim {\n            size: 1\n          }\n          dim {\n            size: 1\n          }\n          dim {\n            size: 508\n          }\n          dim {\n            size: 128\n          }\n        }\n        tensor_content: &quot;<stripped 260096 bytes>&quot;\n      }\n    }\n  }\n}\nnode {\n  name: &quot;head0/bottleneck_b&quot;\n  op: &quot;Const&quot;\n  device: &quot;/cpu:0&quot;\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;value&quot;\n    value {\n      tensor {\n        dtype: DT_FLOAT\n        tensor_shape {\n          dim {\n            size: 128\n          }\n        }\n        tensor_content: &quot;<stripped 512 bytes>&quot;\n      }\n    }\n  }\n}\nnode {\n  name: &quot;nn0/w&quot;\n  op: &quot;Const&quot;\n  device: &quot;/cpu:0&quot;\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;value&quot;\n    value {\n      tensor {\n        dtype: DT_FLOAT\n        tensor_shape {\n          dim {\n            size: 2048\n          }\n          dim {\n            size: 1024\n          }\n        }\n        tensor_content: &quot;<stripped 8388608 bytes>&quot;\n      }\n    }\n  }\n}\nnode {\n  name: &quot;nn0/b&quot;\n  op: &quot;Const&quot;\n  device: &quot;/cpu:0&quot;\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;value&quot;\n    value {\n      tensor {\n        dtype: DT_FLOAT\n        tensor_shape {\n          dim {\n            size: 1024\n          }\n        }\n        tensor_content: &quot;<stripped 4096 bytes>&quot;\n      }\n    }\n  }\n}\nnode {\n  name: &quot;softmax0/w&quot;\n  op: &quot;Const&quot;\n  device: &quot;/cpu:0&quot;\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;value&quot;\n    value {\n      tensor {\n        dtype: DT_FLOAT\n        tensor_shape {\n          dim {\n            size: 1024\n          }\n          dim {\n            size: 1008\n          }\n        }\n        tensor_content: &quot;<stripped 4128768 bytes>&quot;\n      }\n    }\n  }\n}\nnode {\n  name: &quot;softmax0/b&quot;\n  op: &quot;Const&quot;\n  device: &quot;/cpu:0&quot;\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;value&quot;\n    value {\n      tensor {\n        dtype: DT_FLOAT\n        tensor_shape {\n          dim {\n            size: 1008\n          }\n        }\n        tensor_content: &quot;<stripped 4032 bytes>&quot;\n      }\n    }\n  }\n}\nnode {\n  name: &quot;head1/bottleneck_w&quot;\n  op: &quot;Const&quot;\n  device: &quot;/cpu:0&quot;\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;value&quot;\n    value {\n      tensor {\n        dtype: DT_FLOAT\n        tensor_shape {\n          dim {\n            size: 1\n          }\n          dim {\n            size: 1\n          }\n          dim {\n            size: 528\n          }\n          dim {\n            size: 128\n          }\n        }\n        tensor_content: &quot;<stripped 270336 bytes>&quot;\n      }\n    }\n  }\n}\nnode {\n  name: &quot;head1/bottleneck_b&quot;\n  op: &quot;Const&quot;\n  device: &quot;/cpu:0&quot;\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;value&quot;\n    value {\n      tensor {\n        dtype: DT_FLOAT\n        tensor_shape {\n          dim {\n            size: 128\n          }\n        }\n        tensor_content: &quot;<stripped 512 bytes>&quot;\n      }\n    }\n  }\n}\nnode {\n  name: &quot;nn1/w&quot;\n  op: &quot;Const&quot;\n  device: &quot;/cpu:0&quot;\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;value&quot;\n    value {\n      tensor {\n        dtype: DT_FLOAT\n        tensor_shape {\n          dim {\n            size: 2048\n          }\n          dim {\n            size: 1024\n          }\n        }\n        tensor_content: &quot;<stripped 8388608 bytes>&quot;\n      }\n    }\n  }\n}\nnode {\n  name: &quot;nn1/b&quot;\n  op: &quot;Const&quot;\n  device: &quot;/cpu:0&quot;\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;value&quot;\n    value {\n      tensor {\n        dtype: DT_FLOAT\n        tensor_shape {\n          dim {\n            size: 1024\n          }\n        }\n        tensor_content: &quot;<stripped 4096 bytes>&quot;\n      }\n    }\n  }\n}\nnode {\n  name: &quot;softmax1/w&quot;\n  op: &quot;Const&quot;\n  device: &quot;/cpu:0&quot;\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;value&quot;\n    value {\n      tensor {\n        dtype: DT_FLOAT\n        tensor_shape {\n          dim {\n            size: 1024\n          }\n          dim {\n            size: 1008\n          }\n        }\n        tensor_content: &quot;<stripped 4128768 bytes>&quot;\n      }\n    }\n  }\n}\nnode {\n  name: &quot;softmax1/b&quot;\n  op: &quot;Const&quot;\n  device: &quot;/cpu:0&quot;\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;value&quot;\n    value {\n      tensor {\n        dtype: DT_FLOAT\n        tensor_shape {\n          dim {\n            size: 1008\n          }\n        }\n        tensor_content: &quot;<stripped 4032 bytes>&quot;\n      }\n    }\n  }\n}\nnode {\n  name: &quot;softmax2/w&quot;\n  op: &quot;Const&quot;\n  device: &quot;/cpu:0&quot;\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;value&quot;\n    value {\n      tensor {\n        dtype: DT_FLOAT\n        tensor_shape {\n          dim {\n            size: 1024\n          }\n          dim {\n            size: 1008\n          }\n        }\n        tensor_content: &quot;<stripped 4128768 bytes>&quot;\n      }\n    }\n  }\n}\nnode {\n  name: &quot;softmax2/b&quot;\n  op: &quot;Const&quot;\n  device: &quot;/cpu:0&quot;\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;value&quot;\n    value {\n      tensor {\n        dtype: DT_FLOAT\n        tensor_shape {\n          dim {\n            size: 1008\n          }\n        }\n        tensor_content: &quot;<stripped 4032 bytes>&quot;\n      }\n    }\n  }\n}\nnode {\n  name: &quot;conv2d0/pre_relu/conv&quot;\n  op: &quot;Conv2D&quot;\n  input: &quot;input&quot;\n  input: &quot;conv2d0/w&quot;\n  device: &quot;/cpu:0&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;padding&quot;\n    value {\n      s: &quot;SAME&quot;\n    }\n  }\n  attr {\n    key: &quot;strides&quot;\n    value {\n      list {\n        i: 1\n        i: 2\n        i: 2\n        i: 1\n      }\n    }\n  }\n  attr {\n    key: &quot;use_cudnn_on_gpu&quot;\n    value {\n      b: true\n    }\n  }\n}\nnode {\n  name: &quot;conv2d0/pre_relu&quot;\n  op: &quot;BiasAdd&quot;\n  input: &quot;conv2d0/pre_relu/conv&quot;\n  input: &quot;conv2d0/b&quot;\n  device: &quot;/cpu:0&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n}\nnode {\n  name: &quot;conv2d0&quot;\n  op: &quot;Relu&quot;\n  input: &quot;conv2d0/pre_relu&quot;\n  device: &quot;/cpu:0&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n}\nnode {\n  name: &quot;maxpool0&quot;\n  op: &quot;MaxPool&quot;\n  input: &quot;conv2d0&quot;\n  device: &quot;/cpu:0&quot;\n  attr {\n    key: &quot;ksize&quot;\n    value {\n      list {\n        i: 1\n        i: 3\n        i: 3\n        i: 1\n      }\n    }\n  }\n  attr {\n    key: &quot;padding&quot;\n    value {\n      s: &quot;SAME&quot;\n    }\n  }\n  attr {\n    key: &quot;strides&quot;\n    value {\n      list {\n        i: 1\n        i: 2\n        i: 2\n        i: 1\n      }\n    }\n  }\n}\nnode {\n  name: &quot;localresponsenorm0&quot;\n  op: &quot;LRN&quot;\n  input: &quot;maxpool0&quot;\n  device: &quot;/cpu:0&quot;\n  attr {\n    key: &quot;alpha&quot;\n    value {\n      f: 9.999999747378752e-05\n    }\n  }\n  attr {\n    key: &quot;beta&quot;\n    value {\n      f: 0.5\n    }\n  }\n  attr {\n    key: &quot;bias&quot;\n    value {\n      f: 2.0\n    }\n  }\n  attr {\n    key: &quot;depth_radius&quot;\n    value {\n      i: 5\n    }\n  }\n}\nnode {\n  name: &quot;conv2d1/pre_relu/conv&quot;\n  op: &quot;Conv2D&quot;\n  input: &quot;localresponsenorm0&quot;\n  input: &quot;conv2d1/w&quot;\n  device: &quot;/cpu:0&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;padding&quot;\n    value {\n      s: &quot;VALID&quot;\n    }\n  }\n  attr {\n    key: &quot;strides&quot;\n    value {\n      list {\n        i: 1\n        i: 1\n        i: 1\n        i: 1\n      }\n    }\n  }\n  attr {\n    key: &quot;use_cudnn_on_gpu&quot;\n    value {\n      b: true\n    }\n  }\n}\nnode {\n  name: &quot;conv2d1/pre_relu&quot;\n  op: &quot;BiasAdd&quot;\n  input: &quot;conv2d1/pre_relu/conv&quot;\n  input: &quot;conv2d1/b&quot;\n  device: &quot;/cpu:0&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n}\nnode {\n  name: &quot;conv2d1&quot;\n  op: &quot;Relu&quot;\n  input: &quot;conv2d1/pre_relu&quot;\n  device: &quot;/cpu:0&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n}\nnode {\n  name: &quot;conv2d2/pre_relu/conv&quot;\n  op: &quot;Conv2D&quot;\n  input: &quot;conv2d1&quot;\n  input: &quot;conv2d2/w&quot;\n  device: &quot;/cpu:0&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;padding&quot;\n    value {\n      s: &quot;SAME&quot;\n    }\n  }\n  attr {\n    key: &quot;strides&quot;\n    value {\n      list {\n        i: 1\n        i: 1\n        i: 1\n        i: 1\n      }\n    }\n  }\n  attr {\n    key: &quot;use_cudnn_on_gpu&quot;\n    value {\n      b: true\n    }\n  }\n}\nnode {\n  name: &quot;conv2d2/pre_relu&quot;\n  op: &quot;BiasAdd&quot;\n  input: &quot;conv2d2/pre_relu/conv&quot;\n  input: &quot;conv2d2/b&quot;\n  device: &quot;/cpu:0&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n}\nnode {\n  name: &quot;conv2d2&quot;\n  op: &quot;Relu&quot;\n  input: &quot;conv2d2/pre_relu&quot;\n  device: &quot;/cpu:0&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n}\nnode {\n  name: &quot;localresponsenorm1&quot;\n  op: &quot;LRN&quot;\n  input: &quot;conv2d2&quot;\n  device: &quot;/cpu:0&quot;\n  attr {\n    key: &quot;alpha&quot;\n    value {\n      f: 9.999999747378752e-05\n    }\n  }\n  attr {\n    key: &quot;beta&quot;\n    value {\n      f: 0.5\n    }\n  }\n  attr {\n    key: &quot;bias&quot;\n    value {\n      f: 2.0\n    }\n  }\n  attr {\n    key: &quot;depth_radius&quot;\n    value {\n      i: 5\n    }\n  }\n}\nnode {\n  name: &quot;maxpool1&quot;\n  op: &quot;MaxPool&quot;\n  input: &quot;localresponsenorm1&quot;\n  device: &quot;/cpu:0&quot;\n  attr {\n    key: &quot;ksize&quot;\n    value {\n      list {\n        i: 1\n        i: 3\n        i: 3\n        i: 1\n      }\n    }\n  }\n  attr {\n    key: &quot;padding&quot;\n    value {\n      s: &quot;SAME&quot;\n    }\n  }\n  attr {\n    key: &quot;strides&quot;\n    value {\n      list {\n        i: 1\n        i: 2\n        i: 2\n        i: 1\n      }\n    }\n  }\n}\nnode {\n  name: &quot;mixed3a/1x1_pre_relu/conv&quot;\n  op: &quot;Conv2D&quot;\n  input: &quot;maxpool1&quot;\n  input: &quot;mixed3a/1x1_w&quot;\n  device: &quot;/cpu:0&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;padding&quot;\n    value {\n      s: &quot;SAME&quot;\n    }\n  }\n  attr {\n    key: &quot;strides&quot;\n    value {\n      list {\n        i: 1\n        i: 1\n        i: 1\n        i: 1\n      }\n    }\n  }\n  attr {\n    key: &quot;use_cudnn_on_gpu&quot;\n    value {\n      b: true\n    }\n  }\n}\nnode {\n  name: &quot;mixed3a/1x1_pre_relu&quot;\n  op: &quot;BiasAdd&quot;\n  input: &quot;mixed3a/1x1_pre_relu/conv&quot;\n  input: &quot;mixed3a/1x1_b&quot;\n  device: &quot;/cpu:0&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n}\nnode {\n  name: &quot;mixed3a/1x1&quot;\n  op: &quot;Relu&quot;\n  input: &quot;mixed3a/1x1_pre_relu&quot;\n  device: &quot;/cpu:0&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n}\nnode {\n  name: &quot;mixed3a/3x3_bottleneck_pre_relu/conv&quot;\n  op: &quot;Conv2D&quot;\n  input: &quot;maxpool1&quot;\n  input: &quot;mixed3a/3x3_bottleneck_w&quot;\n  device: &quot;/cpu:0&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;padding&quot;\n    value {\n      s: &quot;SAME&quot;\n    }\n  }\n  attr {\n    key: &quot;strides&quot;\n    value {\n      list {\n        i: 1\n        i: 1\n        i: 1\n        i: 1\n      }\n    }\n  }\n  attr {\n    key: &quot;use_cudnn_on_gpu&quot;\n    value {\n      b: true\n    }\n  }\n}\nnode {\n  name: &quot;mixed3a/3x3_bottleneck_pre_relu&quot;\n  op: &quot;BiasAdd&quot;\n  input: &quot;mixed3a/3x3_bottleneck_pre_relu/conv&quot;\n  input: &quot;mixed3a/3x3_bottleneck_b&quot;\n  device: &quot;/cpu:0&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n}\nnode {\n  name: &quot;mixed3a/3x3_bottleneck&quot;\n  op: &quot;Relu&quot;\n  input: &quot;mixed3a/3x3_bottleneck_pre_relu&quot;\n  device: &quot;/cpu:0&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n}\nnode {\n  name: &quot;mixed3a/3x3_pre_relu/conv&quot;\n  op: &quot;Conv2D&quot;\n  input: &quot;mixed3a/3x3_bottleneck&quot;\n  input: &quot;mixed3a/3x3_w&quot;\n  device: &quot;/cpu:0&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;padding&quot;\n    value {\n      s: &quot;SAME&quot;\n    }\n  }\n  attr {\n    key: &quot;strides&quot;\n    value {\n      list {\n        i: 1\n        i: 1\n        i: 1\n        i: 1\n      }\n    }\n  }\n  attr {\n    key: &quot;use_cudnn_on_gpu&quot;\n    value {\n      b: true\n    }\n  }\n}\nnode {\n  name: &quot;mixed3a/3x3_pre_relu&quot;\n  op: &quot;BiasAdd&quot;\n  input: &quot;mixed3a/3x3_pre_relu/conv&quot;\n  input: &quot;mixed3a/3x3_b&quot;\n  device: &quot;/cpu:0&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n}\nnode {\n  name: &quot;mixed3a/3x3&quot;\n  op: &quot;Relu&quot;\n  input: &quot;mixed3a/3x3_pre_relu&quot;\n  device: &quot;/cpu:0&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n}\nnode {\n  name: &quot;mixed3a/5x5_bottleneck_pre_relu/conv&quot;\n  op: &quot;Conv2D&quot;\n  input: &quot;maxpool1&quot;\n  input: &quot;mixed3a/5x5_bottleneck_w&quot;\n  device: &quot;/cpu:0&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;padding&quot;\n    value {\n      s: &quot;SAME&quot;\n    }\n  }\n  attr {\n    key: &quot;strides&quot;\n    value {\n      list {\n        i: 1\n        i: 1\n        i: 1\n        i: 1\n      }\n    }\n  }\n  attr {\n    key: &quot;use_cudnn_on_gpu&quot;\n    value {\n      b: true\n    }\n  }\n}\nnode {\n  name: &quot;mixed3a/5x5_bottleneck_pre_relu&quot;\n  op: &quot;BiasAdd&quot;\n  input: &quot;mixed3a/5x5_bottleneck_pre_relu/conv&quot;\n  input: &quot;mixed3a/5x5_bottleneck_b&quot;\n  device: &quot;/cpu:0&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n}\nnode {\n  name: &quot;mixed3a/5x5_bottleneck&quot;\n  op: &quot;Relu&quot;\n  input: &quot;mixed3a/5x5_bottleneck_pre_relu&quot;\n  device: &quot;/cpu:0&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n}\nnode {\n  name: &quot;mixed3a/5x5_pre_relu/conv&quot;\n  op: &quot;Conv2D&quot;\n  input: &quot;mixed3a/5x5_bottleneck&quot;\n  input: &quot;mixed3a/5x5_w&quot;\n  device: &quot;/cpu:0&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;padding&quot;\n    value {\n      s: &quot;SAME&quot;\n    }\n  }\n  attr {\n    key: &quot;strides&quot;\n    value {\n      list {\n        i: 1\n        i: 1\n        i: 1\n        i: 1\n      }\n    }\n  }\n  attr {\n    key: &quot;use_cudnn_on_gpu&quot;\n    value {\n      b: true\n    }\n  }\n}\nnode {\n  name: &quot;mixed3a/5x5_pre_relu&quot;\n  op: &quot;BiasAdd&quot;\n  input: &quot;mixed3a/5x5_pre_relu/conv&quot;\n  input: &quot;mixed3a/5x5_b&quot;\n  device: &quot;/cpu:0&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n}\nnode {\n  name: &quot;mixed3a/5x5&quot;\n  op: &quot;Relu&quot;\n  input: &quot;mixed3a/5x5_pre_relu&quot;\n  device: &quot;/cpu:0&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n}\nnode {\n  name: &quot;mixed3a/pool&quot;\n  op: &quot;MaxPool&quot;\n  input: &quot;maxpool1&quot;\n  device: &quot;/cpu:0&quot;\n  attr {\n    key: &quot;ksize&quot;\n    value {\n      list {\n        i: 1\n        i: 3\n        i: 3\n        i: 1\n      }\n    }\n  }\n  attr {\n    key: &quot;padding&quot;\n    value {\n      s: &quot;SAME&quot;\n    }\n  }\n  attr {\n    key: &quot;strides&quot;\n    value {\n      list {\n        i: 1\n        i: 1\n        i: 1\n        i: 1\n      }\n    }\n  }\n}\nnode {\n  name: &quot;mixed3a/pool_reduce_pre_relu/conv&quot;\n  op: &quot;Conv2D&quot;\n  input: &quot;mixed3a/pool&quot;\n  input: &quot;mixed3a/pool_reduce_w&quot;\n  device: &quot;/cpu:0&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;padding&quot;\n    value {\n      s: &quot;SAME&quot;\n    }\n  }\n  attr {\n    key: &quot;strides&quot;\n    value {\n      list {\n        i: 1\n        i: 1\n        i: 1\n        i: 1\n      }\n    }\n  }\n  attr {\n    key: &quot;use_cudnn_on_gpu&quot;\n    value {\n      b: true\n    }\n  }\n}\nnode {\n  name: &quot;mixed3a/pool_reduce_pre_relu&quot;\n  op: &quot;BiasAdd&quot;\n  input: &quot;mixed3a/pool_reduce_pre_relu/conv&quot;\n  input: &quot;mixed3a/pool_reduce_b&quot;\n  device: &quot;/cpu:0&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n}\nnode {\n  name: &quot;mixed3a/pool_reduce&quot;\n  op: &quot;Relu&quot;\n  input: &quot;mixed3a/pool_reduce_pre_relu&quot;\n  device: &quot;/cpu:0&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n}\nnode {\n  name: &quot;mixed3a/concat/dim&quot;\n  op: &quot;Const&quot;\n  device: &quot;/cpu:0&quot;\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_INT32\n    }\n  }\n  attr {\n    key: &quot;value&quot;\n    value {\n      tensor {\n        dtype: DT_INT32\n        tensor_shape {\n        }\n        int_val: 3\n      }\n    }\n  }\n}\nnode {\n  name: &quot;mixed3a&quot;\n  op: &quot;Concat&quot;\n  input: &quot;mixed3a/concat/dim&quot;\n  input: &quot;mixed3a/1x1&quot;\n  input: &quot;mixed3a/3x3&quot;\n  input: &quot;mixed3a/5x5&quot;\n  input: &quot;mixed3a/pool_reduce&quot;\n  device: &quot;/cpu:0&quot;\n  attr {\n    key: &quot;N&quot;\n    value {\n      i: 4\n    }\n  }\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n}\nnode {\n  name: &quot;mixed3b/1x1_pre_relu/conv&quot;\n  op: &quot;Conv2D&quot;\n  input: &quot;mixed3a&quot;\n  input: &quot;mixed3b/1x1_w&quot;\n  device: &quot;/cpu:0&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;padding&quot;\n    value {\n      s: &quot;SAME&quot;\n    }\n  }\n  attr {\n    key: &quot;strides&quot;\n    value {\n      list {\n        i: 1\n        i: 1\n        i: 1\n        i: 1\n      }\n    }\n  }\n  attr {\n    key: &quot;use_cudnn_on_gpu&quot;\n    value {\n      b: true\n    }\n  }\n}\nnode {\n  name: &quot;mixed3b/1x1_pre_relu&quot;\n  op: &quot;BiasAdd&quot;\n  input: &quot;mixed3b/1x1_pre_relu/conv&quot;\n  input: &quot;mixed3b/1x1_b&quot;\n  device: &quot;/cpu:0&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n}\nnode {\n  name: &quot;mixed3b/1x1&quot;\n  op: &quot;Relu&quot;\n  input: &quot;mixed3b/1x1_pre_relu&quot;\n  device: &quot;/cpu:0&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n}\nnode {\n  name: &quot;mixed3b/3x3_bottleneck_pre_relu/conv&quot;\n  op: &quot;Conv2D&quot;\n  input: &quot;mixed3a&quot;\n  input: &quot;mixed3b/3x3_bottleneck_w&quot;\n  device: &quot;/cpu:0&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;padding&quot;\n    value {\n      s: &quot;SAME&quot;\n    }\n  }\n  attr {\n    key: &quot;strides&quot;\n    value {\n      list {\n        i: 1\n        i: 1\n        i: 1\n        i: 1\n      }\n    }\n  }\n  attr {\n    key: &quot;use_cudnn_on_gpu&quot;\n    value {\n      b: true\n    }\n  }\n}\nnode {\n  name: &quot;mixed3b/3x3_bottleneck_pre_relu&quot;\n  op: &quot;BiasAdd&quot;\n  input: &quot;mixed3b/3x3_bottleneck_pre_relu/conv&quot;\n  input: &quot;mixed3b/3x3_bottleneck_b&quot;\n  device: &quot;/cpu:0&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n}\nnode {\n  name: &quot;mixed3b/3x3_bottleneck&quot;\n  op: &quot;Relu&quot;\n  input: &quot;mixed3b/3x3_bottleneck_pre_relu&quot;\n  device: &quot;/cpu:0&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n}\nnode {\n  name: &quot;mixed3b/3x3_pre_relu/conv&quot;\n  op: &quot;Conv2D&quot;\n  input: &quot;mixed3b/3x3_bottleneck&quot;\n  input: &quot;mixed3b/3x3_w&quot;\n  device: &quot;/cpu:0&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;padding&quot;\n    value {\n      s: &quot;SAME&quot;\n    }\n  }\n  attr {\n    key: &quot;strides&quot;\n    value {\n      list {\n        i: 1\n        i: 1\n        i: 1\n        i: 1\n      }\n    }\n  }\n  attr {\n    key: &quot;use_cudnn_on_gpu&quot;\n    value {\n      b: true\n    }\n  }\n}\nnode {\n  name: &quot;mixed3b/3x3_pre_relu&quot;\n  op: &quot;BiasAdd&quot;\n  input: &quot;mixed3b/3x3_pre_relu/conv&quot;\n  input: &quot;mixed3b/3x3_b&quot;\n  device: &quot;/cpu:0&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n}\nnode {\n  name: &quot;mixed3b/3x3&quot;\n  op: &quot;Relu&quot;\n  input: &quot;mixed3b/3x3_pre_relu&quot;\n  device: &quot;/cpu:0&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n}\nnode {\n  name: &quot;mixed3b/5x5_bottleneck_pre_relu/conv&quot;\n  op: &quot;Conv2D&quot;\n  input: &quot;mixed3a&quot;\n  input: &quot;mixed3b/5x5_bottleneck_w&quot;\n  device: &quot;/cpu:0&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;padding&quot;\n    value {\n      s: &quot;SAME&quot;\n    }\n  }\n  attr {\n    key: &quot;strides&quot;\n    value {\n      list {\n        i: 1\n        i: 1\n        i: 1\n        i: 1\n      }\n    }\n  }\n  attr {\n    key: &quot;use_cudnn_on_gpu&quot;\n    value {\n      b: true\n    }\n  }\n}\nnode {\n  name: &quot;mixed3b/5x5_bottleneck_pre_relu&quot;\n  op: &quot;BiasAdd&quot;\n  input: &quot;mixed3b/5x5_bottleneck_pre_relu/conv&quot;\n  input: &quot;mixed3b/5x5_bottleneck_b&quot;\n  device: &quot;/cpu:0&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n}\nnode {\n  name: &quot;mixed3b/5x5_bottleneck&quot;\n  op: &quot;Relu&quot;\n  input: &quot;mixed3b/5x5_bottleneck_pre_relu&quot;\n  device: &quot;/cpu:0&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n}\nnode {\n  name: &quot;mixed3b/5x5_pre_relu/conv&quot;\n  op: &quot;Conv2D&quot;\n  input: &quot;mixed3b/5x5_bottleneck&quot;\n  input: &quot;mixed3b/5x5_w&quot;\n  device: &quot;/cpu:0&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;padding&quot;\n    value {\n      s: &quot;SAME&quot;\n    }\n  }\n  attr {\n    key: &quot;strides&quot;\n    value {\n      list {\n        i: 1\n        i: 1\n        i: 1\n        i: 1\n      }\n    }\n  }\n  attr {\n    key: &quot;use_cudnn_on_gpu&quot;\n    value {\n      b: true\n    }\n  }\n}\nnode {\n  name: &quot;mixed3b/5x5_pre_relu&quot;\n  op: &quot;BiasAdd&quot;\n  input: &quot;mixed3b/5x5_pre_relu/conv&quot;\n  input: &quot;mixed3b/5x5_b&quot;\n  device: &quot;/cpu:0&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n}\nnode {\n  name: &quot;mixed3b/5x5&quot;\n  op: &quot;Relu&quot;\n  input: &quot;mixed3b/5x5_pre_relu&quot;\n  device: &quot;/cpu:0&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n}\nnode {\n  name: &quot;mixed3b/pool&quot;\n  op: &quot;MaxPool&quot;\n  input: &quot;mixed3a&quot;\n  device: &quot;/cpu:0&quot;\n  attr {\n    key: &quot;ksize&quot;\n    value {\n      list {\n        i: 1\n        i: 3\n        i: 3\n        i: 1\n      }\n    }\n  }\n  attr {\n    key: &quot;padding&quot;\n    value {\n      s: &quot;SAME&quot;\n    }\n  }\n  attr {\n    key: &quot;strides&quot;\n    value {\n      list {\n        i: 1\n        i: 1\n        i: 1\n        i: 1\n      }\n    }\n  }\n}\nnode {\n  name: &quot;mixed3b/pool_reduce_pre_relu/conv&quot;\n  op: &quot;Conv2D&quot;\n  input: &quot;mixed3b/pool&quot;\n  input: &quot;mixed3b/pool_reduce_w&quot;\n  device: &quot;/cpu:0&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;padding&quot;\n    value {\n      s: &quot;SAME&quot;\n    }\n  }\n  attr {\n    key: &quot;strides&quot;\n    value {\n      list {\n        i: 1\n        i: 1\n        i: 1\n        i: 1\n      }\n    }\n  }\n  attr {\n    key: &quot;use_cudnn_on_gpu&quot;\n    value {\n      b: true\n    }\n  }\n}\nnode {\n  name: &quot;mixed3b/pool_reduce_pre_relu&quot;\n  op: &quot;BiasAdd&quot;\n  input: &quot;mixed3b/pool_reduce_pre_relu/conv&quot;\n  input: &quot;mixed3b/pool_reduce_b&quot;\n  device: &quot;/cpu:0&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n}\nnode {\n  name: &quot;mixed3b/pool_reduce&quot;\n  op: &quot;Relu&quot;\n  input: &quot;mixed3b/pool_reduce_pre_relu&quot;\n  device: &quot;/cpu:0&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n}\nnode {\n  name: &quot;mixed3b/concat/dim&quot;\n  op: &quot;Const&quot;\n  device: &quot;/cpu:0&quot;\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_INT32\n    }\n  }\n  attr {\n    key: &quot;value&quot;\n    value {\n      tensor {\n        dtype: DT_INT32\n        tensor_shape {\n        }\n        int_val: 3\n      }\n    }\n  }\n}\nnode {\n  name: &quot;mixed3b&quot;\n  op: &quot;Concat&quot;\n  input: &quot;mixed3b/concat/dim&quot;\n  input: &quot;mixed3b/1x1&quot;\n  input: &quot;mixed3b/3x3&quot;\n  input: &quot;mixed3b/5x5&quot;\n  input: &quot;mixed3b/pool_reduce&quot;\n  device: &quot;/cpu:0&quot;\n  attr {\n    key: &quot;N&quot;\n    value {\n      i: 4\n    }\n  }\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n}\nnode {\n  name: &quot;maxpool4&quot;\n  op: &quot;MaxPool&quot;\n  input: &quot;mixed3b&quot;\n  device: &quot;/cpu:0&quot;\n  attr {\n    key: &quot;ksize&quot;\n    value {\n      list {\n        i: 1\n        i: 3\n        i: 3\n        i: 1\n      }\n    }\n  }\n  attr {\n    key: &quot;padding&quot;\n    value {\n      s: &quot;SAME&quot;\n    }\n  }\n  attr {\n    key: &quot;strides&quot;\n    value {\n      list {\n        i: 1\n        i: 2\n        i: 2\n        i: 1\n      }\n    }\n  }\n}\nnode {\n  name: &quot;mixed4a/1x1_pre_relu/conv&quot;\n  op: &quot;Conv2D&quot;\n  input: &quot;maxpool4&quot;\n  input: &quot;mixed4a/1x1_w&quot;\n  device: &quot;/cpu:0&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;padding&quot;\n    value {\n      s: &quot;SAME&quot;\n    }\n  }\n  attr {\n    key: &quot;strides&quot;\n    value {\n      list {\n        i: 1\n        i: 1\n        i: 1\n        i: 1\n      }\n    }\n  }\n  attr {\n    key: &quot;use_cudnn_on_gpu&quot;\n    value {\n      b: true\n    }\n  }\n}\nnode {\n  name: &quot;mixed4a/1x1_pre_relu&quot;\n  op: &quot;BiasAdd&quot;\n  input: &quot;mixed4a/1x1_pre_relu/conv&quot;\n  input: &quot;mixed4a/1x1_b&quot;\n  device: &quot;/cpu:0&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n}\nnode {\n  name: &quot;mixed4a/1x1&quot;\n  op: &quot;Relu&quot;\n  input: &quot;mixed4a/1x1_pre_relu&quot;\n  device: &quot;/cpu:0&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n}\nnode {\n  name: &quot;mixed4a/3x3_bottleneck_pre_relu/conv&quot;\n  op: &quot;Conv2D&quot;\n  input: &quot;maxpool4&quot;\n  input: &quot;mixed4a/3x3_bottleneck_w&quot;\n  device: &quot;/cpu:0&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;padding&quot;\n    value {\n      s: &quot;SAME&quot;\n    }\n  }\n  attr {\n    key: &quot;strides&quot;\n    value {\n      list {\n        i: 1\n        i: 1\n        i: 1\n        i: 1\n      }\n    }\n  }\n  attr {\n    key: &quot;use_cudnn_on_gpu&quot;\n    value {\n      b: true\n    }\n  }\n}\nnode {\n  name: &quot;mixed4a/3x3_bottleneck_pre_relu&quot;\n  op: &quot;BiasAdd&quot;\n  input: &quot;mixed4a/3x3_bottleneck_pre_relu/conv&quot;\n  input: &quot;mixed4a/3x3_bottleneck_b&quot;\n  device: &quot;/cpu:0&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n}\nnode {\n  name: &quot;mixed4a/3x3_bottleneck&quot;\n  op: &quot;Relu&quot;\n  input: &quot;mixed4a/3x3_bottleneck_pre_relu&quot;\n  device: &quot;/cpu:0&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n}\nnode {\n  name: &quot;mixed4a/3x3_pre_relu/conv&quot;\n  op: &quot;Conv2D&quot;\n  input: &quot;mixed4a/3x3_bottleneck&quot;\n  input: &quot;mixed4a/3x3_w&quot;\n  device: &quot;/cpu:0&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;padding&quot;\n    value {\n      s: &quot;SAME&quot;\n    }\n  }\n  attr {\n    key: &quot;strides&quot;\n    value {\n      list {\n        i: 1\n        i: 1\n        i: 1\n        i: 1\n      }\n    }\n  }\n  attr {\n    key: &quot;use_cudnn_on_gpu&quot;\n    value {\n      b: true\n    }\n  }\n}\nnode {\n  name: &quot;mixed4a/3x3_pre_relu&quot;\n  op: &quot;BiasAdd&quot;\n  input: &quot;mixed4a/3x3_pre_relu/conv&quot;\n  input: &quot;mixed4a/3x3_b&quot;\n  device: &quot;/cpu:0&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n}\nnode {\n  name: &quot;mixed4a/3x3&quot;\n  op: &quot;Relu&quot;\n  input: &quot;mixed4a/3x3_pre_relu&quot;\n  device: &quot;/cpu:0&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n}\nnode {\n  name: &quot;mixed4a/5x5_bottleneck_pre_relu/conv&quot;\n  op: &quot;Conv2D&quot;\n  input: &quot;maxpool4&quot;\n  input: &quot;mixed4a/5x5_bottleneck_w&quot;\n  device: &quot;/cpu:0&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;padding&quot;\n    value {\n      s: &quot;SAME&quot;\n    }\n  }\n  attr {\n    key: &quot;strides&quot;\n    value {\n      list {\n        i: 1\n        i: 1\n        i: 1\n        i: 1\n      }\n    }\n  }\n  attr {\n    key: &quot;use_cudnn_on_gpu&quot;\n    value {\n      b: true\n    }\n  }\n}\nnode {\n  name: &quot;mixed4a/5x5_bottleneck_pre_relu&quot;\n  op: &quot;BiasAdd&quot;\n  input: &quot;mixed4a/5x5_bottleneck_pre_relu/conv&quot;\n  input: &quot;mixed4a/5x5_bottleneck_b&quot;\n  device: &quot;/cpu:0&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n}\nnode {\n  name: &quot;mixed4a/5x5_bottleneck&quot;\n  op: &quot;Relu&quot;\n  input: &quot;mixed4a/5x5_bottleneck_pre_relu&quot;\n  device: &quot;/cpu:0&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n}\nnode {\n  name: &quot;mixed4a/5x5_pre_relu/conv&quot;\n  op: &quot;Conv2D&quot;\n  input: &quot;mixed4a/5x5_bottleneck&quot;\n  input: &quot;mixed4a/5x5_w&quot;\n  device: &quot;/cpu:0&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;padding&quot;\n    value {\n      s: &quot;SAME&quot;\n    }\n  }\n  attr {\n    key: &quot;strides&quot;\n    value {\n      list {\n        i: 1\n        i: 1\n        i: 1\n        i: 1\n      }\n    }\n  }\n  attr {\n    key: &quot;use_cudnn_on_gpu&quot;\n    value {\n      b: true\n    }\n  }\n}\nnode {\n  name: &quot;mixed4a/5x5_pre_relu&quot;\n  op: &quot;BiasAdd&quot;\n  input: &quot;mixed4a/5x5_pre_relu/conv&quot;\n  input: &quot;mixed4a/5x5_b&quot;\n  device: &quot;/cpu:0&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n}\nnode {\n  name: &quot;mixed4a/5x5&quot;\n  op: &quot;Relu&quot;\n  input: &quot;mixed4a/5x5_pre_relu&quot;\n  device: &quot;/cpu:0&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n}\nnode {\n  name: &quot;mixed4a/pool&quot;\n  op: &quot;MaxPool&quot;\n  input: &quot;maxpool4&quot;\n  device: &quot;/cpu:0&quot;\n  attr {\n    key: &quot;ksize&quot;\n    value {\n      list {\n        i: 1\n        i: 3\n        i: 3\n        i: 1\n      }\n    }\n  }\n  attr {\n    key: &quot;padding&quot;\n    value {\n      s: &quot;SAME&quot;\n    }\n  }\n  attr {\n    key: &quot;strides&quot;\n    value {\n      list {\n        i: 1\n        i: 1\n        i: 1\n        i: 1\n      }\n    }\n  }\n}\nnode {\n  name: &quot;mixed4a/pool_reduce_pre_relu/conv&quot;\n  op: &quot;Conv2D&quot;\n  input: &quot;mixed4a/pool&quot;\n  input: &quot;mixed4a/pool_reduce_w&quot;\n  device: &quot;/cpu:0&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;padding&quot;\n    value {\n      s: &quot;SAME&quot;\n    }\n  }\n  attr {\n    key: &quot;strides&quot;\n    value {\n      list {\n        i: 1\n        i: 1\n        i: 1\n        i: 1\n      }\n    }\n  }\n  attr {\n    key: &quot;use_cudnn_on_gpu&quot;\n    value {\n      b: true\n    }\n  }\n}\nnode {\n  name: &quot;mixed4a/pool_reduce_pre_relu&quot;\n  op: &quot;BiasAdd&quot;\n  input: &quot;mixed4a/pool_reduce_pre_relu/conv&quot;\n  input: &quot;mixed4a/pool_reduce_b&quot;\n  device: &quot;/cpu:0&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n}\nnode {\n  name: &quot;mixed4a/pool_reduce&quot;\n  op: &quot;Relu&quot;\n  input: &quot;mixed4a/pool_reduce_pre_relu&quot;\n  device: &quot;/cpu:0&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n}\nnode {\n  name: &quot;mixed4a/concat/dim&quot;\n  op: &quot;Const&quot;\n  device: &quot;/cpu:0&quot;\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_INT32\n    }\n  }\n  attr {\n    key: &quot;value&quot;\n    value {\n      tensor {\n        dtype: DT_INT32\n        tensor_shape {\n        }\n        int_val: 3\n      }\n    }\n  }\n}\nnode {\n  name: &quot;mixed4a&quot;\n  op: &quot;Concat&quot;\n  input: &quot;mixed4a/concat/dim&quot;\n  input: &quot;mixed4a/1x1&quot;\n  input: &quot;mixed4a/3x3&quot;\n  input: &quot;mixed4a/5x5&quot;\n  input: &quot;mixed4a/pool_reduce&quot;\n  device: &quot;/cpu:0&quot;\n  attr {\n    key: &quot;N&quot;\n    value {\n      i: 4\n    }\n  }\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n}\nnode {\n  name: &quot;mixed4b/1x1_pre_relu/conv&quot;\n  op: &quot;Conv2D&quot;\n  input: &quot;mixed4a&quot;\n  input: &quot;mixed4b/1x1_w&quot;\n  device: &quot;/cpu:0&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;padding&quot;\n    value {\n      s: &quot;SAME&quot;\n    }\n  }\n  attr {\n    key: &quot;strides&quot;\n    value {\n      list {\n        i: 1\n        i: 1\n        i: 1\n        i: 1\n      }\n    }\n  }\n  attr {\n    key: &quot;use_cudnn_on_gpu&quot;\n    value {\n      b: true\n    }\n  }\n}\nnode {\n  name: &quot;mixed4b/1x1_pre_relu&quot;\n  op: &quot;BiasAdd&quot;\n  input: &quot;mixed4b/1x1_pre_relu/conv&quot;\n  input: &quot;mixed4b/1x1_b&quot;\n  device: &quot;/cpu:0&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n}\nnode {\n  name: &quot;mixed4b/1x1&quot;\n  op: &quot;Relu&quot;\n  input: &quot;mixed4b/1x1_pre_relu&quot;\n  device: &quot;/cpu:0&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n}\nnode {\n  name: &quot;mixed4b/3x3_bottleneck_pre_relu/conv&quot;\n  op: &quot;Conv2D&quot;\n  input: &quot;mixed4a&quot;\n  input: &quot;mixed4b/3x3_bottleneck_w&quot;\n  device: &quot;/cpu:0&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;padding&quot;\n    value {\n      s: &quot;SAME&quot;\n    }\n  }\n  attr {\n    key: &quot;strides&quot;\n    value {\n      list {\n        i: 1\n        i: 1\n        i: 1\n        i: 1\n      }\n    }\n  }\n  attr {\n    key: &quot;use_cudnn_on_gpu&quot;\n    value {\n      b: true\n    }\n  }\n}\nnode {\n  name: &quot;mixed4b/3x3_bottleneck_pre_relu&quot;\n  op: &quot;BiasAdd&quot;\n  input: &quot;mixed4b/3x3_bottleneck_pre_relu/conv&quot;\n  input: &quot;mixed4b/3x3_bottleneck_b&quot;\n  device: &quot;/cpu:0&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n}\nnode {\n  name: &quot;mixed4b/3x3_bottleneck&quot;\n  op: &quot;Relu&quot;\n  input: &quot;mixed4b/3x3_bottleneck_pre_relu&quot;\n  device: &quot;/cpu:0&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n}\nnode {\n  name: &quot;mixed4b/3x3_pre_relu/conv&quot;\n  op: &quot;Conv2D&quot;\n  input: &quot;mixed4b/3x3_bottleneck&quot;\n  input: &quot;mixed4b/3x3_w&quot;\n  device: &quot;/cpu:0&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;padding&quot;\n    value {\n      s: &quot;SAME&quot;\n    }\n  }\n  attr {\n    key: &quot;strides&quot;\n    value {\n      list {\n        i: 1\n        i: 1\n        i: 1\n        i: 1\n      }\n    }\n  }\n  attr {\n    key: &quot;use_cudnn_on_gpu&quot;\n    value {\n      b: true\n    }\n  }\n}\nnode {\n  name: &quot;mixed4b/3x3_pre_relu&quot;\n  op: &quot;BiasAdd&quot;\n  input: &quot;mixed4b/3x3_pre_relu/conv&quot;\n  input: &quot;mixed4b/3x3_b&quot;\n  device: &quot;/cpu:0&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n}\nnode {\n  name: &quot;mixed4b/3x3&quot;\n  op: &quot;Relu&quot;\n  input: &quot;mixed4b/3x3_pre_relu&quot;\n  device: &quot;/cpu:0&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n}\nnode {\n  name: &quot;mixed4b/5x5_bottleneck_pre_relu/conv&quot;\n  op: &quot;Conv2D&quot;\n  input: &quot;mixed4a&quot;\n  input: &quot;mixed4b/5x5_bottleneck_w&quot;\n  device: &quot;/cpu:0&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;padding&quot;\n    value {\n      s: &quot;SAME&quot;\n    }\n  }\n  attr {\n    key: &quot;strides&quot;\n    value {\n      list {\n        i: 1\n        i: 1\n        i: 1\n        i: 1\n      }\n    }\n  }\n  attr {\n    key: &quot;use_cudnn_on_gpu&quot;\n    value {\n      b: true\n    }\n  }\n}\nnode {\n  name: &quot;mixed4b/5x5_bottleneck_pre_relu&quot;\n  op: &quot;BiasAdd&quot;\n  input: &quot;mixed4b/5x5_bottleneck_pre_relu/conv&quot;\n  input: &quot;mixed4b/5x5_bottleneck_b&quot;\n  device: &quot;/cpu:0&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n}\nnode {\n  name: &quot;mixed4b/5x5_bottleneck&quot;\n  op: &quot;Relu&quot;\n  input: &quot;mixed4b/5x5_bottleneck_pre_relu&quot;\n  device: &quot;/cpu:0&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n}\nnode {\n  name: &quot;mixed4b/5x5_pre_relu/conv&quot;\n  op: &quot;Conv2D&quot;\n  input: &quot;mixed4b/5x5_bottleneck&quot;\n  input: &quot;mixed4b/5x5_w&quot;\n  device: &quot;/cpu:0&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;padding&quot;\n    value {\n      s: &quot;SAME&quot;\n    }\n  }\n  attr {\n    key: &quot;strides&quot;\n    value {\n      list {\n        i: 1\n        i: 1\n        i: 1\n        i: 1\n      }\n    }\n  }\n  attr {\n    key: &quot;use_cudnn_on_gpu&quot;\n    value {\n      b: true\n    }\n  }\n}\nnode {\n  name: &quot;mixed4b/5x5_pre_relu&quot;\n  op: &quot;BiasAdd&quot;\n  input: &quot;mixed4b/5x5_pre_relu/conv&quot;\n  input: &quot;mixed4b/5x5_b&quot;\n  device: &quot;/cpu:0&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n}\nnode {\n  name: &quot;mixed4b/5x5&quot;\n  op: &quot;Relu&quot;\n  input: &quot;mixed4b/5x5_pre_relu&quot;\n  device: &quot;/cpu:0&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n}\nnode {\n  name: &quot;mixed4b/pool&quot;\n  op: &quot;MaxPool&quot;\n  input: &quot;mixed4a&quot;\n  device: &quot;/cpu:0&quot;\n  attr {\n    key: &quot;ksize&quot;\n    value {\n      list {\n        i: 1\n        i: 3\n        i: 3\n        i: 1\n      }\n    }\n  }\n  attr {\n    key: &quot;padding&quot;\n    value {\n      s: &quot;SAME&quot;\n    }\n  }\n  attr {\n    key: &quot;strides&quot;\n    value {\n      list {\n        i: 1\n        i: 1\n        i: 1\n        i: 1\n      }\n    }\n  }\n}\nnode {\n  name: &quot;mixed4b/pool_reduce_pre_relu/conv&quot;\n  op: &quot;Conv2D&quot;\n  input: &quot;mixed4b/pool&quot;\n  input: &quot;mixed4b/pool_reduce_w&quot;\n  device: &quot;/cpu:0&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;padding&quot;\n    value {\n      s: &quot;SAME&quot;\n    }\n  }\n  attr {\n    key: &quot;strides&quot;\n    value {\n      list {\n        i: 1\n        i: 1\n        i: 1\n        i: 1\n      }\n    }\n  }\n  attr {\n    key: &quot;use_cudnn_on_gpu&quot;\n    value {\n      b: true\n    }\n  }\n}\nnode {\n  name: &quot;mixed4b/pool_reduce_pre_relu&quot;\n  op: &quot;BiasAdd&quot;\n  input: &quot;mixed4b/pool_reduce_pre_relu/conv&quot;\n  input: &quot;mixed4b/pool_reduce_b&quot;\n  device: &quot;/cpu:0&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n}\nnode {\n  name: &quot;mixed4b/pool_reduce&quot;\n  op: &quot;Relu&quot;\n  input: &quot;mixed4b/pool_reduce_pre_relu&quot;\n  device: &quot;/cpu:0&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n}\nnode {\n  name: &quot;mixed4b/concat/dim&quot;\n  op: &quot;Const&quot;\n  device: &quot;/cpu:0&quot;\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_INT32\n    }\n  }\n  attr {\n    key: &quot;value&quot;\n    value {\n      tensor {\n        dtype: DT_INT32\n        tensor_shape {\n        }\n        int_val: 3\n      }\n    }\n  }\n}\nnode {\n  name: &quot;mixed4b&quot;\n  op: &quot;Concat&quot;\n  input: &quot;mixed4b/concat/dim&quot;\n  input: &quot;mixed4b/1x1&quot;\n  input: &quot;mixed4b/3x3&quot;\n  input: &quot;mixed4b/5x5&quot;\n  input: &quot;mixed4b/pool_reduce&quot;\n  device: &quot;/cpu:0&quot;\n  attr {\n    key: &quot;N&quot;\n    value {\n      i: 4\n    }\n  }\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n}\nnode {\n  name: &quot;mixed4c/1x1_pre_relu/conv&quot;\n  op: &quot;Conv2D&quot;\n  input: &quot;mixed4b&quot;\n  input: &quot;mixed4c/1x1_w&quot;\n  device: &quot;/cpu:0&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;padding&quot;\n    value {\n      s: &quot;SAME&quot;\n    }\n  }\n  attr {\n    key: &quot;strides&quot;\n    value {\n      list {\n        i: 1\n        i: 1\n        i: 1\n        i: 1\n      }\n    }\n  }\n  attr {\n    key: &quot;use_cudnn_on_gpu&quot;\n    value {\n      b: true\n    }\n  }\n}\nnode {\n  name: &quot;mixed4c/1x1_pre_relu&quot;\n  op: &quot;BiasAdd&quot;\n  input: &quot;mixed4c/1x1_pre_relu/conv&quot;\n  input: &quot;mixed4c/1x1_b&quot;\n  device: &quot;/cpu:0&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n}\nnode {\n  name: &quot;mixed4c/1x1&quot;\n  op: &quot;Relu&quot;\n  input: &quot;mixed4c/1x1_pre_relu&quot;\n  device: &quot;/cpu:0&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n}\nnode {\n  name: &quot;mixed4c/3x3_bottleneck_pre_relu/conv&quot;\n  op: &quot;Conv2D&quot;\n  input: &quot;mixed4b&quot;\n  input: &quot;mixed4c/3x3_bottleneck_w&quot;\n  device: &quot;/cpu:0&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;padding&quot;\n    value {\n      s: &quot;SAME&quot;\n    }\n  }\n  attr {\n    key: &quot;strides&quot;\n    value {\n      list {\n        i: 1\n        i: 1\n        i: 1\n        i: 1\n      }\n    }\n  }\n  attr {\n    key: &quot;use_cudnn_on_gpu&quot;\n    value {\n      b: true\n    }\n  }\n}\nnode {\n  name: &quot;mixed4c/3x3_bottleneck_pre_relu&quot;\n  op: &quot;BiasAdd&quot;\n  input: &quot;mixed4c/3x3_bottleneck_pre_relu/conv&quot;\n  input: &quot;mixed4c/3x3_bottleneck_b&quot;\n  device: &quot;/cpu:0&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n}\nnode {\n  name: &quot;mixed4c/3x3_bottleneck&quot;\n  op: &quot;Relu&quot;\n  input: &quot;mixed4c/3x3_bottleneck_pre_relu&quot;\n  device: &quot;/cpu:0&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n}\nnode {\n  name: &quot;mixed4c/3x3_pre_relu/conv&quot;\n  op: &quot;Conv2D&quot;\n  input: &quot;mixed4c/3x3_bottleneck&quot;\n  input: &quot;mixed4c/3x3_w&quot;\n  device: &quot;/cpu:0&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;padding&quot;\n    value {\n      s: &quot;SAME&quot;\n    }\n  }\n  attr {\n    key: &quot;strides&quot;\n    value {\n      list {\n        i: 1\n        i: 1\n        i: 1\n        i: 1\n      }\n    }\n  }\n  attr {\n    key: &quot;use_cudnn_on_gpu&quot;\n    value {\n      b: true\n    }\n  }\n}\nnode {\n  name: &quot;mixed4c/3x3_pre_relu&quot;\n  op: &quot;BiasAdd&quot;\n  input: &quot;mixed4c/3x3_pre_relu/conv&quot;\n  input: &quot;mixed4c/3x3_b&quot;\n  device: &quot;/cpu:0&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n}\nnode {\n  name: &quot;mixed4c/3x3&quot;\n  op: &quot;Relu&quot;\n  input: &quot;mixed4c/3x3_pre_relu&quot;\n  device: &quot;/cpu:0&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n}\nnode {\n  name: &quot;mixed4c/5x5_bottleneck_pre_relu/conv&quot;\n  op: &quot;Conv2D&quot;\n  input: &quot;mixed4b&quot;\n  input: &quot;mixed4c/5x5_bottleneck_w&quot;\n  device: &quot;/cpu:0&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;padding&quot;\n    value {\n      s: &quot;SAME&quot;\n    }\n  }\n  attr {\n    key: &quot;strides&quot;\n    value {\n      list {\n        i: 1\n        i: 1\n        i: 1\n        i: 1\n      }\n    }\n  }\n  attr {\n    key: &quot;use_cudnn_on_gpu&quot;\n    value {\n      b: true\n    }\n  }\n}\nnode {\n  name: &quot;mixed4c/5x5_bottleneck_pre_relu&quot;\n  op: &quot;BiasAdd&quot;\n  input: &quot;mixed4c/5x5_bottleneck_pre_relu/conv&quot;\n  input: &quot;mixed4c/5x5_bottleneck_b&quot;\n  device: &quot;/cpu:0&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n}\nnode {\n  name: &quot;mixed4c/5x5_bottleneck&quot;\n  op: &quot;Relu&quot;\n  input: &quot;mixed4c/5x5_bottleneck_pre_relu&quot;\n  device: &quot;/cpu:0&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n}\nnode {\n  name: &quot;mixed4c/5x5_pre_relu/conv&quot;\n  op: &quot;Conv2D&quot;\n  input: &quot;mixed4c/5x5_bottleneck&quot;\n  input: &quot;mixed4c/5x5_w&quot;\n  device: &quot;/cpu:0&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;padding&quot;\n    value {\n      s: &quot;SAME&quot;\n    }\n  }\n  attr {\n    key: &quot;strides&quot;\n    value {\n      list {\n        i: 1\n        i: 1\n        i: 1\n        i: 1\n      }\n    }\n  }\n  attr {\n    key: &quot;use_cudnn_on_gpu&quot;\n    value {\n      b: true\n    }\n  }\n}\nnode {\n  name: &quot;mixed4c/5x5_pre_relu&quot;\n  op: &quot;BiasAdd&quot;\n  input: &quot;mixed4c/5x5_pre_relu/conv&quot;\n  input: &quot;mixed4c/5x5_b&quot;\n  device: &quot;/cpu:0&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n}\nnode {\n  name: &quot;mixed4c/5x5&quot;\n  op: &quot;Relu&quot;\n  input: &quot;mixed4c/5x5_pre_relu&quot;\n  device: &quot;/cpu:0&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n}\nnode {\n  name: &quot;mixed4c/pool&quot;\n  op: &quot;MaxPool&quot;\n  input: &quot;mixed4b&quot;\n  device: &quot;/cpu:0&quot;\n  attr {\n    key: &quot;ksize&quot;\n    value {\n      list {\n        i: 1\n        i: 3\n        i: 3\n        i: 1\n      }\n    }\n  }\n  attr {\n    key: &quot;padding&quot;\n    value {\n      s: &quot;SAME&quot;\n    }\n  }\n  attr {\n    key: &quot;strides&quot;\n    value {\n      list {\n        i: 1\n        i: 1\n        i: 1\n        i: 1\n      }\n    }\n  }\n}\nnode {\n  name: &quot;mixed4c/pool_reduce_pre_relu/conv&quot;\n  op: &quot;Conv2D&quot;\n  input: &quot;mixed4c/pool&quot;\n  input: &quot;mixed4c/pool_reduce_w&quot;\n  device: &quot;/cpu:0&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;padding&quot;\n    value {\n      s: &quot;SAME&quot;\n    }\n  }\n  attr {\n    key: &quot;strides&quot;\n    value {\n      list {\n        i: 1\n        i: 1\n        i: 1\n        i: 1\n      }\n    }\n  }\n  attr {\n    key: &quot;use_cudnn_on_gpu&quot;\n    value {\n      b: true\n    }\n  }\n}\nnode {\n  name: &quot;mixed4c/pool_reduce_pre_relu&quot;\n  op: &quot;BiasAdd&quot;\n  input: &quot;mixed4c/pool_reduce_pre_relu/conv&quot;\n  input: &quot;mixed4c/pool_reduce_b&quot;\n  device: &quot;/cpu:0&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n}\nnode {\n  name: &quot;mixed4c/pool_reduce&quot;\n  op: &quot;Relu&quot;\n  input: &quot;mixed4c/pool_reduce_pre_relu&quot;\n  device: &quot;/cpu:0&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n}\nnode {\n  name: &quot;mixed4c/concat/dim&quot;\n  op: &quot;Const&quot;\n  device: &quot;/cpu:0&quot;\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_INT32\n    }\n  }\n  attr {\n    key: &quot;value&quot;\n    value {\n      tensor {\n        dtype: DT_INT32\n        tensor_shape {\n        }\n        int_val: 3\n      }\n    }\n  }\n}\nnode {\n  name: &quot;mixed4c&quot;\n  op: &quot;Concat&quot;\n  input: &quot;mixed4c/concat/dim&quot;\n  input: &quot;mixed4c/1x1&quot;\n  input: &quot;mixed4c/3x3&quot;\n  input: &quot;mixed4c/5x5&quot;\n  input: &quot;mixed4c/pool_reduce&quot;\n  device: &quot;/cpu:0&quot;\n  attr {\n    key: &quot;N&quot;\n    value {\n      i: 4\n    }\n  }\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n}\nnode {\n  name: &quot;mixed4d/1x1_pre_relu/conv&quot;\n  op: &quot;Conv2D&quot;\n  input: &quot;mixed4c&quot;\n  input: &quot;mixed4d/1x1_w&quot;\n  device: &quot;/cpu:0&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;padding&quot;\n    value {\n      s: &quot;SAME&quot;\n    }\n  }\n  attr {\n    key: &quot;strides&quot;\n    value {\n      list {\n        i: 1\n        i: 1\n        i: 1\n        i: 1\n      }\n    }\n  }\n  attr {\n    key: &quot;use_cudnn_on_gpu&quot;\n    value {\n      b: true\n    }\n  }\n}\nnode {\n  name: &quot;mixed4d/1x1_pre_relu&quot;\n  op: &quot;BiasAdd&quot;\n  input: &quot;mixed4d/1x1_pre_relu/conv&quot;\n  input: &quot;mixed4d/1x1_b&quot;\n  device: &quot;/cpu:0&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n}\nnode {\n  name: &quot;mixed4d/1x1&quot;\n  op: &quot;Relu&quot;\n  input: &quot;mixed4d/1x1_pre_relu&quot;\n  device: &quot;/cpu:0&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n}\nnode {\n  name: &quot;mixed4d/3x3_bottleneck_pre_relu/conv&quot;\n  op: &quot;Conv2D&quot;\n  input: &quot;mixed4c&quot;\n  input: &quot;mixed4d/3x3_bottleneck_w&quot;\n  device: &quot;/cpu:0&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;padding&quot;\n    value {\n      s: &quot;SAME&quot;\n    }\n  }\n  attr {\n    key: &quot;strides&quot;\n    value {\n      list {\n        i: 1\n        i: 1\n        i: 1\n        i: 1\n      }\n    }\n  }\n  attr {\n    key: &quot;use_cudnn_on_gpu&quot;\n    value {\n      b: true\n    }\n  }\n}\nnode {\n  name: &quot;mixed4d/3x3_bottleneck_pre_relu&quot;\n  op: &quot;BiasAdd&quot;\n  input: &quot;mixed4d/3x3_bottleneck_pre_relu/conv&quot;\n  input: &quot;mixed4d/3x3_bottleneck_b&quot;\n  device: &quot;/cpu:0&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n}\nnode {\n  name: &quot;mixed4d/3x3_bottleneck&quot;\n  op: &quot;Relu&quot;\n  input: &quot;mixed4d/3x3_bottleneck_pre_relu&quot;\n  device: &quot;/cpu:0&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n}\nnode {\n  name: &quot;mixed4d/3x3_pre_relu/conv&quot;\n  op: &quot;Conv2D&quot;\n  input: &quot;mixed4d/3x3_bottleneck&quot;\n  input: &quot;mixed4d/3x3_w&quot;\n  device: &quot;/cpu:0&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;padding&quot;\n    value {\n      s: &quot;SAME&quot;\n    }\n  }\n  attr {\n    key: &quot;strides&quot;\n    value {\n      list {\n        i: 1\n        i: 1\n        i: 1\n        i: 1\n      }\n    }\n  }\n  attr {\n    key: &quot;use_cudnn_on_gpu&quot;\n    value {\n      b: true\n    }\n  }\n}\nnode {\n  name: &quot;mixed4d/3x3_pre_relu&quot;\n  op: &quot;BiasAdd&quot;\n  input: &quot;mixed4d/3x3_pre_relu/conv&quot;\n  input: &quot;mixed4d/3x3_b&quot;\n  device: &quot;/cpu:0&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n}\nnode {\n  name: &quot;mixed4d/3x3&quot;\n  op: &quot;Relu&quot;\n  input: &quot;mixed4d/3x3_pre_relu&quot;\n  device: &quot;/cpu:0&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n}\nnode {\n  name: &quot;mixed4d/5x5_bottleneck_pre_relu/conv&quot;\n  op: &quot;Conv2D&quot;\n  input: &quot;mixed4c&quot;\n  input: &quot;mixed4d/5x5_bottleneck_w&quot;\n  device: &quot;/cpu:0&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;padding&quot;\n    value {\n      s: &quot;SAME&quot;\n    }\n  }\n  attr {\n    key: &quot;strides&quot;\n    value {\n      list {\n        i: 1\n        i: 1\n        i: 1\n        i: 1\n      }\n    }\n  }\n  attr {\n    key: &quot;use_cudnn_on_gpu&quot;\n    value {\n      b: true\n    }\n  }\n}\nnode {\n  name: &quot;mixed4d/5x5_bottleneck_pre_relu&quot;\n  op: &quot;BiasAdd&quot;\n  input: &quot;mixed4d/5x5_bottleneck_pre_relu/conv&quot;\n  input: &quot;mixed4d/5x5_bottleneck_b&quot;\n  device: &quot;/cpu:0&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n}\nnode {\n  name: &quot;mixed4d/5x5_bottleneck&quot;\n  op: &quot;Relu&quot;\n  input: &quot;mixed4d/5x5_bottleneck_pre_relu&quot;\n  device: &quot;/cpu:0&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n}\nnode {\n  name: &quot;mixed4d/5x5_pre_relu/conv&quot;\n  op: &quot;Conv2D&quot;\n  input: &quot;mixed4d/5x5_bottleneck&quot;\n  input: &quot;mixed4d/5x5_w&quot;\n  device: &quot;/cpu:0&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;padding&quot;\n    value {\n      s: &quot;SAME&quot;\n    }\n  }\n  attr {\n    key: &quot;strides&quot;\n    value {\n      list {\n        i: 1\n        i: 1\n        i: 1\n        i: 1\n      }\n    }\n  }\n  attr {\n    key: &quot;use_cudnn_on_gpu&quot;\n    value {\n      b: true\n    }\n  }\n}\nnode {\n  name: &quot;mixed4d/5x5_pre_relu&quot;\n  op: &quot;BiasAdd&quot;\n  input: &quot;mixed4d/5x5_pre_relu/conv&quot;\n  input: &quot;mixed4d/5x5_b&quot;\n  device: &quot;/cpu:0&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n}\nnode {\n  name: &quot;mixed4d/5x5&quot;\n  op: &quot;Relu&quot;\n  input: &quot;mixed4d/5x5_pre_relu&quot;\n  device: &quot;/cpu:0&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n}\nnode {\n  name: &quot;mixed4d/pool&quot;\n  op: &quot;MaxPool&quot;\n  input: &quot;mixed4c&quot;\n  device: &quot;/cpu:0&quot;\n  attr {\n    key: &quot;ksize&quot;\n    value {\n      list {\n        i: 1\n        i: 3\n        i: 3\n        i: 1\n      }\n    }\n  }\n  attr {\n    key: &quot;padding&quot;\n    value {\n      s: &quot;SAME&quot;\n    }\n  }\n  attr {\n    key: &quot;strides&quot;\n    value {\n      list {\n        i: 1\n        i: 1\n        i: 1\n        i: 1\n      }\n    }\n  }\n}\nnode {\n  name: &quot;mixed4d/pool_reduce_pre_relu/conv&quot;\n  op: &quot;Conv2D&quot;\n  input: &quot;mixed4d/pool&quot;\n  input: &quot;mixed4d/pool_reduce_w&quot;\n  device: &quot;/cpu:0&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;padding&quot;\n    value {\n      s: &quot;SAME&quot;\n    }\n  }\n  attr {\n    key: &quot;strides&quot;\n    value {\n      list {\n        i: 1\n        i: 1\n        i: 1\n        i: 1\n      }\n    }\n  }\n  attr {\n    key: &quot;use_cudnn_on_gpu&quot;\n    value {\n      b: true\n    }\n  }\n}\nnode {\n  name: &quot;mixed4d/pool_reduce_pre_relu&quot;\n  op: &quot;BiasAdd&quot;\n  input: &quot;mixed4d/pool_reduce_pre_relu/conv&quot;\n  input: &quot;mixed4d/pool_reduce_b&quot;\n  device: &quot;/cpu:0&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n}\nnode {\n  name: &quot;mixed4d/pool_reduce&quot;\n  op: &quot;Relu&quot;\n  input: &quot;mixed4d/pool_reduce_pre_relu&quot;\n  device: &quot;/cpu:0&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n}\nnode {\n  name: &quot;mixed4d/concat/dim&quot;\n  op: &quot;Const&quot;\n  device: &quot;/cpu:0&quot;\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_INT32\n    }\n  }\n  attr {\n    key: &quot;value&quot;\n    value {\n      tensor {\n        dtype: DT_INT32\n        tensor_shape {\n        }\n        int_val: 3\n      }\n    }\n  }\n}\nnode {\n  name: &quot;mixed4d&quot;\n  op: &quot;Concat&quot;\n  input: &quot;mixed4d/concat/dim&quot;\n  input: &quot;mixed4d/1x1&quot;\n  input: &quot;mixed4d/3x3&quot;\n  input: &quot;mixed4d/5x5&quot;\n  input: &quot;mixed4d/pool_reduce&quot;\n  device: &quot;/cpu:0&quot;\n  attr {\n    key: &quot;N&quot;\n    value {\n      i: 4\n    }\n  }\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n}\nnode {\n  name: &quot;mixed4e/1x1_pre_relu/conv&quot;\n  op: &quot;Conv2D&quot;\n  input: &quot;mixed4d&quot;\n  input: &quot;mixed4e/1x1_w&quot;\n  device: &quot;/cpu:0&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;padding&quot;\n    value {\n      s: &quot;SAME&quot;\n    }\n  }\n  attr {\n    key: &quot;strides&quot;\n    value {\n      list {\n        i: 1\n        i: 1\n        i: 1\n        i: 1\n      }\n    }\n  }\n  attr {\n    key: &quot;use_cudnn_on_gpu&quot;\n    value {\n      b: true\n    }\n  }\n}\nnode {\n  name: &quot;mixed4e/1x1_pre_relu&quot;\n  op: &quot;BiasAdd&quot;\n  input: &quot;mixed4e/1x1_pre_relu/conv&quot;\n  input: &quot;mixed4e/1x1_b&quot;\n  device: &quot;/cpu:0&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n}\nnode {\n  name: &quot;mixed4e/1x1&quot;\n  op: &quot;Relu&quot;\n  input: &quot;mixed4e/1x1_pre_relu&quot;\n  device: &quot;/cpu:0&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n}\nnode {\n  name: &quot;mixed4e/3x3_bottleneck_pre_relu/conv&quot;\n  op: &quot;Conv2D&quot;\n  input: &quot;mixed4d&quot;\n  input: &quot;mixed4e/3x3_bottleneck_w&quot;\n  device: &quot;/cpu:0&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;padding&quot;\n    value {\n      s: &quot;SAME&quot;\n    }\n  }\n  attr {\n    key: &quot;strides&quot;\n    value {\n      list {\n        i: 1\n        i: 1\n        i: 1\n        i: 1\n      }\n    }\n  }\n  attr {\n    key: &quot;use_cudnn_on_gpu&quot;\n    value {\n      b: true\n    }\n  }\n}\nnode {\n  name: &quot;mixed4e/3x3_bottleneck_pre_relu&quot;\n  op: &quot;BiasAdd&quot;\n  input: &quot;mixed4e/3x3_bottleneck_pre_relu/conv&quot;\n  input: &quot;mixed4e/3x3_bottleneck_b&quot;\n  device: &quot;/cpu:0&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n}\nnode {\n  name: &quot;mixed4e/3x3_bottleneck&quot;\n  op: &quot;Relu&quot;\n  input: &quot;mixed4e/3x3_bottleneck_pre_relu&quot;\n  device: &quot;/cpu:0&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n}\nnode {\n  name: &quot;mixed4e/3x3_pre_relu/conv&quot;\n  op: &quot;Conv2D&quot;\n  input: &quot;mixed4e/3x3_bottleneck&quot;\n  input: &quot;mixed4e/3x3_w&quot;\n  device: &quot;/cpu:0&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;padding&quot;\n    value {\n      s: &quot;SAME&quot;\n    }\n  }\n  attr {\n    key: &quot;strides&quot;\n    value {\n      list {\n        i: 1\n        i: 1\n        i: 1\n        i: 1\n      }\n    }\n  }\n  attr {\n    key: &quot;use_cudnn_on_gpu&quot;\n    value {\n      b: true\n    }\n  }\n}\nnode {\n  name: &quot;mixed4e/3x3_pre_relu&quot;\n  op: &quot;BiasAdd&quot;\n  input: &quot;mixed4e/3x3_pre_relu/conv&quot;\n  input: &quot;mixed4e/3x3_b&quot;\n  device: &quot;/cpu:0&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n}\nnode {\n  name: &quot;mixed4e/3x3&quot;\n  op: &quot;Relu&quot;\n  input: &quot;mixed4e/3x3_pre_relu&quot;\n  device: &quot;/cpu:0&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n}\nnode {\n  name: &quot;mixed4e/5x5_bottleneck_pre_relu/conv&quot;\n  op: &quot;Conv2D&quot;\n  input: &quot;mixed4d&quot;\n  input: &quot;mixed4e/5x5_bottleneck_w&quot;\n  device: &quot;/cpu:0&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;padding&quot;\n    value {\n      s: &quot;SAME&quot;\n    }\n  }\n  attr {\n    key: &quot;strides&quot;\n    value {\n      list {\n        i: 1\n        i: 1\n        i: 1\n        i: 1\n      }\n    }\n  }\n  attr {\n    key: &quot;use_cudnn_on_gpu&quot;\n    value {\n      b: true\n    }\n  }\n}\nnode {\n  name: &quot;mixed4e/5x5_bottleneck_pre_relu&quot;\n  op: &quot;BiasAdd&quot;\n  input: &quot;mixed4e/5x5_bottleneck_pre_relu/conv&quot;\n  input: &quot;mixed4e/5x5_bottleneck_b&quot;\n  device: &quot;/cpu:0&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n}\nnode {\n  name: &quot;mixed4e/5x5_bottleneck&quot;\n  op: &quot;Relu&quot;\n  input: &quot;mixed4e/5x5_bottleneck_pre_relu&quot;\n  device: &quot;/cpu:0&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n}\nnode {\n  name: &quot;mixed4e/5x5_pre_relu/conv&quot;\n  op: &quot;Conv2D&quot;\n  input: &quot;mixed4e/5x5_bottleneck&quot;\n  input: &quot;mixed4e/5x5_w&quot;\n  device: &quot;/cpu:0&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;padding&quot;\n    value {\n      s: &quot;SAME&quot;\n    }\n  }\n  attr {\n    key: &quot;strides&quot;\n    value {\n      list {\n        i: 1\n        i: 1\n        i: 1\n        i: 1\n      }\n    }\n  }\n  attr {\n    key: &quot;use_cudnn_on_gpu&quot;\n    value {\n      b: true\n    }\n  }\n}\nnode {\n  name: &quot;mixed4e/5x5_pre_relu&quot;\n  op: &quot;BiasAdd&quot;\n  input: &quot;mixed4e/5x5_pre_relu/conv&quot;\n  input: &quot;mixed4e/5x5_b&quot;\n  device: &quot;/cpu:0&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n}\nnode {\n  name: &quot;mixed4e/5x5&quot;\n  op: &quot;Relu&quot;\n  input: &quot;mixed4e/5x5_pre_relu&quot;\n  device: &quot;/cpu:0&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n}\nnode {\n  name: &quot;mixed4e/pool&quot;\n  op: &quot;MaxPool&quot;\n  input: &quot;mixed4d&quot;\n  device: &quot;/cpu:0&quot;\n  attr {\n    key: &quot;ksize&quot;\n    value {\n      list {\n        i: 1\n        i: 3\n        i: 3\n        i: 1\n      }\n    }\n  }\n  attr {\n    key: &quot;padding&quot;\n    value {\n      s: &quot;SAME&quot;\n    }\n  }\n  attr {\n    key: &quot;strides&quot;\n    value {\n      list {\n        i: 1\n        i: 1\n        i: 1\n        i: 1\n      }\n    }\n  }\n}\nnode {\n  name: &quot;mixed4e/pool_reduce_pre_relu/conv&quot;\n  op: &quot;Conv2D&quot;\n  input: &quot;mixed4e/pool&quot;\n  input: &quot;mixed4e/pool_reduce_w&quot;\n  device: &quot;/cpu:0&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;padding&quot;\n    value {\n      s: &quot;SAME&quot;\n    }\n  }\n  attr {\n    key: &quot;strides&quot;\n    value {\n      list {\n        i: 1\n        i: 1\n        i: 1\n        i: 1\n      }\n    }\n  }\n  attr {\n    key: &quot;use_cudnn_on_gpu&quot;\n    value {\n      b: true\n    }\n  }\n}\nnode {\n  name: &quot;mixed4e/pool_reduce_pre_relu&quot;\n  op: &quot;BiasAdd&quot;\n  input: &quot;mixed4e/pool_reduce_pre_relu/conv&quot;\n  input: &quot;mixed4e/pool_reduce_b&quot;\n  device: &quot;/cpu:0&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n}\nnode {\n  name: &quot;mixed4e/pool_reduce&quot;\n  op: &quot;Relu&quot;\n  input: &quot;mixed4e/pool_reduce_pre_relu&quot;\n  device: &quot;/cpu:0&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n}\nnode {\n  name: &quot;mixed4e/concat/dim&quot;\n  op: &quot;Const&quot;\n  device: &quot;/cpu:0&quot;\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_INT32\n    }\n  }\n  attr {\n    key: &quot;value&quot;\n    value {\n      tensor {\n        dtype: DT_INT32\n        tensor_shape {\n        }\n        int_val: 3\n      }\n    }\n  }\n}\nnode {\n  name: &quot;mixed4e&quot;\n  op: &quot;Concat&quot;\n  input: &quot;mixed4e/concat/dim&quot;\n  input: &quot;mixed4e/1x1&quot;\n  input: &quot;mixed4e/3x3&quot;\n  input: &quot;mixed4e/5x5&quot;\n  input: &quot;mixed4e/pool_reduce&quot;\n  device: &quot;/cpu:0&quot;\n  attr {\n    key: &quot;N&quot;\n    value {\n      i: 4\n    }\n  }\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n}\nnode {\n  name: &quot;maxpool10&quot;\n  op: &quot;MaxPool&quot;\n  input: &quot;mixed4e&quot;\n  device: &quot;/cpu:0&quot;\n  attr {\n    key: &quot;ksize&quot;\n    value {\n      list {\n        i: 1\n        i: 3\n        i: 3\n        i: 1\n      }\n    }\n  }\n  attr {\n    key: &quot;padding&quot;\n    value {\n      s: &quot;SAME&quot;\n    }\n  }\n  attr {\n    key: &quot;strides&quot;\n    value {\n      list {\n        i: 1\n        i: 2\n        i: 2\n        i: 1\n      }\n    }\n  }\n}\nnode {\n  name: &quot;mixed5a/1x1_pre_relu/conv&quot;\n  op: &quot;Conv2D&quot;\n  input: &quot;maxpool10&quot;\n  input: &quot;mixed5a/1x1_w&quot;\n  device: &quot;/cpu:0&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;padding&quot;\n    value {\n      s: &quot;SAME&quot;\n    }\n  }\n  attr {\n    key: &quot;strides&quot;\n    value {\n      list {\n        i: 1\n        i: 1\n        i: 1\n        i: 1\n      }\n    }\n  }\n  attr {\n    key: &quot;use_cudnn_on_gpu&quot;\n    value {\n      b: true\n    }\n  }\n}\nnode {\n  name: &quot;mixed5a/1x1_pre_relu&quot;\n  op: &quot;BiasAdd&quot;\n  input: &quot;mixed5a/1x1_pre_relu/conv&quot;\n  input: &quot;mixed5a/1x1_b&quot;\n  device: &quot;/cpu:0&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n}\nnode {\n  name: &quot;mixed5a/1x1&quot;\n  op: &quot;Relu&quot;\n  input: &quot;mixed5a/1x1_pre_relu&quot;\n  device: &quot;/cpu:0&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n}\nnode {\n  name: &quot;mixed5a/3x3_bottleneck_pre_relu/conv&quot;\n  op: &quot;Conv2D&quot;\n  input: &quot;maxpool10&quot;\n  input: &quot;mixed5a/3x3_bottleneck_w&quot;\n  device: &quot;/cpu:0&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;padding&quot;\n    value {\n      s: &quot;SAME&quot;\n    }\n  }\n  attr {\n    key: &quot;strides&quot;\n    value {\n      list {\n        i: 1\n        i: 1\n        i: 1\n        i: 1\n      }\n    }\n  }\n  attr {\n    key: &quot;use_cudnn_on_gpu&quot;\n    value {\n      b: true\n    }\n  }\n}\nnode {\n  name: &quot;mixed5a/3x3_bottleneck_pre_relu&quot;\n  op: &quot;BiasAdd&quot;\n  input: &quot;mixed5a/3x3_bottleneck_pre_relu/conv&quot;\n  input: &quot;mixed5a/3x3_bottleneck_b&quot;\n  device: &quot;/cpu:0&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n}\nnode {\n  name: &quot;mixed5a/3x3_bottleneck&quot;\n  op: &quot;Relu&quot;\n  input: &quot;mixed5a/3x3_bottleneck_pre_relu&quot;\n  device: &quot;/cpu:0&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n}\nnode {\n  name: &quot;mixed5a/3x3_pre_relu/conv&quot;\n  op: &quot;Conv2D&quot;\n  input: &quot;mixed5a/3x3_bottleneck&quot;\n  input: &quot;mixed5a/3x3_w&quot;\n  device: &quot;/cpu:0&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;padding&quot;\n    value {\n      s: &quot;SAME&quot;\n    }\n  }\n  attr {\n    key: &quot;strides&quot;\n    value {\n      list {\n        i: 1\n        i: 1\n        i: 1\n        i: 1\n      }\n    }\n  }\n  attr {\n    key: &quot;use_cudnn_on_gpu&quot;\n    value {\n      b: true\n    }\n  }\n}\nnode {\n  name: &quot;mixed5a/3x3_pre_relu&quot;\n  op: &quot;BiasAdd&quot;\n  input: &quot;mixed5a/3x3_pre_relu/conv&quot;\n  input: &quot;mixed5a/3x3_b&quot;\n  device: &quot;/cpu:0&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n}\nnode {\n  name: &quot;mixed5a/3x3&quot;\n  op: &quot;Relu&quot;\n  input: &quot;mixed5a/3x3_pre_relu&quot;\n  device: &quot;/cpu:0&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n}\nnode {\n  name: &quot;mixed5a/5x5_bottleneck_pre_relu/conv&quot;\n  op: &quot;Conv2D&quot;\n  input: &quot;maxpool10&quot;\n  input: &quot;mixed5a/5x5_bottleneck_w&quot;\n  device: &quot;/cpu:0&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;padding&quot;\n    value {\n      s: &quot;SAME&quot;\n    }\n  }\n  attr {\n    key: &quot;strides&quot;\n    value {\n      list {\n        i: 1\n        i: 1\n        i: 1\n        i: 1\n      }\n    }\n  }\n  attr {\n    key: &quot;use_cudnn_on_gpu&quot;\n    value {\n      b: true\n    }\n  }\n}\nnode {\n  name: &quot;mixed5a/5x5_bottleneck_pre_relu&quot;\n  op: &quot;BiasAdd&quot;\n  input: &quot;mixed5a/5x5_bottleneck_pre_relu/conv&quot;\n  input: &quot;mixed5a/5x5_bottleneck_b&quot;\n  device: &quot;/cpu:0&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n}\nnode {\n  name: &quot;mixed5a/5x5_bottleneck&quot;\n  op: &quot;Relu&quot;\n  input: &quot;mixed5a/5x5_bottleneck_pre_relu&quot;\n  device: &quot;/cpu:0&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n}\nnode {\n  name: &quot;mixed5a/5x5_pre_relu/conv&quot;\n  op: &quot;Conv2D&quot;\n  input: &quot;mixed5a/5x5_bottleneck&quot;\n  input: &quot;mixed5a/5x5_w&quot;\n  device: &quot;/cpu:0&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;padding&quot;\n    value {\n      s: &quot;SAME&quot;\n    }\n  }\n  attr {\n    key: &quot;strides&quot;\n    value {\n      list {\n        i: 1\n        i: 1\n        i: 1\n        i: 1\n      }\n    }\n  }\n  attr {\n    key: &quot;use_cudnn_on_gpu&quot;\n    value {\n      b: true\n    }\n  }\n}\nnode {\n  name: &quot;mixed5a/5x5_pre_relu&quot;\n  op: &quot;BiasAdd&quot;\n  input: &quot;mixed5a/5x5_pre_relu/conv&quot;\n  input: &quot;mixed5a/5x5_b&quot;\n  device: &quot;/cpu:0&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n}\nnode {\n  name: &quot;mixed5a/5x5&quot;\n  op: &quot;Relu&quot;\n  input: &quot;mixed5a/5x5_pre_relu&quot;\n  device: &quot;/cpu:0&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n}\nnode {\n  name: &quot;mixed5a/pool&quot;\n  op: &quot;MaxPool&quot;\n  input: &quot;maxpool10&quot;\n  device: &quot;/cpu:0&quot;\n  attr {\n    key: &quot;ksize&quot;\n    value {\n      list {\n        i: 1\n        i: 3\n        i: 3\n        i: 1\n      }\n    }\n  }\n  attr {\n    key: &quot;padding&quot;\n    value {\n      s: &quot;SAME&quot;\n    }\n  }\n  attr {\n    key: &quot;strides&quot;\n    value {\n      list {\n        i: 1\n        i: 1\n        i: 1\n        i: 1\n      }\n    }\n  }\n}\nnode {\n  name: &quot;mixed5a/pool_reduce_pre_relu/conv&quot;\n  op: &quot;Conv2D&quot;\n  input: &quot;mixed5a/pool&quot;\n  input: &quot;mixed5a/pool_reduce_w&quot;\n  device: &quot;/cpu:0&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;padding&quot;\n    value {\n      s: &quot;SAME&quot;\n    }\n  }\n  attr {\n    key: &quot;strides&quot;\n    value {\n      list {\n        i: 1\n        i: 1\n        i: 1\n        i: 1\n      }\n    }\n  }\n  attr {\n    key: &quot;use_cudnn_on_gpu&quot;\n    value {\n      b: true\n    }\n  }\n}\nnode {\n  name: &quot;mixed5a/pool_reduce_pre_relu&quot;\n  op: &quot;BiasAdd&quot;\n  input: &quot;mixed5a/pool_reduce_pre_relu/conv&quot;\n  input: &quot;mixed5a/pool_reduce_b&quot;\n  device: &quot;/cpu:0&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n}\nnode {\n  name: &quot;mixed5a/pool_reduce&quot;\n  op: &quot;Relu&quot;\n  input: &quot;mixed5a/pool_reduce_pre_relu&quot;\n  device: &quot;/cpu:0&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n}\nnode {\n  name: &quot;mixed5a/concat/dim&quot;\n  op: &quot;Const&quot;\n  device: &quot;/cpu:0&quot;\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_INT32\n    }\n  }\n  attr {\n    key: &quot;value&quot;\n    value {\n      tensor {\n        dtype: DT_INT32\n        tensor_shape {\n        }\n        int_val: 3\n      }\n    }\n  }\n}\nnode {\n  name: &quot;mixed5a&quot;\n  op: &quot;Concat&quot;\n  input: &quot;mixed5a/concat/dim&quot;\n  input: &quot;mixed5a/1x1&quot;\n  input: &quot;mixed5a/3x3&quot;\n  input: &quot;mixed5a/5x5&quot;\n  input: &quot;mixed5a/pool_reduce&quot;\n  device: &quot;/cpu:0&quot;\n  attr {\n    key: &quot;N&quot;\n    value {\n      i: 4\n    }\n  }\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n}\nnode {\n  name: &quot;mixed5b/1x1_pre_relu/conv&quot;\n  op: &quot;Conv2D&quot;\n  input: &quot;mixed5a&quot;\n  input: &quot;mixed5b/1x1_w&quot;\n  device: &quot;/cpu:0&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;padding&quot;\n    value {\n      s: &quot;SAME&quot;\n    }\n  }\n  attr {\n    key: &quot;strides&quot;\n    value {\n      list {\n        i: 1\n        i: 1\n        i: 1\n        i: 1\n      }\n    }\n  }\n  attr {\n    key: &quot;use_cudnn_on_gpu&quot;\n    value {\n      b: true\n    }\n  }\n}\nnode {\n  name: &quot;mixed5b/1x1_pre_relu&quot;\n  op: &quot;BiasAdd&quot;\n  input: &quot;mixed5b/1x1_pre_relu/conv&quot;\n  input: &quot;mixed5b/1x1_b&quot;\n  device: &quot;/cpu:0&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n}\nnode {\n  name: &quot;mixed5b/1x1&quot;\n  op: &quot;Relu&quot;\n  input: &quot;mixed5b/1x1_pre_relu&quot;\n  device: &quot;/cpu:0&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n}\nnode {\n  name: &quot;mixed5b/3x3_bottleneck_pre_relu/conv&quot;\n  op: &quot;Conv2D&quot;\n  input: &quot;mixed5a&quot;\n  input: &quot;mixed5b/3x3_bottleneck_w&quot;\n  device: &quot;/cpu:0&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;padding&quot;\n    value {\n      s: &quot;SAME&quot;\n    }\n  }\n  attr {\n    key: &quot;strides&quot;\n    value {\n      list {\n        i: 1\n        i: 1\n        i: 1\n        i: 1\n      }\n    }\n  }\n  attr {\n    key: &quot;use_cudnn_on_gpu&quot;\n    value {\n      b: true\n    }\n  }\n}\nnode {\n  name: &quot;mixed5b/3x3_bottleneck_pre_relu&quot;\n  op: &quot;BiasAdd&quot;\n  input: &quot;mixed5b/3x3_bottleneck_pre_relu/conv&quot;\n  input: &quot;mixed5b/3x3_bottleneck_b&quot;\n  device: &quot;/cpu:0&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n}\nnode {\n  name: &quot;mixed5b/3x3_bottleneck&quot;\n  op: &quot;Relu&quot;\n  input: &quot;mixed5b/3x3_bottleneck_pre_relu&quot;\n  device: &quot;/cpu:0&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n}\nnode {\n  name: &quot;mixed5b/3x3_pre_relu/conv&quot;\n  op: &quot;Conv2D&quot;\n  input: &quot;mixed5b/3x3_bottleneck&quot;\n  input: &quot;mixed5b/3x3_w&quot;\n  device: &quot;/cpu:0&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;padding&quot;\n    value {\n      s: &quot;SAME&quot;\n    }\n  }\n  attr {\n    key: &quot;strides&quot;\n    value {\n      list {\n        i: 1\n        i: 1\n        i: 1\n        i: 1\n      }\n    }\n  }\n  attr {\n    key: &quot;use_cudnn_on_gpu&quot;\n    value {\n      b: true\n    }\n  }\n}\nnode {\n  name: &quot;mixed5b/3x3_pre_relu&quot;\n  op: &quot;BiasAdd&quot;\n  input: &quot;mixed5b/3x3_pre_relu/conv&quot;\n  input: &quot;mixed5b/3x3_b&quot;\n  device: &quot;/cpu:0&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n}\nnode {\n  name: &quot;mixed5b/3x3&quot;\n  op: &quot;Relu&quot;\n  input: &quot;mixed5b/3x3_pre_relu&quot;\n  device: &quot;/cpu:0&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n}\nnode {\n  name: &quot;mixed5b/5x5_bottleneck_pre_relu/conv&quot;\n  op: &quot;Conv2D&quot;\n  input: &quot;mixed5a&quot;\n  input: &quot;mixed5b/5x5_bottleneck_w&quot;\n  device: &quot;/cpu:0&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;padding&quot;\n    value {\n      s: &quot;SAME&quot;\n    }\n  }\n  attr {\n    key: &quot;strides&quot;\n    value {\n      list {\n        i: 1\n        i: 1\n        i: 1\n        i: 1\n      }\n    }\n  }\n  attr {\n    key: &quot;use_cudnn_on_gpu&quot;\n    value {\n      b: true\n    }\n  }\n}\nnode {\n  name: &quot;mixed5b/5x5_bottleneck_pre_relu&quot;\n  op: &quot;BiasAdd&quot;\n  input: &quot;mixed5b/5x5_bottleneck_pre_relu/conv&quot;\n  input: &quot;mixed5b/5x5_bottleneck_b&quot;\n  device: &quot;/cpu:0&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n}\nnode {\n  name: &quot;mixed5b/5x5_bottleneck&quot;\n  op: &quot;Relu&quot;\n  input: &quot;mixed5b/5x5_bottleneck_pre_relu&quot;\n  device: &quot;/cpu:0&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n}\nnode {\n  name: &quot;mixed5b/5x5_pre_relu/conv&quot;\n  op: &quot;Conv2D&quot;\n  input: &quot;mixed5b/5x5_bottleneck&quot;\n  input: &quot;mixed5b/5x5_w&quot;\n  device: &quot;/cpu:0&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;padding&quot;\n    value {\n      s: &quot;SAME&quot;\n    }\n  }\n  attr {\n    key: &quot;strides&quot;\n    value {\n      list {\n        i: 1\n        i: 1\n        i: 1\n        i: 1\n      }\n    }\n  }\n  attr {\n    key: &quot;use_cudnn_on_gpu&quot;\n    value {\n      b: true\n    }\n  }\n}\nnode {\n  name: &quot;mixed5b/5x5_pre_relu&quot;\n  op: &quot;BiasAdd&quot;\n  input: &quot;mixed5b/5x5_pre_relu/conv&quot;\n  input: &quot;mixed5b/5x5_b&quot;\n  device: &quot;/cpu:0&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n}\nnode {\n  name: &quot;mixed5b/5x5&quot;\n  op: &quot;Relu&quot;\n  input: &quot;mixed5b/5x5_pre_relu&quot;\n  device: &quot;/cpu:0&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n}\nnode {\n  name: &quot;mixed5b/pool&quot;\n  op: &quot;MaxPool&quot;\n  input: &quot;mixed5a&quot;\n  device: &quot;/cpu:0&quot;\n  attr {\n    key: &quot;ksize&quot;\n    value {\n      list {\n        i: 1\n        i: 3\n        i: 3\n        i: 1\n      }\n    }\n  }\n  attr {\n    key: &quot;padding&quot;\n    value {\n      s: &quot;SAME&quot;\n    }\n  }\n  attr {\n    key: &quot;strides&quot;\n    value {\n      list {\n        i: 1\n        i: 1\n        i: 1\n        i: 1\n      }\n    }\n  }\n}\nnode {\n  name: &quot;mixed5b/pool_reduce_pre_relu/conv&quot;\n  op: &quot;Conv2D&quot;\n  input: &quot;mixed5b/pool&quot;\n  input: &quot;mixed5b/pool_reduce_w&quot;\n  device: &quot;/cpu:0&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;padding&quot;\n    value {\n      s: &quot;SAME&quot;\n    }\n  }\n  attr {\n    key: &quot;strides&quot;\n    value {\n      list {\n        i: 1\n        i: 1\n        i: 1\n        i: 1\n      }\n    }\n  }\n  attr {\n    key: &quot;use_cudnn_on_gpu&quot;\n    value {\n      b: true\n    }\n  }\n}\nnode {\n  name: &quot;mixed5b/pool_reduce_pre_relu&quot;\n  op: &quot;BiasAdd&quot;\n  input: &quot;mixed5b/pool_reduce_pre_relu/conv&quot;\n  input: &quot;mixed5b/pool_reduce_b&quot;\n  device: &quot;/cpu:0&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n}\nnode {\n  name: &quot;mixed5b/pool_reduce&quot;\n  op: &quot;Relu&quot;\n  input: &quot;mixed5b/pool_reduce_pre_relu&quot;\n  device: &quot;/cpu:0&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n}\nnode {\n  name: &quot;mixed5b/concat/dim&quot;\n  op: &quot;Const&quot;\n  device: &quot;/cpu:0&quot;\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_INT32\n    }\n  }\n  attr {\n    key: &quot;value&quot;\n    value {\n      tensor {\n        dtype: DT_INT32\n        tensor_shape {\n        }\n        int_val: 3\n      }\n    }\n  }\n}\nnode {\n  name: &quot;mixed5b&quot;\n  op: &quot;Concat&quot;\n  input: &quot;mixed5b/concat/dim&quot;\n  input: &quot;mixed5b/1x1&quot;\n  input: &quot;mixed5b/3x3&quot;\n  input: &quot;mixed5b/5x5&quot;\n  input: &quot;mixed5b/pool_reduce&quot;\n  device: &quot;/cpu:0&quot;\n  attr {\n    key: &quot;N&quot;\n    value {\n      i: 4\n    }\n  }\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n}\nnode {\n  name: &quot;avgpool0&quot;\n  op: &quot;AvgPool&quot;\n  input: &quot;mixed5b&quot;\n  device: &quot;/cpu:0&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;ksize&quot;\n    value {\n      list {\n        i: 1\n        i: 7\n        i: 7\n        i: 1\n      }\n    }\n  }\n  attr {\n    key: &quot;padding&quot;\n    value {\n      s: &quot;VALID&quot;\n    }\n  }\n  attr {\n    key: &quot;strides&quot;\n    value {\n      list {\n        i: 1\n        i: 1\n        i: 1\n        i: 1\n      }\n    }\n  }\n}\nnode {\n  name: &quot;head0/pool&quot;\n  op: &quot;AvgPool&quot;\n  input: &quot;mixed4a&quot;\n  device: &quot;/cpu:0&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;ksize&quot;\n    value {\n      list {\n        i: 1\n        i: 5\n        i: 5\n        i: 1\n      }\n    }\n  }\n  attr {\n    key: &quot;padding&quot;\n    value {\n      s: &quot;VALID&quot;\n    }\n  }\n  attr {\n    key: &quot;strides&quot;\n    value {\n      list {\n        i: 1\n        i: 3\n        i: 3\n        i: 1\n      }\n    }\n  }\n}\nnode {\n  name: &quot;head0/bottleneck_pre_relu/conv&quot;\n  op: &quot;Conv2D&quot;\n  input: &quot;head0/pool&quot;\n  input: &quot;head0/bottleneck_w&quot;\n  device: &quot;/cpu:0&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;padding&quot;\n    value {\n      s: &quot;SAME&quot;\n    }\n  }\n  attr {\n    key: &quot;strides&quot;\n    value {\n      list {\n        i: 1\n        i: 1\n        i: 1\n        i: 1\n      }\n    }\n  }\n  attr {\n    key: &quot;use_cudnn_on_gpu&quot;\n    value {\n      b: true\n    }\n  }\n}\nnode {\n  name: &quot;head0/bottleneck_pre_relu&quot;\n  op: &quot;BiasAdd&quot;\n  input: &quot;head0/bottleneck_pre_relu/conv&quot;\n  input: &quot;head0/bottleneck_b&quot;\n  device: &quot;/cpu:0&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n}\nnode {\n  name: &quot;head0/bottleneck&quot;\n  op: &quot;Relu&quot;\n  input: &quot;head0/bottleneck_pre_relu&quot;\n  device: &quot;/cpu:0&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n}\nnode {\n  name: &quot;head0/bottleneck/reshape/shape&quot;\n  op: &quot;Const&quot;\n  device: &quot;/cpu:0&quot;\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_INT32\n    }\n  }\n  attr {\n    key: &quot;value&quot;\n    value {\n      tensor {\n        dtype: DT_INT32\n        tensor_shape {\n          dim {\n            size: 2\n          }\n        }\n        tensor_content: &quot;\\377\\377\\377\\377\\000\\010\\000\\000&quot;\n      }\n    }\n  }\n}\nnode {\n  name: &quot;head0/bottleneck/reshape&quot;\n  op: &quot;Reshape&quot;\n  input: &quot;head0/bottleneck&quot;\n  input: &quot;head0/bottleneck/reshape/shape&quot;\n  device: &quot;/cpu:0&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n}\nnode {\n  name: &quot;nn0/pre_relu/matmul&quot;\n  op: &quot;MatMul&quot;\n  input: &quot;head0/bottleneck/reshape&quot;\n  input: &quot;nn0/w&quot;\n  device: &quot;/cpu:0&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;transpose_a&quot;\n    value {\n      b: false\n    }\n  }\n  attr {\n    key: &quot;transpose_b&quot;\n    value {\n      b: false\n    }\n  }\n}\nnode {\n  name: &quot;nn0/pre_relu&quot;\n  op: &quot;BiasAdd&quot;\n  input: &quot;nn0/pre_relu/matmul&quot;\n  input: &quot;nn0/b&quot;\n  device: &quot;/cpu:0&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n}\nnode {\n  name: &quot;nn0&quot;\n  op: &quot;Relu&quot;\n  input: &quot;nn0/pre_relu&quot;\n  device: &quot;/cpu:0&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n}\nnode {\n  name: &quot;nn0/reshape/shape&quot;\n  op: &quot;Const&quot;\n  device: &quot;/cpu:0&quot;\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_INT32\n    }\n  }\n  attr {\n    key: &quot;value&quot;\n    value {\n      tensor {\n        dtype: DT_INT32\n        tensor_shape {\n          dim {\n            size: 2\n          }\n        }\n        tensor_content: &quot;\\377\\377\\377\\377\\000\\004\\000\\000&quot;\n      }\n    }\n  }\n}\nnode {\n  name: &quot;nn0/reshape&quot;\n  op: &quot;Reshape&quot;\n  input: &quot;nn0&quot;\n  input: &quot;nn0/reshape/shape&quot;\n  device: &quot;/cpu:0&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n}\nnode {\n  name: &quot;softmax0/pre_activation/matmul&quot;\n  op: &quot;MatMul&quot;\n  input: &quot;nn0/reshape&quot;\n  input: &quot;softmax0/w&quot;\n  device: &quot;/cpu:0&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;transpose_a&quot;\n    value {\n      b: false\n    }\n  }\n  attr {\n    key: &quot;transpose_b&quot;\n    value {\n      b: false\n    }\n  }\n}\nnode {\n  name: &quot;softmax0/pre_activation&quot;\n  op: &quot;BiasAdd&quot;\n  input: &quot;softmax0/pre_activation/matmul&quot;\n  input: &quot;softmax0/b&quot;\n  device: &quot;/cpu:0&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n}\nnode {\n  name: &quot;softmax0&quot;\n  op: &quot;Softmax&quot;\n  input: &quot;softmax0/pre_activation&quot;\n  device: &quot;/cpu:0&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n}\nnode {\n  name: &quot;head1/pool&quot;\n  op: &quot;AvgPool&quot;\n  input: &quot;mixed4d&quot;\n  device: &quot;/cpu:0&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;ksize&quot;\n    value {\n      list {\n        i: 1\n        i: 5\n        i: 5\n        i: 1\n      }\n    }\n  }\n  attr {\n    key: &quot;padding&quot;\n    value {\n      s: &quot;VALID&quot;\n    }\n  }\n  attr {\n    key: &quot;strides&quot;\n    value {\n      list {\n        i: 1\n        i: 3\n        i: 3\n        i: 1\n      }\n    }\n  }\n}\nnode {\n  name: &quot;head1/bottleneck_pre_relu/conv&quot;\n  op: &quot;Conv2D&quot;\n  input: &quot;head1/pool&quot;\n  input: &quot;head1/bottleneck_w&quot;\n  device: &quot;/cpu:0&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;padding&quot;\n    value {\n      s: &quot;SAME&quot;\n    }\n  }\n  attr {\n    key: &quot;strides&quot;\n    value {\n      list {\n        i: 1\n        i: 1\n        i: 1\n        i: 1\n      }\n    }\n  }\n  attr {\n    key: &quot;use_cudnn_on_gpu&quot;\n    value {\n      b: true\n    }\n  }\n}\nnode {\n  name: &quot;head1/bottleneck_pre_relu&quot;\n  op: &quot;BiasAdd&quot;\n  input: &quot;head1/bottleneck_pre_relu/conv&quot;\n  input: &quot;head1/bottleneck_b&quot;\n  device: &quot;/cpu:0&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n}\nnode {\n  name: &quot;head1/bottleneck&quot;\n  op: &quot;Relu&quot;\n  input: &quot;head1/bottleneck_pre_relu&quot;\n  device: &quot;/cpu:0&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n}\nnode {\n  name: &quot;head1/bottleneck/reshape/shape&quot;\n  op: &quot;Const&quot;\n  device: &quot;/cpu:0&quot;\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_INT32\n    }\n  }\n  attr {\n    key: &quot;value&quot;\n    value {\n      tensor {\n        dtype: DT_INT32\n        tensor_shape {\n          dim {\n            size: 2\n          }\n        }\n        tensor_content: &quot;\\377\\377\\377\\377\\000\\010\\000\\000&quot;\n      }\n    }\n  }\n}\nnode {\n  name: &quot;head1/bottleneck/reshape&quot;\n  op: &quot;Reshape&quot;\n  input: &quot;head1/bottleneck&quot;\n  input: &quot;head1/bottleneck/reshape/shape&quot;\n  device: &quot;/cpu:0&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n}\nnode {\n  name: &quot;nn1/pre_relu/matmul&quot;\n  op: &quot;MatMul&quot;\n  input: &quot;head1/bottleneck/reshape&quot;\n  input: &quot;nn1/w&quot;\n  device: &quot;/cpu:0&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;transpose_a&quot;\n    value {\n      b: false\n    }\n  }\n  attr {\n    key: &quot;transpose_b&quot;\n    value {\n      b: false\n    }\n  }\n}\nnode {\n  name: &quot;nn1/pre_relu&quot;\n  op: &quot;BiasAdd&quot;\n  input: &quot;nn1/pre_relu/matmul&quot;\n  input: &quot;nn1/b&quot;\n  device: &quot;/cpu:0&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n}\nnode {\n  name: &quot;nn1&quot;\n  op: &quot;Relu&quot;\n  input: &quot;nn1/pre_relu&quot;\n  device: &quot;/cpu:0&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n}\nnode {\n  name: &quot;nn1/reshape/shape&quot;\n  op: &quot;Const&quot;\n  device: &quot;/cpu:0&quot;\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_INT32\n    }\n  }\n  attr {\n    key: &quot;value&quot;\n    value {\n      tensor {\n        dtype: DT_INT32\n        tensor_shape {\n          dim {\n            size: 2\n          }\n        }\n        tensor_content: &quot;\\377\\377\\377\\377\\000\\004\\000\\000&quot;\n      }\n    }\n  }\n}\nnode {\n  name: &quot;nn1/reshape&quot;\n  op: &quot;Reshape&quot;\n  input: &quot;nn1&quot;\n  input: &quot;nn1/reshape/shape&quot;\n  device: &quot;/cpu:0&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n}\nnode {\n  name: &quot;softmax1/pre_activation/matmul&quot;\n  op: &quot;MatMul&quot;\n  input: &quot;nn1/reshape&quot;\n  input: &quot;softmax1/w&quot;\n  device: &quot;/cpu:0&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;transpose_a&quot;\n    value {\n      b: false\n    }\n  }\n  attr {\n    key: &quot;transpose_b&quot;\n    value {\n      b: false\n    }\n  }\n}\nnode {\n  name: &quot;softmax1/pre_activation&quot;\n  op: &quot;BiasAdd&quot;\n  input: &quot;softmax1/pre_activation/matmul&quot;\n  input: &quot;softmax1/b&quot;\n  device: &quot;/cpu:0&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n}\nnode {\n  name: &quot;softmax1&quot;\n  op: &quot;Softmax&quot;\n  input: &quot;softmax1/pre_activation&quot;\n  device: &quot;/cpu:0&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n}\nnode {\n  name: &quot;avgpool0/reshape/shape&quot;\n  op: &quot;Const&quot;\n  device: &quot;/cpu:0&quot;\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_INT32\n    }\n  }\n  attr {\n    key: &quot;value&quot;\n    value {\n      tensor {\n        dtype: DT_INT32\n        tensor_shape {\n          dim {\n            size: 2\n          }\n        }\n        tensor_content: &quot;\\377\\377\\377\\377\\000\\004\\000\\000&quot;\n      }\n    }\n  }\n}\nnode {\n  name: &quot;avgpool0/reshape&quot;\n  op: &quot;Reshape&quot;\n  input: &quot;avgpool0&quot;\n  input: &quot;avgpool0/reshape/shape&quot;\n  device: &quot;/cpu:0&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n}\nnode {\n  name: &quot;softmax2/pre_activation/matmul&quot;\n  op: &quot;MatMul&quot;\n  input: &quot;avgpool0/reshape&quot;\n  input: &quot;softmax2/w&quot;\n  device: &quot;/cpu:0&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;transpose_a&quot;\n    value {\n      b: false\n    }\n  }\n  attr {\n    key: &quot;transpose_b&quot;\n    value {\n      b: false\n    }\n  }\n}\nnode {\n  name: &quot;softmax2/pre_activation&quot;\n  op: &quot;BiasAdd&quot;\n  input: &quot;softmax2/pre_activation/matmul&quot;\n  input: &quot;softmax2/b&quot;\n  device: &quot;/cpu:0&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n}\nnode {\n  name: &quot;softmax2&quot;\n  op: &quot;Softmax&quot;\n  input: &quot;softmax2/pre_activation&quot;\n  device: &quot;/cpu:0&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n}\nnode {\n  name: &quot;output&quot;\n  op: &quot;Identity&quot;\n  input: &quot;softmax0&quot;\n  device: &quot;/cpu:0&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n}\nnode {\n  name: &quot;output1&quot;\n  op: &quot;Identity&quot;\n  input: &quot;softmax1&quot;\n  device: &quot;/cpu:0&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n}\nnode {\n  name: &quot;output2&quot;\n  op: &quot;Identity&quot;\n  input: &quot;softmax2&quot;\n  device: &quot;/cpu:0&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n}\n';
              }
            </script>
            <link rel=&quot;import&quot; href=&quot;https://tensorboard.appspot.com/tf-graph-basic.build.html&quot; onload=load()>
            <div style=&quot;height:600px&quot;>
              <tf-graph-basic id=&quot;graph0.7352770356060288&quot;></tf-graph-basic>
            </div>
        "></iframe>
        


We'll now get the graph from the storage container, and tell tensorflow to use this as its own graph.  This will add all the computations we need to compute the entire deep net, as well as all of the pre-trained parameters.


```python
tf.import_graph_def(net['graph_def'], name='inception')
```


```python
net['labels']
```




    [(0, 'dummy'),
     (1, 'kit fox'),
     (2, 'English setter'),
     (3, 'Siberian husky'),
     (4, 'Australian terrier'),
     (5, 'English springer'),
     (6, 'grey whale'),
     (7, 'lesser panda'),
     (8, 'Egyptian cat'),
     (9, 'ibex'),
     (10, 'Persian cat'),
     (11, 'cougar'),
     (12, 'gazelle'),
     (13, 'porcupine'),
     (14, 'sea lion'),
     (15, 'malamute'),
     (16, 'badger'),
     (17, 'Great Dane'),
     (18, 'Walker hound'),
     (19, 'Welsh springer spaniel'),
     (20, 'whippet'),
     (21, 'Scottish deerhound'),
     (22, 'killer whale'),
     (23, 'mink'),
     (24, 'African elephant'),
     (25, 'Weimaraner'),
     (26, 'soft-coated wheaten terrier'),
     (27, 'Dandie Dinmont'),
     (28, 'red wolf'),
     (29, 'Old English sheepdog'),
     (30, 'jaguar'),
     (31, 'otterhound'),
     (32, 'bloodhound'),
     (33, 'Airedale'),
     (34, 'hyena'),
     (35, 'meerkat'),
     (36, 'giant schnauzer'),
     (37, 'titi'),
     (38, 'three-toed sloth'),
     (39, 'sorrel'),
     (40, 'black-footed ferret'),
     (41, 'dalmatian'),
     (42, 'black-and-tan coonhound'),
     (43, 'papillon'),
     (44, 'skunk'),
     (45, 'Staffordshire bullterrier'),
     (46, 'Mexican hairless'),
     (47, 'Bouvier des Flandres'),
     (48, 'weasel'),
     (49, 'miniature poodle'),
     (50, 'Cardigan'),
     (51, 'malinois'),
     (52, 'bighorn'),
     (53, 'fox squirrel'),
     (54, 'colobus'),
     (55, 'tiger cat'),
     (56, 'Lhasa'),
     (57, 'impala'),
     (58, 'coyote'),
     (59, 'Yorkshire terrier'),
     (60, 'Newfoundland'),
     (61, 'brown bear'),
     (62, 'red fox'),
     (63, 'Norwegian elkhound'),
     (64, 'Rottweiler'),
     (65, 'hartebeest'),
     (66, 'Saluki'),
     (67, 'grey fox'),
     (68, 'schipperke'),
     (69, 'Pekinese'),
     (70, 'Brabancon griffon'),
     (71, 'West Highland white terrier'),
     (72, 'Sealyham terrier'),
     (73, 'guenon'),
     (74, 'mongoose'),
     (75, 'indri'),
     (76, 'tiger'),
     (77, 'Irish wolfhound'),
     (78, 'wild boar'),
     (79, 'EntleBucher'),
     (80, 'zebra'),
     (81, 'ram'),
     (82, 'French bulldog'),
     (83, 'orangutan'),
     (84, 'basenji'),
     (85, 'leopard'),
     (86, 'Bernese mountain dog'),
     (87, 'Maltese dog'),
     (88, 'Norfolk terrier'),
     (89, 'toy terrier'),
     (90, 'vizsla'),
     (91, 'cairn'),
     (92, 'squirrel monkey'),
     (93, 'groenendael'),
     (94, 'clumber'),
     (95, 'Siamese cat'),
     (96, 'chimpanzee'),
     (97, 'komondor'),
     (98, 'Afghan hound'),
     (99, 'Japanese spaniel'),
     (100, 'proboscis monkey'),
     (101, 'guinea pig'),
     (102, 'white wolf'),
     (103, 'ice bear'),
     (104, 'gorilla'),
     (105, 'borzoi'),
     (106, 'toy poodle'),
     (107, 'Kerry blue terrier'),
     (108, 'ox'),
     (109, 'Scotch terrier'),
     (110, 'Tibetan mastiff'),
     (111, 'spider monkey'),
     (112, 'Doberman'),
     (113, 'Boston bull'),
     (114, 'Greater Swiss Mountain dog'),
     (115, 'Appenzeller'),
     (116, 'Shih-Tzu'),
     (117, 'Irish water spaniel'),
     (118, 'Pomeranian'),
     (119, 'Bedlington terrier'),
     (120, 'warthog'),
     (121, 'Arabian camel'),
     (122, 'siamang'),
     (123, 'miniature schnauzer'),
     (124, 'collie'),
     (125, 'golden retriever'),
     (126, 'Irish terrier'),
     (127, 'affenpinscher'),
     (128, 'Border collie'),
     (129, 'hare'),
     (130, 'boxer'),
     (131, 'silky terrier'),
     (132, 'beagle'),
     (133, 'Leonberg'),
     (134, 'German short-haired pointer'),
     (135, 'patas'),
     (136, 'dhole'),
     (137, 'baboon'),
     (138, 'macaque'),
     (139, 'Chesapeake Bay retriever'),
     (140, 'bull mastiff'),
     (141, 'kuvasz'),
     (142, 'capuchin'),
     (143, 'pug'),
     (144, 'curly-coated retriever'),
     (145, 'Norwich terrier'),
     (146, 'flat-coated retriever'),
     (147, 'hog'),
     (148, 'keeshond'),
     (149, 'Eskimo dog'),
     (150, 'Brittany spaniel'),
     (151, 'standard poodle'),
     (152, 'Lakeland terrier'),
     (153, 'snow leopard'),
     (154, 'Gordon setter'),
     (155, 'dingo'),
     (156, 'standard schnauzer'),
     (157, 'hamster'),
     (158, 'Tibetan terrier'),
     (159, 'Arctic fox'),
     (160, 'wire-haired fox terrier'),
     (161, 'basset'),
     (162, 'water buffalo'),
     (163, 'American black bear'),
     (164, 'Angora'),
     (165, 'bison'),
     (166, 'howler monkey'),
     (167, 'hippopotamus'),
     (168, 'chow'),
     (169, 'giant panda'),
     (170, 'American Staffordshire terrier'),
     (171, 'Shetland sheepdog'),
     (172, 'Great Pyrenees'),
     (173, 'Chihuahua'),
     (174, 'tabby'),
     (175, 'marmoset'),
     (176, 'Labrador retriever'),
     (177, 'Saint Bernard'),
     (178, 'armadillo'),
     (179, 'Samoyed'),
     (180, 'bluetick'),
     (181, 'redbone'),
     (182, 'polecat'),
     (183, 'marmot'),
     (184, 'kelpie'),
     (185, 'gibbon'),
     (186, 'llama'),
     (187, 'miniature pinscher'),
     (188, 'wood rabbit'),
     (189, 'Italian greyhound'),
     (190, 'lion'),
     (191, 'cocker spaniel'),
     (192, 'Irish setter'),
     (193, 'dugong'),
     (194, 'Indian elephant'),
     (195, 'beaver'),
     (196, 'Sussex spaniel'),
     (197, 'Pembroke'),
     (198, 'Blenheim spaniel'),
     (199, 'Madagascar cat'),
     (200, 'Rhodesian ridgeback'),
     (201, 'lynx'),
     (202, 'African hunting dog'),
     (203, 'langur'),
     (204, 'Ibizan hound'),
     (205, 'timber wolf'),
     (206, 'cheetah'),
     (207, 'English foxhound'),
     (208, 'briard'),
     (209, 'sloth bear'),
     (210, 'Border terrier'),
     (211, 'German shepherd'),
     (212, 'otter'),
     (213, 'koala'),
     (214, 'tusker'),
     (215, 'echidna'),
     (216, 'wallaby'),
     (217, 'platypus'),
     (218, 'wombat'),
     (219, 'revolver'),
     (220, 'umbrella'),
     (221, 'schooner'),
     (222, 'soccer ball'),
     (223, 'accordion'),
     (224, 'ant'),
     (225, 'starfish'),
     (226, 'chambered nautilus'),
     (227, 'grand piano'),
     (228, 'laptop'),
     (229, 'strawberry'),
     (230, 'airliner'),
     (231, 'warplane'),
     (232, 'airship'),
     (233, 'balloon'),
     (234, 'space shuttle'),
     (235, 'fireboat'),
     (236, 'gondola'),
     (237, 'speedboat'),
     (238, 'lifeboat'),
     (239, 'canoe'),
     (240, 'yawl'),
     (241, 'catamaran'),
     (242, 'trimaran'),
     (243, 'container ship'),
     (244, 'liner'),
     (245, 'pirate'),
     (246, 'aircraft carrier'),
     (247, 'submarine'),
     (248, 'wreck'),
     (249, 'half track'),
     (250, 'tank'),
     (251, 'missile'),
     (252, 'bobsled'),
     (253, 'dogsled'),
     (254, 'bicycle-built-for-two'),
     (255, 'mountain bike'),
     (256, 'freight car'),
     (257, 'passenger car'),
     (258, 'barrow'),
     (259, 'shopping cart'),
     (260, 'motor scooter'),
     (261, 'forklift'),
     (262, 'electric locomotive'),
     (263, 'steam locomotive'),
     (264, 'amphibian'),
     (265, 'ambulance'),
     (266, 'beach wagon'),
     (267, 'cab'),
     (268, 'convertible'),
     (269, 'jeep'),
     (270, 'limousine'),
     (271, 'minivan'),
     (272, 'Model T'),
     (273, 'racer'),
     (274, 'sports car'),
     (275, 'go-kart'),
     (276, 'golfcart'),
     (277, 'moped'),
     (278, 'snowplow'),
     (279, 'fire engine'),
     (280, 'garbage truck'),
     (281, 'pickup'),
     (282, 'tow truck'),
     (283, 'trailer truck'),
     (284, 'moving van'),
     (285, 'police van'),
     (286, 'recreational vehicle'),
     (287, 'streetcar'),
     (288, 'snowmobile'),
     (289, 'tractor'),
     (290, 'mobile home'),
     (291, 'tricycle'),
     (292, 'unicycle'),
     (293, 'horse cart'),
     (294, 'jinrikisha'),
     (295, 'oxcart'),
     (296, 'bassinet'),
     (297, 'cradle'),
     (298, 'crib'),
     (299, 'four-poster'),
     (300, 'bookcase'),
     (301, 'china cabinet'),
     (302, 'medicine chest'),
     (303, 'chiffonier'),
     (304, 'table lamp'),
     (305, 'file'),
     (306, 'park bench'),
     (307, 'barber chair'),
     (308, 'throne'),
     (309, 'folding chair'),
     (310, 'rocking chair'),
     (311, 'studio couch'),
     (312, 'toilet seat'),
     (313, 'desk'),
     (314, 'pool table'),
     (315, 'dining table'),
     (316, 'entertainment center'),
     (317, 'wardrobe'),
     (318, 'Granny Smith'),
     (319, 'orange'),
     (320, 'lemon'),
     (321, 'fig'),
     (322, 'pineapple'),
     (323, 'banana'),
     (324, 'jackfruit'),
     (325, 'custard apple'),
     (326, 'pomegranate'),
     (327, 'acorn'),
     (328, 'hip'),
     (329, 'ear'),
     (330, 'rapeseed'),
     (331, 'corn'),
     (332, 'buckeye'),
     (333, 'organ'),
     (334, 'upright'),
     (335, 'chime'),
     (336, 'drum'),
     (337, 'gong'),
     (338, 'maraca'),
     (339, 'marimba'),
     (340, 'steel drum'),
     (341, 'banjo'),
     (342, 'cello'),
     (343, 'violin'),
     (344, 'harp'),
     (345, 'acoustic guitar'),
     (346, 'electric guitar'),
     (347, 'cornet'),
     (348, 'French horn'),
     (349, 'trombone'),
     (350, 'harmonica'),
     (351, 'ocarina'),
     (352, 'panpipe'),
     (353, 'bassoon'),
     (354, 'oboe'),
     (355, 'sax'),
     (356, 'flute'),
     (357, 'daisy'),
     (358, "yellow lady's slipper"),
     (359, 'cliff'),
     (360, 'valley'),
     (361, 'alp'),
     (362, 'volcano'),
     (363, 'promontory'),
     (364, 'sandbar'),
     (365, 'coral reef'),
     (366, 'lakeside'),
     (367, 'seashore'),
     (368, 'geyser'),
     (369, 'hatchet'),
     (370, 'cleaver'),
     (371, 'letter opener'),
     (372, 'plane'),
     (373, 'power drill'),
     (374, 'lawn mower'),
     (375, 'hammer'),
     (376, 'corkscrew'),
     (377, 'can opener'),
     (378, 'plunger'),
     (379, 'screwdriver'),
     (380, 'shovel'),
     (381, 'plow'),
     (382, 'chain saw'),
     (383, 'cock'),
     (384, 'hen'),
     (385, 'ostrich'),
     (386, 'brambling'),
     (387, 'goldfinch'),
     (388, 'house finch'),
     (389, 'junco'),
     (390, 'indigo bunting'),
     (391, 'robin'),
     (392, 'bulbul'),
     (393, 'jay'),
     (394, 'magpie'),
     (395, 'chickadee'),
     (396, 'water ouzel'),
     (397, 'kite'),
     (398, 'bald eagle'),
     (399, 'vulture'),
     (400, 'great grey owl'),
     (401, 'black grouse'),
     (402, 'ptarmigan'),
     (403, 'ruffed grouse'),
     (404, 'prairie chicken'),
     (405, 'peacock'),
     (406, 'quail'),
     (407, 'partridge'),
     (408, 'African grey'),
     (409, 'macaw'),
     (410, 'sulphur-crested cockatoo'),
     (411, 'lorikeet'),
     (412, 'coucal'),
     (413, 'bee eater'),
     (414, 'hornbill'),
     (415, 'hummingbird'),
     (416, 'jacamar'),
     (417, 'toucan'),
     (418, 'drake'),
     (419, 'red-breasted merganser'),
     (420, 'goose'),
     (421, 'black swan'),
     (422, 'white stork'),
     (423, 'black stork'),
     (424, 'spoonbill'),
     (425, 'flamingo'),
     (426, 'American egret'),
     (427, 'little blue heron'),
     (428, 'bittern'),
     (429, 'crane'),
     (430, 'limpkin'),
     (431, 'American coot'),
     (432, 'bustard'),
     (433, 'ruddy turnstone'),
     (434, 'red-backed sandpiper'),
     (435, 'redshank'),
     (436, 'dowitcher'),
     (437, 'oystercatcher'),
     (438, 'European gallinule'),
     (439, 'pelican'),
     (440, 'king penguin'),
     (441, 'albatross'),
     (442, 'great white shark'),
     (443, 'tiger shark'),
     (444, 'hammerhead'),
     (445, 'electric ray'),
     (446, 'stingray'),
     (447, 'barracouta'),
     (448, 'coho'),
     (449, 'tench'),
     (450, 'goldfish'),
     (451, 'eel'),
     (452, 'rock beauty'),
     (453, 'anemone fish'),
     (454, 'lionfish'),
     (455, 'puffer'),
     (456, 'sturgeon'),
     (457, 'gar'),
     (458, 'loggerhead'),
     (459, 'leatherback turtle'),
     (460, 'mud turtle'),
     (461, 'terrapin'),
     (462, 'box turtle'),
     (463, 'banded gecko'),
     (464, 'common iguana'),
     (465, 'American chameleon'),
     (466, 'whiptail'),
     (467, 'agama'),
     (468, 'frilled lizard'),
     (469, 'alligator lizard'),
     (470, 'Gila monster'),
     (471, 'green lizard'),
     (472, 'African chameleon'),
     (473, 'Komodo dragon'),
     (474, 'triceratops'),
     (475, 'African crocodile'),
     (476, 'American alligator'),
     (477, 'thunder snake'),
     (478, 'ringneck snake'),
     (479, 'hognose snake'),
     (480, 'green snake'),
     (481, 'king snake'),
     (482, 'garter snake'),
     (483, 'water snake'),
     (484, 'vine snake'),
     (485, 'night snake'),
     (486, 'boa constrictor'),
     (487, 'rock python'),
     (488, 'Indian cobra'),
     (489, 'green mamba'),
     (490, 'sea snake'),
     (491, 'horned viper'),
     (492, 'diamondback'),
     (493, 'sidewinder'),
     (494, 'European fire salamander'),
     (495, 'common newt'),
     (496, 'eft'),
     (497, 'spotted salamander'),
     (498, 'axolotl'),
     (499, 'bullfrog'),
     (500, 'tree frog'),
     (501, 'tailed frog'),
     (502, 'whistle'),
     (503, 'wing'),
     (504, 'paintbrush'),
     (505, 'hand blower'),
     (506, 'oxygen mask'),
     (507, 'snorkel'),
     (508, 'loudspeaker'),
     (509, 'microphone'),
     (510, 'screen'),
     (511, 'mouse'),
     (512, 'electric fan'),
     (513, 'oil filter'),
     (514, 'strainer'),
     (515, 'space heater'),
     (516, 'stove'),
     (517, 'guillotine'),
     (518, 'barometer'),
     (519, 'rule'),
     (520, 'odometer'),
     (521, 'scale'),
     (522, 'analog clock'),
     (523, 'digital clock'),
     (524, 'wall clock'),
     (525, 'hourglass'),
     (526, 'sundial'),
     (527, 'parking meter'),
     (528, 'stopwatch'),
     (529, 'digital watch'),
     (530, 'stethoscope'),
     (531, 'syringe'),
     (532, 'magnetic compass'),
     (533, 'binoculars'),
     (534, 'projector'),
     (535, 'sunglasses'),
     (536, 'loupe'),
     (537, 'radio telescope'),
     (538, 'bow'),
     (539, 'cannon [ground]'),
     (540, 'assault rifle'),
     (541, 'rifle'),
     (542, 'projectile'),
     (543, 'computer keyboard'),
     (544, 'typewriter keyboard'),
     (545, 'crane'),
     (546, 'lighter'),
     (547, 'abacus'),
     (548, 'cash machine'),
     (549, 'slide rule'),
     (550, 'desktop computer'),
     (551, 'hand-held computer'),
     (552, 'notebook'),
     (553, 'web site'),
     (554, 'harvester'),
     (555, 'thresher'),
     (556, 'printer'),
     (557, 'slot'),
     (558, 'vending machine'),
     (559, 'sewing machine'),
     (560, 'joystick'),
     (561, 'switch'),
     (562, 'hook'),
     (563, 'car wheel'),
     (564, 'paddlewheel'),
     (565, 'pinwheel'),
     (566, "potter's wheel"),
     (567, 'gas pump'),
     (568, 'carousel'),
     (569, 'swing'),
     (570, 'reel'),
     (571, 'radiator'),
     (572, 'puck'),
     (573, 'hard disc'),
     (574, 'sunglass'),
     (575, 'pick'),
     (576, 'car mirror'),
     (577, 'solar dish'),
     (578, 'remote control'),
     (579, 'disk brake'),
     (580, 'buckle'),
     (581, 'hair slide'),
     (582, 'knot'),
     (583, 'combination lock'),
     (584, 'padlock'),
     (585, 'nail'),
     (586, 'safety pin'),
     (587, 'screw'),
     (588, 'muzzle'),
     (589, 'seat belt'),
     (590, 'ski'),
     (591, 'candle'),
     (592, "jack-o'-lantern"),
     (593, 'spotlight'),
     (594, 'torch'),
     (595, 'neck brace'),
     (596, 'pier'),
     (597, 'tripod'),
     (598, 'maypole'),
     (599, 'mousetrap'),
     (600, 'spider web'),
     (601, 'trilobite'),
     (602, 'harvestman'),
     (603, 'scorpion'),
     (604, 'black and gold garden spider'),
     (605, 'barn spider'),
     (606, 'garden spider'),
     (607, 'black widow'),
     (608, 'tarantula'),
     (609, 'wolf spider'),
     (610, 'tick'),
     (611, 'centipede'),
     (612, 'isopod'),
     (613, 'Dungeness crab'),
     (614, 'rock crab'),
     (615, 'fiddler crab'),
     (616, 'king crab'),
     (617, 'American lobster'),
     (618, 'spiny lobster'),
     (619, 'crayfish'),
     (620, 'hermit crab'),
     (621, 'tiger beetle'),
     (622, 'ladybug'),
     (623, 'ground beetle'),
     (624, 'long-horned beetle'),
     (625, 'leaf beetle'),
     (626, 'dung beetle'),
     (627, 'rhinoceros beetle'),
     (628, 'weevil'),
     (629, 'fly'),
     (630, 'bee'),
     (631, 'grasshopper'),
     (632, 'cricket'),
     (633, 'walking stick'),
     (634, 'cockroach'),
     (635, 'mantis'),
     (636, 'cicada'),
     (637, 'leafhopper'),
     (638, 'lacewing'),
     (639, 'dragonfly'),
     (640, 'damselfly'),
     (641, 'admiral'),
     (642, 'ringlet'),
     (643, 'monarch'),
     (644, 'cabbage butterfly'),
     (645, 'sulphur butterfly'),
     (646, 'lycaenid'),
     (647, 'jellyfish'),
     (648, 'sea anemone'),
     (649, 'brain coral'),
     (650, 'flatworm'),
     (651, 'nematode'),
     (652, 'conch'),
     (653, 'snail'),
     (654, 'slug'),
     (655, 'sea slug'),
     (656, 'chiton'),
     (657, 'sea urchin'),
     (658, 'sea cucumber'),
     (659, 'iron'),
     (660, 'espresso maker'),
     (661, 'microwave'),
     (662, 'Dutch oven'),
     (663, 'rotisserie'),
     (664, 'toaster'),
     (665, 'waffle iron'),
     (666, 'vacuum'),
     (667, 'dishwasher'),
     (668, 'refrigerator'),
     (669, 'washer'),
     (670, 'Crock Pot'),
     (671, 'frying pan'),
     (672, 'wok'),
     (673, 'caldron'),
     (674, 'coffeepot'),
     (675, 'teapot'),
     (676, 'spatula'),
     (677, 'altar'),
     (678, 'triumphal arch'),
     (679, 'patio'),
     (680, 'steel arch bridge'),
     (681, 'suspension bridge'),
     (682, 'viaduct'),
     (683, 'barn'),
     (684, 'greenhouse'),
     (685, 'palace'),
     (686, 'monastery'),
     (687, 'library'),
     (688, 'apiary'),
     (689, 'boathouse'),
     (690, 'church'),
     (691, 'mosque'),
     (692, 'stupa'),
     (693, 'planetarium'),
     (694, 'restaurant'),
     (695, 'cinema'),
     (696, 'home theater'),
     (697, 'lumbermill'),
     (698, 'coil'),
     (699, 'obelisk'),
     (700, 'totem pole'),
     (701, 'castle'),
     (702, 'prison'),
     (703, 'grocery store'),
     (704, 'bakery'),
     (705, 'barbershop'),
     (706, 'bookshop'),
     (707, 'butcher shop'),
     (708, 'confectionery'),
     (709, 'shoe shop'),
     (710, 'tobacco shop'),
     (711, 'toyshop'),
     (712, 'fountain'),
     (713, 'cliff dwelling'),
     (714, 'yurt'),
     (715, 'dock'),
     (716, 'brass'),
     (717, 'megalith'),
     (718, 'bannister'),
     (719, 'breakwater'),
     (720, 'dam'),
     (721, 'chainlink fence'),
     (722, 'picket fence'),
     (723, 'worm fence'),
     (724, 'stone wall'),
     (725, 'grille'),
     (726, 'sliding door'),
     (727, 'turnstile'),
     (728, 'mountain tent'),
     (729, 'scoreboard'),
     (730, 'honeycomb'),
     (731, 'plate rack'),
     (732, 'pedestal'),
     (733, 'beacon'),
     (734, 'mashed potato'),
     (735, 'bell pepper'),
     (736, 'head cabbage'),
     (737, 'broccoli'),
     (738, 'cauliflower'),
     (739, 'zucchini'),
     (740, 'spaghetti squash'),
     (741, 'acorn squash'),
     (742, 'butternut squash'),
     (743, 'cucumber'),
     (744, 'artichoke'),
     (745, 'cardoon'),
     (746, 'mushroom'),
     (747, 'shower curtain'),
     (748, 'jean'),
     (749, 'carton'),
     (750, 'handkerchief'),
     (751, 'sandal'),
     (752, 'ashcan'),
     (753, 'safe'),
     (754, 'plate'),
     (755, 'necklace'),
     (756, 'croquet ball'),
     (757, 'fur coat'),
     (758, 'thimble'),
     (759, 'pajama'),
     (760, 'running shoe'),
     (761, 'cocktail shaker'),
     (762, 'chest'),
     (763, 'manhole cover'),
     (764, 'modem'),
     (765, 'tub'),
     (766, 'tray'),
     (767, 'balance beam'),
     (768, 'bagel'),
     (769, 'prayer rug'),
     (770, 'kimono'),
     (771, 'hot pot'),
     (772, 'whiskey jug'),
     (773, 'knee pad'),
     (774, 'book jacket'),
     (775, 'spindle'),
     (776, 'ski mask'),
     (777, 'beer bottle'),
     (778, 'crash helmet'),
     (779, 'bottlecap'),
     (780, 'tile roof'),
     (781, 'mask'),
     (782, 'maillot'),
     (783, 'Petri dish'),
     (784, 'football helmet'),
     (785, 'bathing cap'),
     (786, 'teddy bear'),
     (787, 'holster'),
     (788, 'pop bottle'),
     (789, 'photocopier'),
     (790, 'vestment'),
     (791, 'crossword puzzle'),
     (792, 'golf ball'),
     (793, 'trifle'),
     (794, 'suit'),
     (795, 'water tower'),
     (796, 'feather boa'),
     (797, 'cloak'),
     (798, 'red wine'),
     (799, 'drumstick'),
     (800, 'shield'),
     (801, 'Christmas stocking'),
     (802, 'hoopskirt'),
     (803, 'menu'),
     (804, 'stage'),
     (805, 'bonnet'),
     (806, 'meat loaf'),
     (807, 'baseball'),
     (808, 'face powder'),
     (809, 'scabbard'),
     (810, 'sunscreen'),
     (811, 'beer glass'),
     (812, 'hen-of-the-woods'),
     (813, 'guacamole'),
     (814, 'lampshade'),
     (815, 'wool'),
     (816, 'hay'),
     (817, 'bow tie'),
     (818, 'mailbag'),
     (819, 'water jug'),
     (820, 'bucket'),
     (821, 'dishrag'),
     (822, 'soup bowl'),
     (823, 'eggnog'),
     (824, 'mortar'),
     (825, 'trench coat'),
     (826, 'paddle'),
     (827, 'chain'),
     (828, 'swab'),
     (829, 'mixing bowl'),
     (830, 'potpie'),
     (831, 'wine bottle'),
     (832, 'shoji'),
     (833, 'bulletproof vest'),
     (834, 'drilling platform'),
     (835, 'binder'),
     (836, 'cardigan'),
     (837, 'sweatshirt'),
     (838, 'pot'),
     (839, 'birdhouse'),
     (840, 'hamper'),
     (841, 'ping-pong ball'),
     (842, 'pencil box'),
     (843, 'pay-phone'),
     (844, 'consomme'),
     (845, 'apron'),
     (846, 'punching bag'),
     (847, 'backpack'),
     (848, 'groom'),
     (849, 'bearskin'),
     (850, 'pencil sharpener'),
     (851, 'broom'),
     (852, 'mosquito net'),
     (853, 'abaya'),
     (854, 'mortarboard'),
     (855, 'poncho'),
     (856, 'crutch'),
     (857, 'Polaroid camera'),
     (858, 'space bar'),
     (859, 'cup'),
     (860, 'racket'),
     (861, 'traffic light'),
     (862, 'quill'),
     (863, 'radio'),
     (864, 'dough'),
     (865, 'cuirass'),
     (866, 'military uniform'),
     (867, 'lipstick'),
     (868, 'shower cap'),
     (869, 'monitor'),
     (870, 'oscilloscope'),
     (871, 'mitten'),
     (872, 'brassiere'),
     (873, 'French loaf'),
     (874, 'vase'),
     (875, 'milk can'),
     (876, 'rugby ball'),
     (877, 'paper towel'),
     (878, 'earthstar'),
     (879, 'envelope'),
     (880, 'miniskirt'),
     (881, 'cowboy hat'),
     (882, 'trolleybus'),
     (883, 'perfume'),
     (884, 'bathtub'),
     (885, 'hotdog'),
     (886, 'coral fungus'),
     (887, 'bullet train'),
     (888, 'pillow'),
     (889, 'toilet tissue'),
     (890, 'cassette'),
     (891, "carpenter's kit"),
     (892, 'ladle'),
     (893, 'stinkhorn'),
     (894, 'lotion'),
     (895, 'hair spray'),
     (896, 'academic gown'),
     (897, 'dome'),
     (898, 'crate'),
     (899, 'wig'),
     (900, 'burrito'),
     (901, 'pill bottle'),
     (902, 'chain mail'),
     (903, 'theater curtain'),
     (904, 'window shade'),
     (905, 'barrel'),
     (906, 'washbasin'),
     (907, 'ballpoint'),
     (908, 'basketball'),
     (909, 'bath towel'),
     (910, 'cowboy boot'),
     (911, 'gown'),
     (912, 'window screen'),
     (913, 'agaric'),
     (914, 'cellular telephone'),
     (915, 'nipple'),
     (916, 'barbell'),
     (917, 'mailbox'),
     (918, 'lab coat'),
     (919, 'fire screen'),
     (920, 'minibus'),
     (921, 'packet'),
     (922, 'maze'),
     (923, 'pole'),
     (924, 'horizontal bar'),
     (925, 'sombrero'),
     (926, 'pickelhaube'),
     (927, 'rain barrel'),
     (928, 'wallet'),
     (929, 'cassette player'),
     (930, 'comic book'),
     (931, 'piggy bank'),
     (932, 'street sign'),
     (933, 'bell cote'),
     (934, 'fountain pen'),
     (935, 'Windsor tie'),
     (936, 'volleyball'),
     (937, 'overskirt'),
     (938, 'sarong'),
     (939, 'purse'),
     (940, 'bolo tie'),
     (941, 'bib'),
     (942, 'parachute'),
     (943, 'sleeping bag'),
     (944, 'television'),
     (945, 'swimming trunks'),
     (946, 'measuring cup'),
     (947, 'espresso'),
     (948, 'pizza'),
     (949, 'breastplate'),
     (950, 'shopping basket'),
     (951, 'wooden spoon'),
     (952, 'saltshaker'),
     (953, 'chocolate sauce'),
     (954, 'ballplayer'),
     (955, 'goblet'),
     (956, 'gyromitra'),
     (957, 'stretcher'),
     (958, 'water bottle'),
     (959, 'dial telephone'),
     (960, 'soap dispenser'),
     (961, 'jersey'),
     (962, 'school bus'),
     (963, 'jigsaw puzzle'),
     (964, 'plastic bag'),
     (965, 'reflex camera'),
     (966, 'diaper'),
     (967, 'Band Aid'),
     (968, 'ice lolly'),
     (969, 'velvet'),
     (970, 'tennis ball'),
     (971, 'gasmask'),
     (972, 'doormat'),
     (973, 'Loafer'),
     (974, 'ice cream'),
     (975, 'pretzel'),
     (976, 'quilt'),
     (977, 'maillot'),
     (978, 'tape player'),
     (979, 'clog'),
     (980, 'iPod'),
     (981, 'bolete'),
     (982, 'scuba diver'),
     (983, 'pitcher'),
     (984, 'matchstick'),
     (985, 'bikini'),
     (986, 'sock'),
     (987, 'CD player'),
     (988, 'lens cap'),
     (989, 'thatch'),
     (990, 'vault'),
     (991, 'beaker'),
     (992, 'bubble'),
     (993, 'cheeseburger'),
     (994, 'parallel bars'),
     (995, 'flagpole'),
     (996, 'coffee mug'),
     (997, 'rubber eraser'),
     (998, 'stole'),
     (999, 'carbonara'),
     ...]



<TODO: visual of graph>

Let's have a look at the graph:


```python
g = tf.get_default_graph()
names = [op.name for op in g.get_operations()]
print(names)
```

    ['inception/input', 'inception/conv2d0_w', 'inception/conv2d0_b', 'inception/conv2d1_w', 'inception/conv2d1_b', 'inception/conv2d2_w', 'inception/conv2d2_b', 'inception/mixed3a_1x1_w', 'inception/mixed3a_1x1_b', 'inception/mixed3a_3x3_bottleneck_w', 'inception/mixed3a_3x3_bottleneck_b', 'inception/mixed3a_3x3_w', 'inception/mixed3a_3x3_b', 'inception/mixed3a_5x5_bottleneck_w', 'inception/mixed3a_5x5_bottleneck_b', 'inception/mixed3a_5x5_w', 'inception/mixed3a_5x5_b', 'inception/mixed3a_pool_reduce_w', 'inception/mixed3a_pool_reduce_b', 'inception/mixed3b_1x1_w', 'inception/mixed3b_1x1_b', 'inception/mixed3b_3x3_bottleneck_w', 'inception/mixed3b_3x3_bottleneck_b', 'inception/mixed3b_3x3_w', 'inception/mixed3b_3x3_b', 'inception/mixed3b_5x5_bottleneck_w', 'inception/mixed3b_5x5_bottleneck_b', 'inception/mixed3b_5x5_w', 'inception/mixed3b_5x5_b', 'inception/mixed3b_pool_reduce_w', 'inception/mixed3b_pool_reduce_b', 'inception/mixed4a_1x1_w', 'inception/mixed4a_1x1_b', 'inception/mixed4a_3x3_bottleneck_w', 'inception/mixed4a_3x3_bottleneck_b', 'inception/mixed4a_3x3_w', 'inception/mixed4a_3x3_b', 'inception/mixed4a_5x5_bottleneck_w', 'inception/mixed4a_5x5_bottleneck_b', 'inception/mixed4a_5x5_w', 'inception/mixed4a_5x5_b', 'inception/mixed4a_pool_reduce_w', 'inception/mixed4a_pool_reduce_b', 'inception/mixed4b_1x1_w', 'inception/mixed4b_1x1_b', 'inception/mixed4b_3x3_bottleneck_w', 'inception/mixed4b_3x3_bottleneck_b', 'inception/mixed4b_3x3_w', 'inception/mixed4b_3x3_b', 'inception/mixed4b_5x5_bottleneck_w', 'inception/mixed4b_5x5_bottleneck_b', 'inception/mixed4b_5x5_w', 'inception/mixed4b_5x5_b', 'inception/mixed4b_pool_reduce_w', 'inception/mixed4b_pool_reduce_b', 'inception/mixed4c_1x1_w', 'inception/mixed4c_1x1_b', 'inception/mixed4c_3x3_bottleneck_w', 'inception/mixed4c_3x3_bottleneck_b', 'inception/mixed4c_3x3_w', 'inception/mixed4c_3x3_b', 'inception/mixed4c_5x5_bottleneck_w', 'inception/mixed4c_5x5_bottleneck_b', 'inception/mixed4c_5x5_w', 'inception/mixed4c_5x5_b', 'inception/mixed4c_pool_reduce_w', 'inception/mixed4c_pool_reduce_b', 'inception/mixed4d_1x1_w', 'inception/mixed4d_1x1_b', 'inception/mixed4d_3x3_bottleneck_w', 'inception/mixed4d_3x3_bottleneck_b', 'inception/mixed4d_3x3_w', 'inception/mixed4d_3x3_b', 'inception/mixed4d_5x5_bottleneck_w', 'inception/mixed4d_5x5_bottleneck_b', 'inception/mixed4d_5x5_w', 'inception/mixed4d_5x5_b', 'inception/mixed4d_pool_reduce_w', 'inception/mixed4d_pool_reduce_b', 'inception/mixed4e_1x1_w', 'inception/mixed4e_1x1_b', 'inception/mixed4e_3x3_bottleneck_w', 'inception/mixed4e_3x3_bottleneck_b', 'inception/mixed4e_3x3_w', 'inception/mixed4e_3x3_b', 'inception/mixed4e_5x5_bottleneck_w', 'inception/mixed4e_5x5_bottleneck_b', 'inception/mixed4e_5x5_w', 'inception/mixed4e_5x5_b', 'inception/mixed4e_pool_reduce_w', 'inception/mixed4e_pool_reduce_b', 'inception/mixed5a_1x1_w', 'inception/mixed5a_1x1_b', 'inception/mixed5a_3x3_bottleneck_w', 'inception/mixed5a_3x3_bottleneck_b', 'inception/mixed5a_3x3_w', 'inception/mixed5a_3x3_b', 'inception/mixed5a_5x5_bottleneck_w', 'inception/mixed5a_5x5_bottleneck_b', 'inception/mixed5a_5x5_w', 'inception/mixed5a_5x5_b', 'inception/mixed5a_pool_reduce_w', 'inception/mixed5a_pool_reduce_b', 'inception/mixed5b_1x1_w', 'inception/mixed5b_1x1_b', 'inception/mixed5b_3x3_bottleneck_w', 'inception/mixed5b_3x3_bottleneck_b', 'inception/mixed5b_3x3_w', 'inception/mixed5b_3x3_b', 'inception/mixed5b_5x5_bottleneck_w', 'inception/mixed5b_5x5_bottleneck_b', 'inception/mixed5b_5x5_w', 'inception/mixed5b_5x5_b', 'inception/mixed5b_pool_reduce_w', 'inception/mixed5b_pool_reduce_b', 'inception/head0_bottleneck_w', 'inception/head0_bottleneck_b', 'inception/nn0_w', 'inception/nn0_b', 'inception/softmax0_w', 'inception/softmax0_b', 'inception/head1_bottleneck_w', 'inception/head1_bottleneck_b', 'inception/nn1_w', 'inception/nn1_b', 'inception/softmax1_w', 'inception/softmax1_b', 'inception/softmax2_w', 'inception/softmax2_b', 'inception/conv2d0_pre_relu/conv', 'inception/conv2d0_pre_relu', 'inception/conv2d0', 'inception/maxpool0', 'inception/localresponsenorm0', 'inception/conv2d1_pre_relu/conv', 'inception/conv2d1_pre_relu', 'inception/conv2d1', 'inception/conv2d2_pre_relu/conv', 'inception/conv2d2_pre_relu', 'inception/conv2d2', 'inception/localresponsenorm1', 'inception/maxpool1', 'inception/mixed3a_1x1_pre_relu/conv', 'inception/mixed3a_1x1_pre_relu', 'inception/mixed3a_1x1', 'inception/mixed3a_3x3_bottleneck_pre_relu/conv', 'inception/mixed3a_3x3_bottleneck_pre_relu', 'inception/mixed3a_3x3_bottleneck', 'inception/mixed3a_3x3_pre_relu/conv', 'inception/mixed3a_3x3_pre_relu', 'inception/mixed3a_3x3', 'inception/mixed3a_5x5_bottleneck_pre_relu/conv', 'inception/mixed3a_5x5_bottleneck_pre_relu', 'inception/mixed3a_5x5_bottleneck', 'inception/mixed3a_5x5_pre_relu/conv', 'inception/mixed3a_5x5_pre_relu', 'inception/mixed3a_5x5', 'inception/mixed3a_pool', 'inception/mixed3a_pool_reduce_pre_relu/conv', 'inception/mixed3a_pool_reduce_pre_relu', 'inception/mixed3a_pool_reduce', 'inception/mixed3a/concat_dim', 'inception/mixed3a', 'inception/mixed3b_1x1_pre_relu/conv', 'inception/mixed3b_1x1_pre_relu', 'inception/mixed3b_1x1', 'inception/mixed3b_3x3_bottleneck_pre_relu/conv', 'inception/mixed3b_3x3_bottleneck_pre_relu', 'inception/mixed3b_3x3_bottleneck', 'inception/mixed3b_3x3_pre_relu/conv', 'inception/mixed3b_3x3_pre_relu', 'inception/mixed3b_3x3', 'inception/mixed3b_5x5_bottleneck_pre_relu/conv', 'inception/mixed3b_5x5_bottleneck_pre_relu', 'inception/mixed3b_5x5_bottleneck', 'inception/mixed3b_5x5_pre_relu/conv', 'inception/mixed3b_5x5_pre_relu', 'inception/mixed3b_5x5', 'inception/mixed3b_pool', 'inception/mixed3b_pool_reduce_pre_relu/conv', 'inception/mixed3b_pool_reduce_pre_relu', 'inception/mixed3b_pool_reduce', 'inception/mixed3b/concat_dim', 'inception/mixed3b', 'inception/maxpool4', 'inception/mixed4a_1x1_pre_relu/conv', 'inception/mixed4a_1x1_pre_relu', 'inception/mixed4a_1x1', 'inception/mixed4a_3x3_bottleneck_pre_relu/conv', 'inception/mixed4a_3x3_bottleneck_pre_relu', 'inception/mixed4a_3x3_bottleneck', 'inception/mixed4a_3x3_pre_relu/conv', 'inception/mixed4a_3x3_pre_relu', 'inception/mixed4a_3x3', 'inception/mixed4a_5x5_bottleneck_pre_relu/conv', 'inception/mixed4a_5x5_bottleneck_pre_relu', 'inception/mixed4a_5x5_bottleneck', 'inception/mixed4a_5x5_pre_relu/conv', 'inception/mixed4a_5x5_pre_relu', 'inception/mixed4a_5x5', 'inception/mixed4a_pool', 'inception/mixed4a_pool_reduce_pre_relu/conv', 'inception/mixed4a_pool_reduce_pre_relu', 'inception/mixed4a_pool_reduce', 'inception/mixed4a/concat_dim', 'inception/mixed4a', 'inception/mixed4b_1x1_pre_relu/conv', 'inception/mixed4b_1x1_pre_relu', 'inception/mixed4b_1x1', 'inception/mixed4b_3x3_bottleneck_pre_relu/conv', 'inception/mixed4b_3x3_bottleneck_pre_relu', 'inception/mixed4b_3x3_bottleneck', 'inception/mixed4b_3x3_pre_relu/conv', 'inception/mixed4b_3x3_pre_relu', 'inception/mixed4b_3x3', 'inception/mixed4b_5x5_bottleneck_pre_relu/conv', 'inception/mixed4b_5x5_bottleneck_pre_relu', 'inception/mixed4b_5x5_bottleneck', 'inception/mixed4b_5x5_pre_relu/conv', 'inception/mixed4b_5x5_pre_relu', 'inception/mixed4b_5x5', 'inception/mixed4b_pool', 'inception/mixed4b_pool_reduce_pre_relu/conv', 'inception/mixed4b_pool_reduce_pre_relu', 'inception/mixed4b_pool_reduce', 'inception/mixed4b/concat_dim', 'inception/mixed4b', 'inception/mixed4c_1x1_pre_relu/conv', 'inception/mixed4c_1x1_pre_relu', 'inception/mixed4c_1x1', 'inception/mixed4c_3x3_bottleneck_pre_relu/conv', 'inception/mixed4c_3x3_bottleneck_pre_relu', 'inception/mixed4c_3x3_bottleneck', 'inception/mixed4c_3x3_pre_relu/conv', 'inception/mixed4c_3x3_pre_relu', 'inception/mixed4c_3x3', 'inception/mixed4c_5x5_bottleneck_pre_relu/conv', 'inception/mixed4c_5x5_bottleneck_pre_relu', 'inception/mixed4c_5x5_bottleneck', 'inception/mixed4c_5x5_pre_relu/conv', 'inception/mixed4c_5x5_pre_relu', 'inception/mixed4c_5x5', 'inception/mixed4c_pool', 'inception/mixed4c_pool_reduce_pre_relu/conv', 'inception/mixed4c_pool_reduce_pre_relu', 'inception/mixed4c_pool_reduce', 'inception/mixed4c/concat_dim', 'inception/mixed4c', 'inception/mixed4d_1x1_pre_relu/conv', 'inception/mixed4d_1x1_pre_relu', 'inception/mixed4d_1x1', 'inception/mixed4d_3x3_bottleneck_pre_relu/conv', 'inception/mixed4d_3x3_bottleneck_pre_relu', 'inception/mixed4d_3x3_bottleneck', 'inception/mixed4d_3x3_pre_relu/conv', 'inception/mixed4d_3x3_pre_relu', 'inception/mixed4d_3x3', 'inception/mixed4d_5x5_bottleneck_pre_relu/conv', 'inception/mixed4d_5x5_bottleneck_pre_relu', 'inception/mixed4d_5x5_bottleneck', 'inception/mixed4d_5x5_pre_relu/conv', 'inception/mixed4d_5x5_pre_relu', 'inception/mixed4d_5x5', 'inception/mixed4d_pool', 'inception/mixed4d_pool_reduce_pre_relu/conv', 'inception/mixed4d_pool_reduce_pre_relu', 'inception/mixed4d_pool_reduce', 'inception/mixed4d/concat_dim', 'inception/mixed4d', 'inception/mixed4e_1x1_pre_relu/conv', 'inception/mixed4e_1x1_pre_relu', 'inception/mixed4e_1x1', 'inception/mixed4e_3x3_bottleneck_pre_relu/conv', 'inception/mixed4e_3x3_bottleneck_pre_relu', 'inception/mixed4e_3x3_bottleneck', 'inception/mixed4e_3x3_pre_relu/conv', 'inception/mixed4e_3x3_pre_relu', 'inception/mixed4e_3x3', 'inception/mixed4e_5x5_bottleneck_pre_relu/conv', 'inception/mixed4e_5x5_bottleneck_pre_relu', 'inception/mixed4e_5x5_bottleneck', 'inception/mixed4e_5x5_pre_relu/conv', 'inception/mixed4e_5x5_pre_relu', 'inception/mixed4e_5x5', 'inception/mixed4e_pool', 'inception/mixed4e_pool_reduce_pre_relu/conv', 'inception/mixed4e_pool_reduce_pre_relu', 'inception/mixed4e_pool_reduce', 'inception/mixed4e/concat_dim', 'inception/mixed4e', 'inception/maxpool10', 'inception/mixed5a_1x1_pre_relu/conv', 'inception/mixed5a_1x1_pre_relu', 'inception/mixed5a_1x1', 'inception/mixed5a_3x3_bottleneck_pre_relu/conv', 'inception/mixed5a_3x3_bottleneck_pre_relu', 'inception/mixed5a_3x3_bottleneck', 'inception/mixed5a_3x3_pre_relu/conv', 'inception/mixed5a_3x3_pre_relu', 'inception/mixed5a_3x3', 'inception/mixed5a_5x5_bottleneck_pre_relu/conv', 'inception/mixed5a_5x5_bottleneck_pre_relu', 'inception/mixed5a_5x5_bottleneck', 'inception/mixed5a_5x5_pre_relu/conv', 'inception/mixed5a_5x5_pre_relu', 'inception/mixed5a_5x5', 'inception/mixed5a_pool', 'inception/mixed5a_pool_reduce_pre_relu/conv', 'inception/mixed5a_pool_reduce_pre_relu', 'inception/mixed5a_pool_reduce', 'inception/mixed5a/concat_dim', 'inception/mixed5a', 'inception/mixed5b_1x1_pre_relu/conv', 'inception/mixed5b_1x1_pre_relu', 'inception/mixed5b_1x1', 'inception/mixed5b_3x3_bottleneck_pre_relu/conv', 'inception/mixed5b_3x3_bottleneck_pre_relu', 'inception/mixed5b_3x3_bottleneck', 'inception/mixed5b_3x3_pre_relu/conv', 'inception/mixed5b_3x3_pre_relu', 'inception/mixed5b_3x3', 'inception/mixed5b_5x5_bottleneck_pre_relu/conv', 'inception/mixed5b_5x5_bottleneck_pre_relu', 'inception/mixed5b_5x5_bottleneck', 'inception/mixed5b_5x5_pre_relu/conv', 'inception/mixed5b_5x5_pre_relu', 'inception/mixed5b_5x5', 'inception/mixed5b_pool', 'inception/mixed5b_pool_reduce_pre_relu/conv', 'inception/mixed5b_pool_reduce_pre_relu', 'inception/mixed5b_pool_reduce', 'inception/mixed5b/concat_dim', 'inception/mixed5b', 'inception/avgpool0', 'inception/head0_pool', 'inception/head0_bottleneck_pre_relu/conv', 'inception/head0_bottleneck_pre_relu', 'inception/head0_bottleneck', 'inception/head0_bottleneck/reshape/shape', 'inception/head0_bottleneck/reshape', 'inception/nn0_pre_relu/matmul', 'inception/nn0_pre_relu', 'inception/nn0', 'inception/nn0/reshape/shape', 'inception/nn0/reshape', 'inception/softmax0_pre_activation/matmul', 'inception/softmax0_pre_activation', 'inception/softmax0', 'inception/head1_pool', 'inception/head1_bottleneck_pre_relu/conv', 'inception/head1_bottleneck_pre_relu', 'inception/head1_bottleneck', 'inception/head1_bottleneck/reshape/shape', 'inception/head1_bottleneck/reshape', 'inception/nn1_pre_relu/matmul', 'inception/nn1_pre_relu', 'inception/nn1', 'inception/nn1/reshape/shape', 'inception/nn1/reshape', 'inception/softmax1_pre_activation/matmul', 'inception/softmax1_pre_activation', 'inception/softmax1', 'inception/avgpool0/reshape/shape', 'inception/avgpool0/reshape', 'inception/softmax2_pre_activation/matmul', 'inception/softmax2_pre_activation', 'inception/softmax2', 'inception/output', 'inception/output1', 'inception/output2']


The input to the graph is stored in the first tensor output, and the probability of the 1000 possible objects is in the last layer:


```python
input_name = names[0] + ':0'
x = g.get_tensor_by_name(input_name)
```


```python
softmax = g.get_tensor_by_name(names[-1] + ':0')
```

<a name="predicting-with-the-inception-network"></a>
## Predicting with the Inception Network

Let's try to use the network to predict now:


```python
from skimage.data import coffee
og = coffee()
plt.imshow(og)
print(og.min(), og.max())
```

    0 255



![png](lecture-4_files/lecture-4_18_1.png)


We'll crop and resize the image to 224 x 224 pixels. I've provided a simple helper function which will do this for us:


```python
# Note that in the lecture, I used a slightly different inception
# model, and this one requires us to subtract the mean from the input image.
# The preprocess function will also crop/resize the image to 299x299
img = inception.preprocess(og)
print(og.shape), print(img.shape)
```

    (400, 600, 3)
    (299, 299, 3)





    (None, None)




```python
# So this will now be a different range than what we had in the lecture:
print(img.min(), img.max())
```

    -117.0 138.0


As we've seen from the last session, our images must be shaped as a 4-dimensional shape describing the number of images, height, width, and number of channels.  So our original 3-dimensional image of height, width, channels needs an additional dimension on the 0th axis.


```python
img_4d = img[np.newaxis]
print(img_4d.shape)
```

    (1, 299, 299, 3)



```python
fig, axs = plt.subplots(1, 2)
axs[0].imshow(og)

# Note that unlike the lecture, we have to call the `inception.deprocess` function
# so that it adds back the mean!
axs[1].imshow(inception.deprocess(img))
```




    <matplotlib.image.AxesImage at 0x13ef9fcc0>




![png](lecture-4_files/lecture-4_24_1.png)



```python
res = np.squeeze(softmax.eval(feed_dict={x: img_4d}))
```


```python
# Note that this network is slightly different than the one used in the lecture.
# Instead of just 1 output, there will be 16 outputs of 1008 probabilities.
# We only use the first 1000 probabilities (the extra ones are for negative/unseen labels)
res.shape
```




    (16, 1008)



The result of the network is a 1000 element vector, with probabilities of each class.  Inside our `net` dictionary are the labels for every element.  We can sort these and use the labels of the 1000 classes to see what the top 5 predicted probabilities and labels are:


```python
# Note that this is one way to aggregate the different probabilities.  We could also
# take the argmax.
res = np.mean(res, 0)
res = res / np.sum(res)
```


```python
print([(res[idx], net['labels'][idx])
       for idx in res.argsort()[-5:][::-1]])
```

    [(0.99849206, (947, 'espresso')), (0.000631253, (859, 'cup')), (0.00050241494, (953, 'chocolate sauce')), (0.00019483207, (844, 'consomme')), (0.00013370356, (822, 'soup bowl'))]


<a name="visualizing-filters"></a>
## Visualizing Filters

Wow so it works!  But how!?  Well that's an ongoing research question.  There has been a lot of great developments in the last few years to help us understand what might be happening.  Let's try to first visualize the weights of the convolution filters, like we've done with our MNIST network before.


```python
W = g.get_tensor_by_name('inception/conv2d0_w:0')
W_eval = W.eval()
print(W_eval.shape)
```

    (7, 7, 3, 64)


With MNIST, our input number of filters was 1, since our input number of channels was also 1, as all of MNIST is grayscale.  But in this case, our input number of channels is 3, and so the input number of convolution filters is also 3.  We can try to see every single individual filter using the library tool I've provided:


```python
from libs import utils
W_montage = utils.montage_filters(W_eval)
plt.figure(figsize=(10,10))
plt.imshow(W_montage, interpolation='nearest')
```




    <matplotlib.image.AxesImage at 0x1379950b8>




![png](lecture-4_files/lecture-4_33_1.png)


Or, we can also try to look at them as RGB filters, showing the influence of each color channel, for each neuron or output filter.


```python
Ws = [utils.montage_filters(W_eval[:, :, [i], :]) for i in range(3)]
Ws = np.rollaxis(np.array(Ws), 0, 3)
plt.figure(figsize=(10,10))
plt.imshow(Ws, interpolation='nearest')
```




    <matplotlib.image.AxesImage at 0x143c37550>




![png](lecture-4_files/lecture-4_35_1.png)


In order to better see what these are doing, let's normalize the filters range:


```python
np.min(Ws), np.max(Ws)
Ws = (Ws / np.max(np.abs(Ws)) * 128 + 128).astype(np.uint8)
plt.figure(figsize=(10,10))
plt.imshow(Ws, interpolation='nearest')
```




    <matplotlib.image.AxesImage at 0x14408d710>




![png](lecture-4_files/lecture-4_37_1.png)


Like with our MNIST example, we can probably guess what some of these are doing.  They are responding to edges, corners, and center-surround or some kind of contrast of two things, like red, green, blue yellow, which interestingly is also what neuroscience of vision tells us about how the human vision identifies color, which is through opponency of red/green and blue/yellow.  To get a better sense, we can try to look at the output of the convolution:


```python
feature = g.get_tensor_by_name('inception/conv2d0_pre_relu:0')
```

Let's look at the shape:


```python
layer_shape = tf.shape(feature).eval(feed_dict={x:img_4d})
print(layer_shape)
```

    [  1 150 150  64]


So our original image which was 1 x 224 x 224 x 3 color channels, now has 64 new channels of information.  The image's height and width are also halved, because of the stride of 2 in the convolution.  We've just seen what each of the convolution filters look like.  Let's try to see how they filter the image now by looking at the resulting convolution.


```python
f = feature.eval(feed_dict={x: img_4d})
montage = utils.montage_filters(np.rollaxis(np.expand_dims(f[0], 3), 3, 2))
fig, axs = plt.subplots(1, 3, figsize=(20, 10))
axs[0].imshow(inception.deprocess(img))
axs[0].set_title('Original Image')
axs[1].imshow(Ws, interpolation='nearest')
axs[1].set_title('Convolution Filters')
axs[2].imshow(montage, cmap='gray')
axs[2].set_title('Convolution Outputs')
```




    <matplotlib.text.Text at 0x1406f3cc0>




![png](lecture-4_files/lecture-4_43_1.png)


It's a little hard to see what's happening here but let's try.  The third filter for instance seems to be a lot like the gabor filter we created in the first session.  It respond to horizontal edges, since it has a bright component at the top, and a dark component on the bottom.  Looking at the output of the convolution, we can see that the horizontal edges really pop out.

<a name="visualizing-the-gradient"></a>
## Visualizing the Gradient

So this is a pretty useful technique for the first convolution layer.  But when we get to the next layer, all of sudden we have 64 different channels of information being fed to more convolution filters of some very high dimensions.  It's very hard to conceptualize that many dimensions, let alone also try and figure out what it could be doing with all the possible combinations it has with other neurons in other layers.

If we want to understand what the deeper layers are really doing, we're going to have to start to use backprop to show us the gradients of a particular neuron with respect to our input image.  Let's visualize the network's gradient activation when backpropagated to the original input image.  This is effectively telling us which pixels are responding to the predicted class or given neuron.

We use a forward pass up to the layer that we are interested in, and then a backprop to help us understand what pixels in particular contributed to the final activation of that layer.  We will need to create an operation which will find the max neuron of all activations in a layer, and then calculate the gradient of that objective with respect to the input image.


```python
feature = g.get_tensor_by_name('inception/conv2d0_pre_relu:0')
gradient = tf.gradients(tf.reduce_max(feature, 3), x)
```

When we run this network now, we will specify the gradient operation we've created, instead of the softmax layer of the network.  This will run a forward prop up to the layer we asked to find the gradient with, and then run a back prop all the way to the input image.


```python
res = sess.run(gradient, feed_dict={x: img_4d})[0]
```

Let's visualize the original image and the output of the backpropagated gradient:


```python
fig, axs = plt.subplots(1, 2)
axs[0].imshow(inception.deprocess(img))
axs[1].imshow(res[0])
```




    <matplotlib.image.AxesImage at 0x146bc2940>




![png](lecture-4_files/lecture-4_49_1.png)


Well that looks like a complete mess!  What we can do is normalize the activations in a way that let's us see it more in terms of the normal range of color values.


```python
def normalize(img, s=0.1):
    '''Normalize the image range for visualization'''
    z = img / np.std(img)
    return np.uint8(np.clip(
        (z - z.mean()) / max(z.std(), 1e-4) * s + 0.5,
        0, 1) * 255)
```


```python
r = normalize(res)
fig, axs = plt.subplots(1, 2)
axs[0].imshow(inception.deprocess(img))
axs[1].imshow(r[0])
```




    <matplotlib.image.AxesImage at 0x14fc042b0>




![png](lecture-4_files/lecture-4_52_1.png)


Much better!  This sort of makes sense!  There are some strong edges and we can really see what colors are changing along those edges.

We can try within individual layers as well, pulling out individual neurons to see what each of them are responding to.  Let's first create a few functions which will help us visualize a single neuron in a layer, and every neuron of a layer:


```python
def compute_gradient(input_placeholder, img, layer_name, neuron_i):
    feature = g.get_tensor_by_name(layer_name)
    gradient = tf.gradients(tf.reduce_mean(feature[:, :, :, neuron_i]), x)
    res = sess.run(gradient, feed_dict={input_placeholder: img})[0]
    return res

def compute_gradients(input_placeholder, img, layer_name):
    feature = g.get_tensor_by_name(layer_name)
    layer_shape = tf.shape(feature).eval(feed_dict={input_placeholder: img})
    gradients = []
    for neuron_i in range(layer_shape[-1]):
        gradients.append(compute_gradient(input_placeholder, img, layer_name, neuron_i))
    return gradients
```

Now we can pass in a layer name, and see the gradient of every neuron in that layer with respect to the input image as a montage.  Let's try the second convolutional layer.  This can take awhile depending on your computer:


```python
gradients = compute_gradients(x, img_4d, 'inception/conv2d1_pre_relu:0')
gradients_norm = [normalize(gradient_i[0]) for gradient_i in gradients]
montage = utils.montage(np.array(gradients_norm))
```


```python
plt.figure(figsize=(12, 12))
plt.imshow(montage)
```




    <matplotlib.image.AxesImage at 0x14139f2e8>




![png](lecture-4_files/lecture-4_57_1.png)


So it's clear that each neuron is responding to some type of feature.  It looks like a lot of them are interested in the texture of the cup, and seem to respond in different ways across the image.  Some seem to be more interested in the shape of the cup, responding pretty strongly to the circular opening, while others seem to catch the liquid in the cup more.  There even seems to be one that just responds to the spoon, and another which responds to only the plate.

Let's try to get a sense of how the activations in each layer progress. We can get every max pooling layer like so:


```python
features = [name for name in names if 'maxpool' in name.split()[-1]]
print(features)
```

    ['inception/maxpool0', 'inception/maxpool1', 'inception/maxpool4', 'inception/maxpool10']


So I didn't mention what max pooling is.  But it 's a simple operation.  You can think of it like a convolution, except instead of using a learned kernel, it will just find the maximum value in the window, for performing "max pooling", or find the average value, for performing "average pooling".

We'll now loop over every feature and create an operation that first will find the maximally activated neuron.  It will then find the sum of all activations across every pixel and input channel of this neuron, and then calculate its gradient with respect to the input image.


```python
n_plots = len(features) + 1
fig, axs = plt.subplots(1, n_plots, figsize=(20, 5))
base = img_4d
axs[0].imshow(inception.deprocess(img))
for feature_i, featurename in enumerate(features):
    feature = g.get_tensor_by_name(featurename + ':0')
    neuron = tf.reduce_max(feature, len(feature.get_shape())-1)
    gradient = tf.gradients(tf.reduce_sum(neuron), x)
    this_res = sess.run(gradient[0], feed_dict={x: base})[0]
    axs[feature_i+1].imshow(normalize(this_res))
    axs[feature_i+1].set_title(featurename)
```


![png](lecture-4_files/lecture-4_61_0.png)


To really understand what's happening in these later layers, we're going to have to experiment with some other visualization techniques.

<a name="deep-dreaming"></a>
# Deep Dreaming

Sometime in May of 2015, A researcher at Google, Alexander Mordvintsev, took a deep network meant to recognize objects in an image, and instead used it to *generate new objects in an image.  The internet quickly exploded after seeing one of the images it produced.  Soon after, Google posted a blog entry on how to perform the technique they re-dubbed "Inceptionism", <TODO: cut to blog and scroll> and tons of interesting outputs were soon created.  Somehow the name Deep Dreaming caught on, and tons of new creative applications came out, from twitter bots (DeepForger), to streaming television (twitch.tv), to apps, it was soon everywhere.

What Deep Dreaming is doing is taking the backpropagated gradient activations and simply adding it back to the image, running the same process again and again in a loop.  I think "dreaming" is a great description of what's going on.  We're really pushing the network in a direction, and seeing what happens when left to its devices.  What it is effectively doing is amplifying whatever our objective is, but we get to see how that objective is optimized in the input space rather than deep in the network in some arbitrarily high dimensional space that no one can understand.

There are many tricks one can add to this idea, such as blurring, adding constraints on the total activations, decaying the gradient, infinitely zooming into the image by cropping and scaling, adding jitter by randomly moving the image around, or plenty of other ideas waiting to be explored.

<a name="simplest-approach"></a>
## Simplest Approach

Let's try the simplest approach for deep dream using a few of these layers.  We're going to try the first max pooling layer to begin with.  We'll specify our objective which is to follow the gradient of the mean of the selected layers's activation.  What we should see is that same objective being amplified so that we can start to understand in terms of the input image what the mean activation of that layer tends to like, or respond to.  We'll also produce a gif of every few frames.  For the remainder of this section, we'll need to rescale our 0-255 range image to 0-1 as it will speed up things:


```python
# Rescale to 0-1 range
img_4d = img_4d / np.max(img_4d)

# Get the max pool layer
layer = g.get_tensor_by_name('inception/maxpool0:0')

# Find the gradient of this layer's mean activation with respect to the input image
gradient = tf.gradients(tf.reduce_mean(layer), x)

# Copy the input image as we'll add the gradient to it in a loop
img_copy = img_4d.copy()

# We'll run it for 50 iterations
n_iterations = 50

# Think of this as our learning rate.  This is how much of the gradient we'll add to the input image
step = 1.0

# Every 10 iterations, we'll add an image to a GIF
gif_step = 10

# Storage for our GIF
imgs = []
for it_i in range(n_iterations):
    print(it_i, end=', ')

    # This will calculate the gradient of the layer we chose with respect to the input image.
    this_res = sess.run(gradient[0], feed_dict={x: img_copy})[0]

    # Let's normalize it by the maximum activation
    this_res /= (np.max(np.abs(this_res)) + 1e-8)

    # Then add it to the input image
    img_copy += this_res * step

    # And add to our gif
    if it_i % gif_step == 0:
        imgs.append(normalize(img_copy[0]))

# Build the gif
gif.build_gif(imgs, saveto='1-simplest-mean-layer.gif')
```

    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 




    <matplotlib.animation.ArtistAnimation at 0x152121128>




![png](lecture-4_files/lecture-4_63_2.png)



```python
ipyd.Image(url='1-simplest-mean-layer.gif', height=200, width=200)
```




<img src="1-simplest-mean-layer.gif" width="200" height="200"/>



What we can see is pretty quickly, the activations tends to pick up the fine detailed edges of the cup, plate, and spoon.  Their structure is very local, meaning they are really describing information at a very small scale.

We could also specify the maximal neuron's mean activation, instead of the mean of the entire layer:


```python
# Find the maximal neuron in a layer
neuron = tf.reduce_max(layer, len(layer.get_shape())-1)
# Then find the mean over this neuron
gradient = tf.gradients(tf.reduce_mean(neuron), x)
```

The rest is exactly the same as before:


```python
img_copy = img_4d.copy()
imgs = []
for it_i in range(n_iterations):
    print(it_i, end=', ')
    this_res = sess.run(gradient[0], feed_dict={x: img_copy})[0]
    this_res /= (np.max(np.abs(this_res)) + 1e-8)
    img_copy += this_res * step
    if it_i % gif_step == 0:
        imgs.append(normalize(img_copy[0]))
gif.build_gif(imgs, saveto='1-simplest-max-neuron.gif')
```

    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 




    <matplotlib.animation.ArtistAnimation at 0x13fdb3dd8>




![png](lecture-4_files/lecture-4_68_2.png)



```python
ipyd.Image(url='1-simplest-max-neuron.gif', height=200, width=200)
```




<img src="1-simplest-max-neuron.gif" width="200" height="200"/>



What we should see here is how the maximal neuron in a layer's activation is slowly maximized through gradient ascent. So over time, we're increasing the overall activation of the neuron we asked for.

Let's try doing this for each of our max pool layers, in increasing depth, and let it run a little longer.  This will take a long time depending on your machine!


```python
# For each max pooling feature, we'll produce a GIF
for feature_i in features:
    layer = g.get_tensor_by_name(feature_i + ':0')
    gradient = tf.gradients(tf.reduce_mean(layer), x)
    img_copy = img_4d.copy()
    imgs = []
    for it_i in range(n_iterations):
        print(it_i, end=', ')
        this_res = sess.run(gradient[0], feed_dict={x: img_copy})[0]
        this_res /= (np.max(np.abs(this_res)) + 1e-8)
        img_copy += this_res * step
        if it_i % gif_step == 0:
            imgs.append(normalize(img_copy[0]))
    gif.build_gif(
        imgs, saveto='1-simplest-' + feature_i.split('/')[-1] + '.gif')
```

    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 


![png](lecture-4_files/lecture-4_71_1.png)



![png](lecture-4_files/lecture-4_71_2.png)



![png](lecture-4_files/lecture-4_71_3.png)



![png](lecture-4_files/lecture-4_71_4.png)


When we look at the outputs of these, we should see the representations in corresponding layers being amplified on the original input image.  As we get to later layers, it really starts to appear to hallucinate, and the patterns start to get more complex.  That's not all though.  The patterns also seem to grow larger.  What that means is that at later layers, the representations span a larger part of the image.  In neuroscience, we might say that this has a larger receptive field, since it is receptive to the content in a wider visual field.

Let's try the same thing except now we'll feed in noise instead of an image:


```python
# Create some noise, centered at gray
img_noise = inception.preprocess(
    (np.random.randint(100, 150, size=(224, 224, 3))))[np.newaxis]
print(img_noise.min(), img_noise.max())
```

    -17.0 31.9579


And the rest is the same:


```python
for feature_i in features:
    layer = g.get_tensor_by_name(feature_i + ':0')
    gradient = tf.gradients(tf.reduce_mean(layer), x)
    img_copy = img_noise.copy()
    imgs = []
    for it_i in range(n_iterations):
        print(it_i, end=', ')
        this_res = sess.run(gradient[0], feed_dict={x: img_copy})[0]
        this_res /= (np.max(np.abs(this_res)) + 1e-8)
        img_copy += this_res * step
        if it_i % gif_step == 0:
            imgs.append(normalize(img_copy[0]))
    gif.build_gif(
        imgs, saveto='1-simplest-noise-' + feature_i.split('/')[-1] + '.gif')
```

    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 


![png](lecture-4_files/lecture-4_75_1.png)



![png](lecture-4_files/lecture-4_75_2.png)



![png](lecture-4_files/lecture-4_75_3.png)



![png](lecture-4_files/lecture-4_75_4.png)


What we should see is that patterns start to emerge, and with higher and higher complexity as we get deeper into the network!

Think back to when we were trying to hand engineer a convolution kernel in the first session.  This should seem pretty amazing to you now.  We've just seen that a network trained in the same way that we trained on MNIST in the last session, just with more layers, and a lot more data, can represent so much detail, and such complex patterns.

<a name="specifying-the-objective"></a>
## Specifying the Objective

Let's summarize a bit what we've done.  What we're doing is let the input image's activation at some later layer or neuron determine what we want to optimize.  We feed an image into the network and see what its activations are for a given neuron or entire layer are by backproping the gradient of that activation back to the input image.  Remember, the gradient is just telling us how things change.  So by following the direction of the gradient, we're going up the gradient, or ascending, and maximizing the selected layer or neuron's activation by changing our input image.

By going up the gradient, we're saying, let's maximize this neuron or layer's activation.  That's different to what we we're doing with gradient descent of a cost function.  Because that was a cost function, we wanted to minimize it, and we were following the negative direction of our gradient.  So the only difference now is we are following the positive direction of our gradient, and performing gradient ascent.

We can also explore specifying a particular gradient activation that we want to maximize.  So rather than simply maximizing its activation, we'll specify what we want the activation to look like, and follow the gradient to get us there.  For instance, let's say we want to only have a particular neuron active, and nothing else.  We can do that by creating an array of 0s the shape of one of our layers, and then filling in 1s for the output of that neuron:


```python
# Let's pick one of the later layers
layer = g.get_tensor_by_name('inception/mixed5b_pool_reduce_pre_relu:0')

# And find its shape
layer_shape = tf.shape(layer).eval(feed_dict={x:img_4d})

# We can find out how many neurons it has by feeding it an image and
# calculating the shape.  The number of output channels is the last dimension.
n_els = tf.shape(layer).eval(feed_dict={x:img_4d})[-1]

# Let's pick a random output channel
neuron_i = np.random.randint(n_els)

# And we'll create an activation of this layer which is entirely 0
layer_vec = np.zeros(layer_shape)

# Except for the randomly chosen neuron which will be full of 1s
layer_vec[..., neuron_i] = 1

# We'll go back to finding the maximal neuron in a layer
neuron = tf.reduce_max(layer, len(layer.get_shape())-1)

# And finding the mean over this neuron
gradient = tf.gradients(tf.reduce_mean(neuron), x)
```

We then feed this into our `feed_dict` parameter and do the same thing as before, ascending the gradient.  We'll try this for a few different neurons to see what they look like.  Again, this will take a long time depending on your computer!


```python
n_iterations = 30
for i in range(5):
    neuron_i = np.random.randint(n_els)
    layer_vec = np.zeros(layer_shape)
    layer_vec[..., neuron_i] = 1
    img_copy = img_noise.copy() / 255.0
    imgs = []
    for it_i in range(n_iterations):
        print(it_i, end=', ')
        this_res = sess.run(gradient[0], feed_dict={
            x: img_copy,
            layer: layer_vec})[0]
        this_res /= (np.max(np.abs(this_res)) + 1e-8)
        img_copy += this_res * step
        if it_i % gif_step == 0:
            imgs.append(normalize(img_copy[0]))
    gif.build_gif(imgs, saveto='2-objective-' + str(neuron_i) + '.gif')
```

    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 


![png](lecture-4_files/lecture-4_79_1.png)



![png](lecture-4_files/lecture-4_79_2.png)



![png](lecture-4_files/lecture-4_79_3.png)



![png](lecture-4_files/lecture-4_79_4.png)



![png](lecture-4_files/lecture-4_79_5.png)


So there is definitely something very interesting happening in each of these neurons.  Even though each image starts off exactly the same, from the same noise image, they each end up in a very different place.  What we're seeing is how each neuron we've chosen seems to be encoding something complex.  They even somehow trigger our perception in a way that says, "Oh, that sort of looks like... maybe a house, or a dog, or a fish, or person...something"

Since our network is trained on objects, we know what each neuron of the last layer should represent.  So we can actually try this with the very last layer, the final layer which we know should represent 1 of 1000 possible objects.  Let's see how to do this.  Let's first find a good neuron:


```python
net['labels']
```




    [(0, 'dummy'),
     (1, 'kit fox'),
     (2, 'English setter'),
     (3, 'Siberian husky'),
     (4, 'Australian terrier'),
     (5, 'English springer'),
     (6, 'grey whale'),
     (7, 'lesser panda'),
     (8, 'Egyptian cat'),
     (9, 'ibex'),
     (10, 'Persian cat'),
     (11, 'cougar'),
     (12, 'gazelle'),
     (13, 'porcupine'),
     (14, 'sea lion'),
     (15, 'malamute'),
     (16, 'badger'),
     (17, 'Great Dane'),
     (18, 'Walker hound'),
     (19, 'Welsh springer spaniel'),
     (20, 'whippet'),
     (21, 'Scottish deerhound'),
     (22, 'killer whale'),
     (23, 'mink'),
     (24, 'African elephant'),
     (25, 'Weimaraner'),
     (26, 'soft-coated wheaten terrier'),
     (27, 'Dandie Dinmont'),
     (28, 'red wolf'),
     (29, 'Old English sheepdog'),
     (30, 'jaguar'),
     (31, 'otterhound'),
     (32, 'bloodhound'),
     (33, 'Airedale'),
     (34, 'hyena'),
     (35, 'meerkat'),
     (36, 'giant schnauzer'),
     (37, 'titi'),
     (38, 'three-toed sloth'),
     (39, 'sorrel'),
     (40, 'black-footed ferret'),
     (41, 'dalmatian'),
     (42, 'black-and-tan coonhound'),
     (43, 'papillon'),
     (44, 'skunk'),
     (45, 'Staffordshire bullterrier'),
     (46, 'Mexican hairless'),
     (47, 'Bouvier des Flandres'),
     (48, 'weasel'),
     (49, 'miniature poodle'),
     (50, 'Cardigan'),
     (51, 'malinois'),
     (52, 'bighorn'),
     (53, 'fox squirrel'),
     (54, 'colobus'),
     (55, 'tiger cat'),
     (56, 'Lhasa'),
     (57, 'impala'),
     (58, 'coyote'),
     (59, 'Yorkshire terrier'),
     (60, 'Newfoundland'),
     (61, 'brown bear'),
     (62, 'red fox'),
     (63, 'Norwegian elkhound'),
     (64, 'Rottweiler'),
     (65, 'hartebeest'),
     (66, 'Saluki'),
     (67, 'grey fox'),
     (68, 'schipperke'),
     (69, 'Pekinese'),
     (70, 'Brabancon griffon'),
     (71, 'West Highland white terrier'),
     (72, 'Sealyham terrier'),
     (73, 'guenon'),
     (74, 'mongoose'),
     (75, 'indri'),
     (76, 'tiger'),
     (77, 'Irish wolfhound'),
     (78, 'wild boar'),
     (79, 'EntleBucher'),
     (80, 'zebra'),
     (81, 'ram'),
     (82, 'French bulldog'),
     (83, 'orangutan'),
     (84, 'basenji'),
     (85, 'leopard'),
     (86, 'Bernese mountain dog'),
     (87, 'Maltese dog'),
     (88, 'Norfolk terrier'),
     (89, 'toy terrier'),
     (90, 'vizsla'),
     (91, 'cairn'),
     (92, 'squirrel monkey'),
     (93, 'groenendael'),
     (94, 'clumber'),
     (95, 'Siamese cat'),
     (96, 'chimpanzee'),
     (97, 'komondor'),
     (98, 'Afghan hound'),
     (99, 'Japanese spaniel'),
     (100, 'proboscis monkey'),
     (101, 'guinea pig'),
     (102, 'white wolf'),
     (103, 'ice bear'),
     (104, 'gorilla'),
     (105, 'borzoi'),
     (106, 'toy poodle'),
     (107, 'Kerry blue terrier'),
     (108, 'ox'),
     (109, 'Scotch terrier'),
     (110, 'Tibetan mastiff'),
     (111, 'spider monkey'),
     (112, 'Doberman'),
     (113, 'Boston bull'),
     (114, 'Greater Swiss Mountain dog'),
     (115, 'Appenzeller'),
     (116, 'Shih-Tzu'),
     (117, 'Irish water spaniel'),
     (118, 'Pomeranian'),
     (119, 'Bedlington terrier'),
     (120, 'warthog'),
     (121, 'Arabian camel'),
     (122, 'siamang'),
     (123, 'miniature schnauzer'),
     (124, 'collie'),
     (125, 'golden retriever'),
     (126, 'Irish terrier'),
     (127, 'affenpinscher'),
     (128, 'Border collie'),
     (129, 'hare'),
     (130, 'boxer'),
     (131, 'silky terrier'),
     (132, 'beagle'),
     (133, 'Leonberg'),
     (134, 'German short-haired pointer'),
     (135, 'patas'),
     (136, 'dhole'),
     (137, 'baboon'),
     (138, 'macaque'),
     (139, 'Chesapeake Bay retriever'),
     (140, 'bull mastiff'),
     (141, 'kuvasz'),
     (142, 'capuchin'),
     (143, 'pug'),
     (144, 'curly-coated retriever'),
     (145, 'Norwich terrier'),
     (146, 'flat-coated retriever'),
     (147, 'hog'),
     (148, 'keeshond'),
     (149, 'Eskimo dog'),
     (150, 'Brittany spaniel'),
     (151, 'standard poodle'),
     (152, 'Lakeland terrier'),
     (153, 'snow leopard'),
     (154, 'Gordon setter'),
     (155, 'dingo'),
     (156, 'standard schnauzer'),
     (157, 'hamster'),
     (158, 'Tibetan terrier'),
     (159, 'Arctic fox'),
     (160, 'wire-haired fox terrier'),
     (161, 'basset'),
     (162, 'water buffalo'),
     (163, 'American black bear'),
     (164, 'Angora'),
     (165, 'bison'),
     (166, 'howler monkey'),
     (167, 'hippopotamus'),
     (168, 'chow'),
     (169, 'giant panda'),
     (170, 'American Staffordshire terrier'),
     (171, 'Shetland sheepdog'),
     (172, 'Great Pyrenees'),
     (173, 'Chihuahua'),
     (174, 'tabby'),
     (175, 'marmoset'),
     (176, 'Labrador retriever'),
     (177, 'Saint Bernard'),
     (178, 'armadillo'),
     (179, 'Samoyed'),
     (180, 'bluetick'),
     (181, 'redbone'),
     (182, 'polecat'),
     (183, 'marmot'),
     (184, 'kelpie'),
     (185, 'gibbon'),
     (186, 'llama'),
     (187, 'miniature pinscher'),
     (188, 'wood rabbit'),
     (189, 'Italian greyhound'),
     (190, 'lion'),
     (191, 'cocker spaniel'),
     (192, 'Irish setter'),
     (193, 'dugong'),
     (194, 'Indian elephant'),
     (195, 'beaver'),
     (196, 'Sussex spaniel'),
     (197, 'Pembroke'),
     (198, 'Blenheim spaniel'),
     (199, 'Madagascar cat'),
     (200, 'Rhodesian ridgeback'),
     (201, 'lynx'),
     (202, 'African hunting dog'),
     (203, 'langur'),
     (204, 'Ibizan hound'),
     (205, 'timber wolf'),
     (206, 'cheetah'),
     (207, 'English foxhound'),
     (208, 'briard'),
     (209, 'sloth bear'),
     (210, 'Border terrier'),
     (211, 'German shepherd'),
     (212, 'otter'),
     (213, 'koala'),
     (214, 'tusker'),
     (215, 'echidna'),
     (216, 'wallaby'),
     (217, 'platypus'),
     (218, 'wombat'),
     (219, 'revolver'),
     (220, 'umbrella'),
     (221, 'schooner'),
     (222, 'soccer ball'),
     (223, 'accordion'),
     (224, 'ant'),
     (225, 'starfish'),
     (226, 'chambered nautilus'),
     (227, 'grand piano'),
     (228, 'laptop'),
     (229, 'strawberry'),
     (230, 'airliner'),
     (231, 'warplane'),
     (232, 'airship'),
     (233, 'balloon'),
     (234, 'space shuttle'),
     (235, 'fireboat'),
     (236, 'gondola'),
     (237, 'speedboat'),
     (238, 'lifeboat'),
     (239, 'canoe'),
     (240, 'yawl'),
     (241, 'catamaran'),
     (242, 'trimaran'),
     (243, 'container ship'),
     (244, 'liner'),
     (245, 'pirate'),
     (246, 'aircraft carrier'),
     (247, 'submarine'),
     (248, 'wreck'),
     (249, 'half track'),
     (250, 'tank'),
     (251, 'missile'),
     (252, 'bobsled'),
     (253, 'dogsled'),
     (254, 'bicycle-built-for-two'),
     (255, 'mountain bike'),
     (256, 'freight car'),
     (257, 'passenger car'),
     (258, 'barrow'),
     (259, 'shopping cart'),
     (260, 'motor scooter'),
     (261, 'forklift'),
     (262, 'electric locomotive'),
     (263, 'steam locomotive'),
     (264, 'amphibian'),
     (265, 'ambulance'),
     (266, 'beach wagon'),
     (267, 'cab'),
     (268, 'convertible'),
     (269, 'jeep'),
     (270, 'limousine'),
     (271, 'minivan'),
     (272, 'Model T'),
     (273, 'racer'),
     (274, 'sports car'),
     (275, 'go-kart'),
     (276, 'golfcart'),
     (277, 'moped'),
     (278, 'snowplow'),
     (279, 'fire engine'),
     (280, 'garbage truck'),
     (281, 'pickup'),
     (282, 'tow truck'),
     (283, 'trailer truck'),
     (284, 'moving van'),
     (285, 'police van'),
     (286, 'recreational vehicle'),
     (287, 'streetcar'),
     (288, 'snowmobile'),
     (289, 'tractor'),
     (290, 'mobile home'),
     (291, 'tricycle'),
     (292, 'unicycle'),
     (293, 'horse cart'),
     (294, 'jinrikisha'),
     (295, 'oxcart'),
     (296, 'bassinet'),
     (297, 'cradle'),
     (298, 'crib'),
     (299, 'four-poster'),
     (300, 'bookcase'),
     (301, 'china cabinet'),
     (302, 'medicine chest'),
     (303, 'chiffonier'),
     (304, 'table lamp'),
     (305, 'file'),
     (306, 'park bench'),
     (307, 'barber chair'),
     (308, 'throne'),
     (309, 'folding chair'),
     (310, 'rocking chair'),
     (311, 'studio couch'),
     (312, 'toilet seat'),
     (313, 'desk'),
     (314, 'pool table'),
     (315, 'dining table'),
     (316, 'entertainment center'),
     (317, 'wardrobe'),
     (318, 'Granny Smith'),
     (319, 'orange'),
     (320, 'lemon'),
     (321, 'fig'),
     (322, 'pineapple'),
     (323, 'banana'),
     (324, 'jackfruit'),
     (325, 'custard apple'),
     (326, 'pomegranate'),
     (327, 'acorn'),
     (328, 'hip'),
     (329, 'ear'),
     (330, 'rapeseed'),
     (331, 'corn'),
     (332, 'buckeye'),
     (333, 'organ'),
     (334, 'upright'),
     (335, 'chime'),
     (336, 'drum'),
     (337, 'gong'),
     (338, 'maraca'),
     (339, 'marimba'),
     (340, 'steel drum'),
     (341, 'banjo'),
     (342, 'cello'),
     (343, 'violin'),
     (344, 'harp'),
     (345, 'acoustic guitar'),
     (346, 'electric guitar'),
     (347, 'cornet'),
     (348, 'French horn'),
     (349, 'trombone'),
     (350, 'harmonica'),
     (351, 'ocarina'),
     (352, 'panpipe'),
     (353, 'bassoon'),
     (354, 'oboe'),
     (355, 'sax'),
     (356, 'flute'),
     (357, 'daisy'),
     (358, "yellow lady's slipper"),
     (359, 'cliff'),
     (360, 'valley'),
     (361, 'alp'),
     (362, 'volcano'),
     (363, 'promontory'),
     (364, 'sandbar'),
     (365, 'coral reef'),
     (366, 'lakeside'),
     (367, 'seashore'),
     (368, 'geyser'),
     (369, 'hatchet'),
     (370, 'cleaver'),
     (371, 'letter opener'),
     (372, 'plane'),
     (373, 'power drill'),
     (374, 'lawn mower'),
     (375, 'hammer'),
     (376, 'corkscrew'),
     (377, 'can opener'),
     (378, 'plunger'),
     (379, 'screwdriver'),
     (380, 'shovel'),
     (381, 'plow'),
     (382, 'chain saw'),
     (383, 'cock'),
     (384, 'hen'),
     (385, 'ostrich'),
     (386, 'brambling'),
     (387, 'goldfinch'),
     (388, 'house finch'),
     (389, 'junco'),
     (390, 'indigo bunting'),
     (391, 'robin'),
     (392, 'bulbul'),
     (393, 'jay'),
     (394, 'magpie'),
     (395, 'chickadee'),
     (396, 'water ouzel'),
     (397, 'kite'),
     (398, 'bald eagle'),
     (399, 'vulture'),
     (400, 'great grey owl'),
     (401, 'black grouse'),
     (402, 'ptarmigan'),
     (403, 'ruffed grouse'),
     (404, 'prairie chicken'),
     (405, 'peacock'),
     (406, 'quail'),
     (407, 'partridge'),
     (408, 'African grey'),
     (409, 'macaw'),
     (410, 'sulphur-crested cockatoo'),
     (411, 'lorikeet'),
     (412, 'coucal'),
     (413, 'bee eater'),
     (414, 'hornbill'),
     (415, 'hummingbird'),
     (416, 'jacamar'),
     (417, 'toucan'),
     (418, 'drake'),
     (419, 'red-breasted merganser'),
     (420, 'goose'),
     (421, 'black swan'),
     (422, 'white stork'),
     (423, 'black stork'),
     (424, 'spoonbill'),
     (425, 'flamingo'),
     (426, 'American egret'),
     (427, 'little blue heron'),
     (428, 'bittern'),
     (429, 'crane'),
     (430, 'limpkin'),
     (431, 'American coot'),
     (432, 'bustard'),
     (433, 'ruddy turnstone'),
     (434, 'red-backed sandpiper'),
     (435, 'redshank'),
     (436, 'dowitcher'),
     (437, 'oystercatcher'),
     (438, 'European gallinule'),
     (439, 'pelican'),
     (440, 'king penguin'),
     (441, 'albatross'),
     (442, 'great white shark'),
     (443, 'tiger shark'),
     (444, 'hammerhead'),
     (445, 'electric ray'),
     (446, 'stingray'),
     (447, 'barracouta'),
     (448, 'coho'),
     (449, 'tench'),
     (450, 'goldfish'),
     (451, 'eel'),
     (452, 'rock beauty'),
     (453, 'anemone fish'),
     (454, 'lionfish'),
     (455, 'puffer'),
     (456, 'sturgeon'),
     (457, 'gar'),
     (458, 'loggerhead'),
     (459, 'leatherback turtle'),
     (460, 'mud turtle'),
     (461, 'terrapin'),
     (462, 'box turtle'),
     (463, 'banded gecko'),
     (464, 'common iguana'),
     (465, 'American chameleon'),
     (466, 'whiptail'),
     (467, 'agama'),
     (468, 'frilled lizard'),
     (469, 'alligator lizard'),
     (470, 'Gila monster'),
     (471, 'green lizard'),
     (472, 'African chameleon'),
     (473, 'Komodo dragon'),
     (474, 'triceratops'),
     (475, 'African crocodile'),
     (476, 'American alligator'),
     (477, 'thunder snake'),
     (478, 'ringneck snake'),
     (479, 'hognose snake'),
     (480, 'green snake'),
     (481, 'king snake'),
     (482, 'garter snake'),
     (483, 'water snake'),
     (484, 'vine snake'),
     (485, 'night snake'),
     (486, 'boa constrictor'),
     (487, 'rock python'),
     (488, 'Indian cobra'),
     (489, 'green mamba'),
     (490, 'sea snake'),
     (491, 'horned viper'),
     (492, 'diamondback'),
     (493, 'sidewinder'),
     (494, 'European fire salamander'),
     (495, 'common newt'),
     (496, 'eft'),
     (497, 'spotted salamander'),
     (498, 'axolotl'),
     (499, 'bullfrog'),
     (500, 'tree frog'),
     (501, 'tailed frog'),
     (502, 'whistle'),
     (503, 'wing'),
     (504, 'paintbrush'),
     (505, 'hand blower'),
     (506, 'oxygen mask'),
     (507, 'snorkel'),
     (508, 'loudspeaker'),
     (509, 'microphone'),
     (510, 'screen'),
     (511, 'mouse'),
     (512, 'electric fan'),
     (513, 'oil filter'),
     (514, 'strainer'),
     (515, 'space heater'),
     (516, 'stove'),
     (517, 'guillotine'),
     (518, 'barometer'),
     (519, 'rule'),
     (520, 'odometer'),
     (521, 'scale'),
     (522, 'analog clock'),
     (523, 'digital clock'),
     (524, 'wall clock'),
     (525, 'hourglass'),
     (526, 'sundial'),
     (527, 'parking meter'),
     (528, 'stopwatch'),
     (529, 'digital watch'),
     (530, 'stethoscope'),
     (531, 'syringe'),
     (532, 'magnetic compass'),
     (533, 'binoculars'),
     (534, 'projector'),
     (535, 'sunglasses'),
     (536, 'loupe'),
     (537, 'radio telescope'),
     (538, 'bow'),
     (539, 'cannon [ground]'),
     (540, 'assault rifle'),
     (541, 'rifle'),
     (542, 'projectile'),
     (543, 'computer keyboard'),
     (544, 'typewriter keyboard'),
     (545, 'crane'),
     (546, 'lighter'),
     (547, 'abacus'),
     (548, 'cash machine'),
     (549, 'slide rule'),
     (550, 'desktop computer'),
     (551, 'hand-held computer'),
     (552, 'notebook'),
     (553, 'web site'),
     (554, 'harvester'),
     (555, 'thresher'),
     (556, 'printer'),
     (557, 'slot'),
     (558, 'vending machine'),
     (559, 'sewing machine'),
     (560, 'joystick'),
     (561, 'switch'),
     (562, 'hook'),
     (563, 'car wheel'),
     (564, 'paddlewheel'),
     (565, 'pinwheel'),
     (566, "potter's wheel"),
     (567, 'gas pump'),
     (568, 'carousel'),
     (569, 'swing'),
     (570, 'reel'),
     (571, 'radiator'),
     (572, 'puck'),
     (573, 'hard disc'),
     (574, 'sunglass'),
     (575, 'pick'),
     (576, 'car mirror'),
     (577, 'solar dish'),
     (578, 'remote control'),
     (579, 'disk brake'),
     (580, 'buckle'),
     (581, 'hair slide'),
     (582, 'knot'),
     (583, 'combination lock'),
     (584, 'padlock'),
     (585, 'nail'),
     (586, 'safety pin'),
     (587, 'screw'),
     (588, 'muzzle'),
     (589, 'seat belt'),
     (590, 'ski'),
     (591, 'candle'),
     (592, "jack-o'-lantern"),
     (593, 'spotlight'),
     (594, 'torch'),
     (595, 'neck brace'),
     (596, 'pier'),
     (597, 'tripod'),
     (598, 'maypole'),
     (599, 'mousetrap'),
     (600, 'spider web'),
     (601, 'trilobite'),
     (602, 'harvestman'),
     (603, 'scorpion'),
     (604, 'black and gold garden spider'),
     (605, 'barn spider'),
     (606, 'garden spider'),
     (607, 'black widow'),
     (608, 'tarantula'),
     (609, 'wolf spider'),
     (610, 'tick'),
     (611, 'centipede'),
     (612, 'isopod'),
     (613, 'Dungeness crab'),
     (614, 'rock crab'),
     (615, 'fiddler crab'),
     (616, 'king crab'),
     (617, 'American lobster'),
     (618, 'spiny lobster'),
     (619, 'crayfish'),
     (620, 'hermit crab'),
     (621, 'tiger beetle'),
     (622, 'ladybug'),
     (623, 'ground beetle'),
     (624, 'long-horned beetle'),
     (625, 'leaf beetle'),
     (626, 'dung beetle'),
     (627, 'rhinoceros beetle'),
     (628, 'weevil'),
     (629, 'fly'),
     (630, 'bee'),
     (631, 'grasshopper'),
     (632, 'cricket'),
     (633, 'walking stick'),
     (634, 'cockroach'),
     (635, 'mantis'),
     (636, 'cicada'),
     (637, 'leafhopper'),
     (638, 'lacewing'),
     (639, 'dragonfly'),
     (640, 'damselfly'),
     (641, 'admiral'),
     (642, 'ringlet'),
     (643, 'monarch'),
     (644, 'cabbage butterfly'),
     (645, 'sulphur butterfly'),
     (646, 'lycaenid'),
     (647, 'jellyfish'),
     (648, 'sea anemone'),
     (649, 'brain coral'),
     (650, 'flatworm'),
     (651, 'nematode'),
     (652, 'conch'),
     (653, 'snail'),
     (654, 'slug'),
     (655, 'sea slug'),
     (656, 'chiton'),
     (657, 'sea urchin'),
     (658, 'sea cucumber'),
     (659, 'iron'),
     (660, 'espresso maker'),
     (661, 'microwave'),
     (662, 'Dutch oven'),
     (663, 'rotisserie'),
     (664, 'toaster'),
     (665, 'waffle iron'),
     (666, 'vacuum'),
     (667, 'dishwasher'),
     (668, 'refrigerator'),
     (669, 'washer'),
     (670, 'Crock Pot'),
     (671, 'frying pan'),
     (672, 'wok'),
     (673, 'caldron'),
     (674, 'coffeepot'),
     (675, 'teapot'),
     (676, 'spatula'),
     (677, 'altar'),
     (678, 'triumphal arch'),
     (679, 'patio'),
     (680, 'steel arch bridge'),
     (681, 'suspension bridge'),
     (682, 'viaduct'),
     (683, 'barn'),
     (684, 'greenhouse'),
     (685, 'palace'),
     (686, 'monastery'),
     (687, 'library'),
     (688, 'apiary'),
     (689, 'boathouse'),
     (690, 'church'),
     (691, 'mosque'),
     (692, 'stupa'),
     (693, 'planetarium'),
     (694, 'restaurant'),
     (695, 'cinema'),
     (696, 'home theater'),
     (697, 'lumbermill'),
     (698, 'coil'),
     (699, 'obelisk'),
     (700, 'totem pole'),
     (701, 'castle'),
     (702, 'prison'),
     (703, 'grocery store'),
     (704, 'bakery'),
     (705, 'barbershop'),
     (706, 'bookshop'),
     (707, 'butcher shop'),
     (708, 'confectionery'),
     (709, 'shoe shop'),
     (710, 'tobacco shop'),
     (711, 'toyshop'),
     (712, 'fountain'),
     (713, 'cliff dwelling'),
     (714, 'yurt'),
     (715, 'dock'),
     (716, 'brass'),
     (717, 'megalith'),
     (718, 'bannister'),
     (719, 'breakwater'),
     (720, 'dam'),
     (721, 'chainlink fence'),
     (722, 'picket fence'),
     (723, 'worm fence'),
     (724, 'stone wall'),
     (725, 'grille'),
     (726, 'sliding door'),
     (727, 'turnstile'),
     (728, 'mountain tent'),
     (729, 'scoreboard'),
     (730, 'honeycomb'),
     (731, 'plate rack'),
     (732, 'pedestal'),
     (733, 'beacon'),
     (734, 'mashed potato'),
     (735, 'bell pepper'),
     (736, 'head cabbage'),
     (737, 'broccoli'),
     (738, 'cauliflower'),
     (739, 'zucchini'),
     (740, 'spaghetti squash'),
     (741, 'acorn squash'),
     (742, 'butternut squash'),
     (743, 'cucumber'),
     (744, 'artichoke'),
     (745, 'cardoon'),
     (746, 'mushroom'),
     (747, 'shower curtain'),
     (748, 'jean'),
     (749, 'carton'),
     (750, 'handkerchief'),
     (751, 'sandal'),
     (752, 'ashcan'),
     (753, 'safe'),
     (754, 'plate'),
     (755, 'necklace'),
     (756, 'croquet ball'),
     (757, 'fur coat'),
     (758, 'thimble'),
     (759, 'pajama'),
     (760, 'running shoe'),
     (761, 'cocktail shaker'),
     (762, 'chest'),
     (763, 'manhole cover'),
     (764, 'modem'),
     (765, 'tub'),
     (766, 'tray'),
     (767, 'balance beam'),
     (768, 'bagel'),
     (769, 'prayer rug'),
     (770, 'kimono'),
     (771, 'hot pot'),
     (772, 'whiskey jug'),
     (773, 'knee pad'),
     (774, 'book jacket'),
     (775, 'spindle'),
     (776, 'ski mask'),
     (777, 'beer bottle'),
     (778, 'crash helmet'),
     (779, 'bottlecap'),
     (780, 'tile roof'),
     (781, 'mask'),
     (782, 'maillot'),
     (783, 'Petri dish'),
     (784, 'football helmet'),
     (785, 'bathing cap'),
     (786, 'teddy bear'),
     (787, 'holster'),
     (788, 'pop bottle'),
     (789, 'photocopier'),
     (790, 'vestment'),
     (791, 'crossword puzzle'),
     (792, 'golf ball'),
     (793, 'trifle'),
     (794, 'suit'),
     (795, 'water tower'),
     (796, 'feather boa'),
     (797, 'cloak'),
     (798, 'red wine'),
     (799, 'drumstick'),
     (800, 'shield'),
     (801, 'Christmas stocking'),
     (802, 'hoopskirt'),
     (803, 'menu'),
     (804, 'stage'),
     (805, 'bonnet'),
     (806, 'meat loaf'),
     (807, 'baseball'),
     (808, 'face powder'),
     (809, 'scabbard'),
     (810, 'sunscreen'),
     (811, 'beer glass'),
     (812, 'hen-of-the-woods'),
     (813, 'guacamole'),
     (814, 'lampshade'),
     (815, 'wool'),
     (816, 'hay'),
     (817, 'bow tie'),
     (818, 'mailbag'),
     (819, 'water jug'),
     (820, 'bucket'),
     (821, 'dishrag'),
     (822, 'soup bowl'),
     (823, 'eggnog'),
     (824, 'mortar'),
     (825, 'trench coat'),
     (826, 'paddle'),
     (827, 'chain'),
     (828, 'swab'),
     (829, 'mixing bowl'),
     (830, 'potpie'),
     (831, 'wine bottle'),
     (832, 'shoji'),
     (833, 'bulletproof vest'),
     (834, 'drilling platform'),
     (835, 'binder'),
     (836, 'cardigan'),
     (837, 'sweatshirt'),
     (838, 'pot'),
     (839, 'birdhouse'),
     (840, 'hamper'),
     (841, 'ping-pong ball'),
     (842, 'pencil box'),
     (843, 'pay-phone'),
     (844, 'consomme'),
     (845, 'apron'),
     (846, 'punching bag'),
     (847, 'backpack'),
     (848, 'groom'),
     (849, 'bearskin'),
     (850, 'pencil sharpener'),
     (851, 'broom'),
     (852, 'mosquito net'),
     (853, 'abaya'),
     (854, 'mortarboard'),
     (855, 'poncho'),
     (856, 'crutch'),
     (857, 'Polaroid camera'),
     (858, 'space bar'),
     (859, 'cup'),
     (860, 'racket'),
     (861, 'traffic light'),
     (862, 'quill'),
     (863, 'radio'),
     (864, 'dough'),
     (865, 'cuirass'),
     (866, 'military uniform'),
     (867, 'lipstick'),
     (868, 'shower cap'),
     (869, 'monitor'),
     (870, 'oscilloscope'),
     (871, 'mitten'),
     (872, 'brassiere'),
     (873, 'French loaf'),
     (874, 'vase'),
     (875, 'milk can'),
     (876, 'rugby ball'),
     (877, 'paper towel'),
     (878, 'earthstar'),
     (879, 'envelope'),
     (880, 'miniskirt'),
     (881, 'cowboy hat'),
     (882, 'trolleybus'),
     (883, 'perfume'),
     (884, 'bathtub'),
     (885, 'hotdog'),
     (886, 'coral fungus'),
     (887, 'bullet train'),
     (888, 'pillow'),
     (889, 'toilet tissue'),
     (890, 'cassette'),
     (891, "carpenter's kit"),
     (892, 'ladle'),
     (893, 'stinkhorn'),
     (894, 'lotion'),
     (895, 'hair spray'),
     (896, 'academic gown'),
     (897, 'dome'),
     (898, 'crate'),
     (899, 'wig'),
     (900, 'burrito'),
     (901, 'pill bottle'),
     (902, 'chain mail'),
     (903, 'theater curtain'),
     (904, 'window shade'),
     (905, 'barrel'),
     (906, 'washbasin'),
     (907, 'ballpoint'),
     (908, 'basketball'),
     (909, 'bath towel'),
     (910, 'cowboy boot'),
     (911, 'gown'),
     (912, 'window screen'),
     (913, 'agaric'),
     (914, 'cellular telephone'),
     (915, 'nipple'),
     (916, 'barbell'),
     (917, 'mailbox'),
     (918, 'lab coat'),
     (919, 'fire screen'),
     (920, 'minibus'),
     (921, 'packet'),
     (922, 'maze'),
     (923, 'pole'),
     (924, 'horizontal bar'),
     (925, 'sombrero'),
     (926, 'pickelhaube'),
     (927, 'rain barrel'),
     (928, 'wallet'),
     (929, 'cassette player'),
     (930, 'comic book'),
     (931, 'piggy bank'),
     (932, 'street sign'),
     (933, 'bell cote'),
     (934, 'fountain pen'),
     (935, 'Windsor tie'),
     (936, 'volleyball'),
     (937, 'overskirt'),
     (938, 'sarong'),
     (939, 'purse'),
     (940, 'bolo tie'),
     (941, 'bib'),
     (942, 'parachute'),
     (943, 'sleeping bag'),
     (944, 'television'),
     (945, 'swimming trunks'),
     (946, 'measuring cup'),
     (947, 'espresso'),
     (948, 'pizza'),
     (949, 'breastplate'),
     (950, 'shopping basket'),
     (951, 'wooden spoon'),
     (952, 'saltshaker'),
     (953, 'chocolate sauce'),
     (954, 'ballplayer'),
     (955, 'goblet'),
     (956, 'gyromitra'),
     (957, 'stretcher'),
     (958, 'water bottle'),
     (959, 'dial telephone'),
     (960, 'soap dispenser'),
     (961, 'jersey'),
     (962, 'school bus'),
     (963, 'jigsaw puzzle'),
     (964, 'plastic bag'),
     (965, 'reflex camera'),
     (966, 'diaper'),
     (967, 'Band Aid'),
     (968, 'ice lolly'),
     (969, 'velvet'),
     (970, 'tennis ball'),
     (971, 'gasmask'),
     (972, 'doormat'),
     (973, 'Loafer'),
     (974, 'ice cream'),
     (975, 'pretzel'),
     (976, 'quilt'),
     (977, 'maillot'),
     (978, 'tape player'),
     (979, 'clog'),
     (980, 'iPod'),
     (981, 'bolete'),
     (982, 'scuba diver'),
     (983, 'pitcher'),
     (984, 'matchstick'),
     (985, 'bikini'),
     (986, 'sock'),
     (987, 'CD player'),
     (988, 'lens cap'),
     (989, 'thatch'),
     (990, 'vault'),
     (991, 'beaker'),
     (992, 'bubble'),
     (993, 'cheeseburger'),
     (994, 'parallel bars'),
     (995, 'flagpole'),
     (996, 'coffee mug'),
     (997, 'rubber eraser'),
     (998, 'stole'),
     (999, 'carbonara'),
     ...]




```python
# let's try a school bus.
neuron_i = 962
print(net['labels'][neuron_i])
```

    (962, 'school bus')



```python
# We'll pick the very last layer
layer = g.get_tensor_by_name(names[-1] + ':0')

# Then find the max activation of this layer
gradient = tf.gradients(tf.reduce_max(layer), x)

# We'll find its shape and create the activation we want to maximize w/ gradient ascent
layer_shape = tf.shape(layer).eval(feed_dict={x: img_noise})
layer_vec = np.zeros(layer_shape)
layer_vec[..., neuron_i] = 1
```

And then train just like before:


```python
n_iterations = 100
gif_step = 10

img_copy = img_noise.copy()
imgs = []
for it_i in range(n_iterations):
    print(it_i, end=', ')
    this_res = sess.run(gradient[0], feed_dict={
        x: img_copy,
        layer: layer_vec})[0]
    this_res /= (np.max(np.abs(this_res)) + 1e-8)
    img_copy += this_res * step
    if it_i % gif_step == 0:
        imgs.append(normalize(img_copy[0]))
gif.build_gif(imgs, saveto='2-object-' + str(neuron_i) + '.gif')
```

    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 




    <matplotlib.animation.ArtistAnimation at 0x1464b54e0>




![png](lecture-4_files/lecture-4_85_2.png)


So what we should see is the noise image become more like patterns that might appear on a school bus.

<a name="decaying-the-gradient"></a>
## Decaying the Gradient

There is a lot we can explore with this process to get a clearer picture.  Some of the more interesting visualizations come about through regularization techniques such as smoothing the activations every so often, or clipping the gradients to a certain range.  We'll see how all of these together can help us get a much cleaner image.  We'll start with decay.  This will slowly reduce the range of values:


```python
decay = 0.95

img_copy = img_noise.copy()
imgs = []
for it_i in range(n_iterations):
    print(it_i, end=', ')
    this_res = sess.run(gradient[0], feed_dict={
        x: img_copy,
        layer: layer_vec})[0]
    this_res /= (np.max(np.abs(this_res)) + 1e-8)
    img_copy += this_res * step
    img_copy *= decay
    if it_i % gif_step == 0:
        imgs.append(normalize(img_copy[0]))
gif.build_gif(imgs, saveto='3-decay-' + str(neuron_i) + '.gif')
```

    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 




    <matplotlib.animation.ArtistAnimation at 0x14d775dd8>




![png](lecture-4_files/lecture-4_87_2.png)


<a name="blurring-the-gradient"></a>
## Blurring the Gradient

Let's now try and see how blurring with a gaussian changes the visualization.


```python
# Let's get a gaussian filter
from scipy.ndimage.filters import gaussian_filter

# Which we'll smooth with a standard deviation of 0.5
sigma = 1.0

# And we'll smooth it every 4 iterations
blur_step = 5
```

Now during our training, we'll smooth every `blur_step` iterations with the given `sigma`.


```python
img_copy = img_noise.copy()
imgs = []
for it_i in range(n_iterations):
    print(it_i, end=', ')
    this_res = sess.run(gradient[0], feed_dict={
        x: img_copy,
        layer: layer_vec})[0]
    this_res /= (np.max(np.abs(this_res)) + 1e-8)
    img_copy += this_res * step
    img_copy *= decay
    if it_i % blur_step == 0:
        for ch_i in range(3):
            img_copy[..., ch_i] = gaussian_filter(img_copy[..., ch_i], sigma)
    if it_i % gif_step == 0:
        imgs.append(normalize(img_copy[0]))
gif.build_gif(imgs, saveto='4-gaussian-' + str(neuron_i) + '.gif')
```

    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 




    <matplotlib.animation.ArtistAnimation at 0x15ae54fd0>




![png](lecture-4_files/lecture-4_91_2.png)



```python
ipyd.Image(url='4-gaussian-962.gif', height=300, width=300)
```




<img src="4-gaussian-962.gif" width="300" height="300"/>



Now we're really starting to get closer to something that resembles a school bus, or maybe a school bus double rainbow.

<a name="clipping-the-gradient"></a>
## Clipping the Gradient

Let's now see what happens if we clip the gradient's activations:


```python
pth = 5
img_copy = img_noise.copy()
imgs = []
for it_i in range(n_iterations):
    print(it_i, end=', ')
    this_res = sess.run(gradient[0], feed_dict={
        x: img_copy,
        layer: layer_vec})[0]
    this_res /= (np.max(np.abs(this_res)) + 1e-8)
    img_copy += this_res * step
    img_copy *= decay
    if it_i % blur_step == 0:
        for ch_i in range(3):
            img_copy[..., ch_i] = gaussian_filter(img_copy[..., ch_i], sigma)

    mask = (abs(img_copy) < np.percentile(abs(img_copy), pth))
    img_copy = img_copy - img_copy*mask

    if it_i % gif_step == 0:
        imgs.append(normalize(img_copy[0]))

plt.imshow(normalize(img_copy[0]))
gif.build_gif(imgs, saveto='5-clip-' + str(neuron_i) + '.gif')
```

    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 




    <matplotlib.animation.ArtistAnimation at 0x170b79080>




![png](lecture-4_files/lecture-4_94_2.png)



![png](lecture-4_files/lecture-4_94_3.png)



```python
ipyd.Image(url='5-clip-962.gif', height=300, width=300)
```




<img src="5-clip-962.gif" width="300" height="300"/>



<a name="infinite-zoom--fractal"></a>
## Infinite Zoom / Fractal

Some of the first visualizations to come out would infinitely zoom into the image, creating a fractal image of the deep dream.  We can do this by cropping the image with a 1 pixel border, and then resizing that image back to the original size.


```python
from skimage.transform import resize
img_copy = img_noise.copy()
crop = 1
n_iterations = 1000
imgs = []
n_img, height, width, ch = img_copy.shape
for it_i in range(n_iterations):
    print(it_i, end=', ')
    this_res = sess.run(gradient[0], feed_dict={
        x: img_copy,
        layer: layer_vec})[0]
    this_res /= (np.max(np.abs(this_res)) + 1e-8)
    img_copy += this_res * step
    img_copy *= decay

    if it_i % blur_step == 0:
        for ch_i in range(3):
            img_copy[..., ch_i] = gaussian_filter(img_copy[..., ch_i], sigma)

    mask = (abs(img_copy) < np.percentile(abs(img_copy), pth))
    img_copy = img_copy - img_copy * mask

    # Crop a 1 pixel border from height and width
    img_copy = img_copy[:, crop:-crop, crop:-crop, :]

    # Resize (Note: in the lecture, we used scipy's resize which
    # could not resize images outside of 0-1 range, and so we had
    # to store the image ranges.  This is a much simpler resize
    # method that allows us to `preserve_range`.)
    img_copy = resize(img_copy[0], (height, width), order=3,
                 clip=False, preserve_range=True
                 )[np.newaxis].astype(np.float32)

    if it_i % gif_step == 0:
        imgs.append(normalize(img_copy[0]))

gif.build_gif(imgs, saveto='6-fractal.gif')
```

    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255, 256, 257, 258, 259, 260, 261, 262, 263, 264, 265, 266, 267, 268, 269, 270, 271, 272, 273, 274, 275, 276, 277, 278, 279, 280, 281, 282, 283, 284, 285, 286, 287, 288, 289, 290, 291, 292, 293, 294, 295, 296, 297, 298, 299, 300, 301, 302, 303, 304, 305, 306, 307, 308, 309, 310, 311, 312, 313, 314, 315, 316, 317, 318, 319, 320, 321, 322, 323, 324, 325, 326, 327, 328, 329, 330, 331, 332, 333, 334, 335, 336, 337, 338, 339, 340, 341, 342, 343, 344, 345, 346, 347, 348, 349, 350, 351, 352, 353, 354, 355, 356, 357, 358, 359, 360, 361, 362, 363, 364, 365, 366, 367, 368, 369, 370, 371, 372, 373, 374, 375, 376, 377, 378, 379, 380, 381, 382, 383, 384, 385, 386, 387, 388, 389, 390, 391, 392, 393, 394, 395, 396, 397, 398, 399, 400, 401, 402, 403, 404, 405, 406, 407, 408, 409, 410, 411, 412, 413, 414, 415, 416, 417, 418, 419, 420, 421, 422, 423, 424, 425, 426, 427, 428, 429, 430, 431, 432, 433, 434, 435, 436, 437, 438, 439, 440, 441, 442, 443, 444, 445, 446, 447, 448, 449, 450, 451, 452, 453, 454, 455, 456, 457, 458, 459, 460, 461, 462, 463, 464, 465, 466, 467, 468, 469, 470, 471, 472, 473, 474, 475, 476, 477, 478, 479, 480, 481, 482, 483, 484, 485, 486, 487, 488, 489, 490, 491, 492, 493, 494, 495, 496, 497, 498, 499, 500, 501, 502, 503, 504, 505, 506, 507, 508, 509, 510, 511, 512, 513, 514, 515, 516, 517, 518, 519, 520, 521, 522, 523, 524, 525, 526, 527, 528, 529, 530, 531, 532, 533, 534, 535, 536, 537, 538, 539, 540, 541, 542, 543, 544, 545, 546, 547, 548, 549, 550, 551, 552, 553, 554, 555, 556, 557, 558, 559, 560, 561, 562, 563, 564, 565, 566, 567, 568, 569, 570, 571, 572, 573, 574, 575, 576, 577, 578, 579, 580, 581, 582, 583, 584, 585, 586, 587, 588, 589, 590, 591, 592, 593, 594, 595, 596, 597, 598, 599, 600, 601, 602, 603, 604, 605, 606, 607, 608, 609, 610, 611, 612, 613, 614, 615, 616, 617, 618, 619, 620, 621, 622, 623, 624, 625, 626, 627, 628, 629, 630, 631, 632, 633, 634, 635, 636, 637, 638, 639, 640, 641, 642, 643, 644, 645, 646, 647, 648, 649, 650, 651, 652, 653, 654, 655, 656, 657, 658, 659, 660, 661, 662, 663, 664, 665, 666, 667, 668, 669, 670, 671, 672, 673, 674, 675, 676, 677, 678, 679, 680, 681, 682, 683, 684, 685, 686, 687, 688, 689, 690, 691, 692, 693, 694, 695, 696, 697, 698, 699, 700, 701, 702, 703, 704, 705, 706, 707, 708, 709, 710, 711, 712, 713, 714, 715, 716, 717, 718, 719, 720, 721, 722, 723, 724, 725, 726, 727, 728, 729, 730, 731, 732, 733, 734, 735, 736, 737, 738, 739, 740, 741, 742, 743, 744, 745, 746, 747, 748, 749, 750, 751, 752, 753, 754, 755, 756, 757, 758, 759, 760, 761, 762, 763, 764, 765, 766, 767, 768, 769, 770, 771, 772, 773, 774, 775, 776, 777, 778, 779, 780, 781, 782, 783, 784, 785, 786, 787, 788, 789, 790, 791, 792, 793, 794, 795, 796, 797, 798, 799, 800, 801, 802, 803, 804, 805, 806, 807, 808, 809, 810, 811, 812, 813, 814, 815, 816, 817, 818, 819, 820, 821, 822, 823, 824, 825, 826, 827, 828, 829, 830, 831, 832, 833, 834, 835, 836, 837, 838, 839, 840, 841, 842, 843, 844, 845, 846, 847, 848, 849, 850, 851, 852, 853, 854, 855, 856, 857, 858, 859, 860, 861, 862, 863, 864, 865, 866, 867, 868, 869, 870, 871, 872, 873, 874, 875, 876, 877, 878, 879, 880, 881, 882, 883, 884, 885, 886, 887, 888, 889, 890, 891, 892, 893, 894, 895, 896, 897, 898, 899, 900, 901, 902, 903, 904, 905, 906, 907, 908, 909, 910, 911, 912, 913, 914, 915, 916, 917, 918, 919, 920, 921, 922, 923, 924, 925, 926, 927, 928, 929, 930, 931, 932, 933, 934, 935, 936, 937, 938, 939, 940, 941, 942, 943, 944, 945, 946, 947, 948, 949, 950, 951, 952, 953, 954, 955, 956, 957, 958, 959, 960, 961, 962, 963, 964, 965, 966, 967, 968, 969, 970, 971, 972, 973, 974, 975, 976, 977, 978, 979, 980, 981, 982, 983, 984, 985, 986, 987, 988, 989, 990, 991, 992, 993, 994, 995, 996, 997, 998, 999, 




    <matplotlib.animation.ArtistAnimation at 0x177b0c9b0>




![png](lecture-4_files/lecture-4_97_2.png)



```python
ipyd.Image(url='6-fractal.gif', height=300, width=300)
```




<img src="6-fractal.gif" width="300" height="300"/>



<a name="style-net"></a>
# Style Net

Leon Gatys and his co-authors demonstrated a pretty epic extension to deep dream which showed that neural networks trained on objects like the one we've been using actually represent both content and style, and that these can be independently manipulated, for instance taking the content from one image, and the style from another.  They showed how you could artistically stylize the same image with a wide range of different painterly aesthetics.  Let's take a look at how we can do that.  We're going to use the same network that they've used in their paper, VGG.  This network is a lot less complicated than the Inception network, but at the expense of having a lot more parameters.

<a name="vgg-network"></a>
## VGG Network

In the resources section, you can find the library for loading this network, just like you've done w/ the Inception network.  Let's reset the graph:


```python
import tensorflow as tf
from libs import utils
```


```python
from tensorflow.python.framework.ops import reset_default_graph
sess.close()
reset_default_graph()
```

And now we'll load up the new network, except unlike before, we're going to explicitly create a graph, and tell the session to use this graph.  If we didn't do this, tensorflow would just use the default graph that is always there.  But since we're going to be making a few graphs, we'll need to do it like this.


```python
from libs import vgg16
net = vgg16.get_vgg_model()
```

Note: We will explicitly define a context manager here to handle the graph and place the graph in CPU memory instead of GPU memory, as this is a very large network!


```python
g = tf.Graph()
with tf.Session(graph=g) as sess, g.device('/cpu:0'):
    tf.import_graph_def(net['graph_def'], name='vgg')
    names = [op.name for op in g.get_operations()]
```

Let's take a look at the network:


```python
nb_utils.show_graph(net['graph_def'])
```



            <iframe seamless style="width:800px;height:620px;border:0" srcdoc="
            <script>
              function load() {
                document.getElementById(&quot;graph0.09864523476836828&quot;).pbtxt = 'node {\n  name: &quot;images&quot;\n  op: &quot;Placeholder&quot;\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;shape&quot;\n    value {\n      shape {\n      }\n    }\n  }\n}\nnode {\n  name: &quot;mul/y&quot;\n  op: &quot;Const&quot;\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;value&quot;\n    value {\n      tensor {\n        dtype: DT_FLOAT\n        tensor_shape {\n        }\n        float_val: 255.0\n      }\n    }\n  }\n}\nnode {\n  name: &quot;mul&quot;\n  op: &quot;Mul&quot;\n  input: &quot;images&quot;\n  input: &quot;mul/y&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n}\nnode {\n  name: &quot;split/split/dim&quot;\n  op: &quot;Const&quot;\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_INT32\n    }\n  }\n  attr {\n    key: &quot;value&quot;\n    value {\n      tensor {\n        dtype: DT_INT32\n        tensor_shape {\n        }\n        int_val: 3\n      }\n    }\n  }\n}\nnode {\n  name: &quot;split&quot;\n  op: &quot;Split&quot;\n  input: &quot;split/split/dim&quot;\n  input: &quot;mul&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;num_split&quot;\n    value {\n      i: 3\n    }\n  }\n}\nnode {\n  name: &quot;sub/y&quot;\n  op: &quot;Const&quot;\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;value&quot;\n    value {\n      tensor {\n        dtype: DT_FLOAT\n        tensor_shape {\n        }\n        float_val: 103.93900299072266\n      }\n    }\n  }\n}\nnode {\n  name: &quot;sub&quot;\n  op: &quot;Sub&quot;\n  input: &quot;split:2&quot;\n  input: &quot;sub/y&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n}\nnode {\n  name: &quot;sub/1/y&quot;\n  op: &quot;Const&quot;\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;value&quot;\n    value {\n      tensor {\n        dtype: DT_FLOAT\n        tensor_shape {\n        }\n        float_val: 116.77899932861328\n      }\n    }\n  }\n}\nnode {\n  name: &quot;sub/1&quot;\n  op: &quot;Sub&quot;\n  input: &quot;split:1&quot;\n  input: &quot;sub/1/y&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n}\nnode {\n  name: &quot;sub/2/y&quot;\n  op: &quot;Const&quot;\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;value&quot;\n    value {\n      tensor {\n        dtype: DT_FLOAT\n        tensor_shape {\n        }\n        float_val: 123.68000030517578\n      }\n    }\n  }\n}\nnode {\n  name: &quot;sub/2&quot;\n  op: &quot;Sub&quot;\n  input: &quot;split&quot;\n  input: &quot;sub/2/y&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n}\nnode {\n  name: &quot;concat/concat/dim&quot;\n  op: &quot;Const&quot;\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_INT32\n    }\n  }\n  attr {\n    key: &quot;value&quot;\n    value {\n      tensor {\n        dtype: DT_INT32\n        tensor_shape {\n        }\n        int_val: 3\n      }\n    }\n  }\n}\nnode {\n  name: &quot;concat&quot;\n  op: &quot;Concat&quot;\n  input: &quot;concat/concat/dim&quot;\n  input: &quot;sub&quot;\n  input: &quot;sub/1&quot;\n  input: &quot;sub/2&quot;\n  attr {\n    key: &quot;N&quot;\n    value {\n      i: 3\n    }\n  }\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n}\nnode {\n  name: &quot;conv1/1/filter&quot;\n  op: &quot;Const&quot;\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;value&quot;\n    value {\n      tensor {\n        dtype: DT_FLOAT\n        tensor_shape {\n          dim {\n            size: 3\n          }\n          dim {\n            size: 3\n          }\n          dim {\n            size: 3\n          }\n          dim {\n            size: 64\n          }\n        }\n        tensor_content: &quot;<stripped 6912 bytes>&quot;\n      }\n    }\n  }\n}\nnode {\n  name: &quot;conv1/1/Conv2D&quot;\n  op: &quot;Conv2D&quot;\n  input: &quot;concat&quot;\n  input: &quot;conv1/1/filter&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;data_format&quot;\n    value {\n      s: &quot;NHWC&quot;\n    }\n  }\n  attr {\n    key: &quot;padding&quot;\n    value {\n      s: &quot;SAME&quot;\n    }\n  }\n  attr {\n    key: &quot;strides&quot;\n    value {\n      list {\n        i: 1\n        i: 1\n        i: 1\n        i: 1\n      }\n    }\n  }\n  attr {\n    key: &quot;use_cudnn_on_gpu&quot;\n    value {\n      b: true\n    }\n  }\n}\nnode {\n  name: &quot;conv1/1/biases&quot;\n  op: &quot;Const&quot;\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;value&quot;\n    value {\n      tensor {\n        dtype: DT_FLOAT\n        tensor_shape {\n          dim {\n            size: 64\n          }\n        }\n        tensor_content: &quot;<stripped 256 bytes>&quot;\n      }\n    }\n  }\n}\nnode {\n  name: &quot;conv1/1/BiasAdd&quot;\n  op: &quot;BiasAdd&quot;\n  input: &quot;conv1/1/Conv2D&quot;\n  input: &quot;conv1/1/biases&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;data_format&quot;\n    value {\n      s: &quot;NHWC&quot;\n    }\n  }\n}\nnode {\n  name: &quot;conv1/1/conv1_1&quot;\n  op: &quot;Relu&quot;\n  input: &quot;conv1/1/BiasAdd&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n}\nnode {\n  name: &quot;conv1/2/filter&quot;\n  op: &quot;Const&quot;\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;value&quot;\n    value {\n      tensor {\n        dtype: DT_FLOAT\n        tensor_shape {\n          dim {\n            size: 3\n          }\n          dim {\n            size: 3\n          }\n          dim {\n            size: 64\n          }\n          dim {\n            size: 64\n          }\n        }\n        tensor_content: &quot;<stripped 147456 bytes>&quot;\n      }\n    }\n  }\n}\nnode {\n  name: &quot;conv1/2/Conv2D&quot;\n  op: &quot;Conv2D&quot;\n  input: &quot;conv1/1/conv1_1&quot;\n  input: &quot;conv1/2/filter&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;data_format&quot;\n    value {\n      s: &quot;NHWC&quot;\n    }\n  }\n  attr {\n    key: &quot;padding&quot;\n    value {\n      s: &quot;SAME&quot;\n    }\n  }\n  attr {\n    key: &quot;strides&quot;\n    value {\n      list {\n        i: 1\n        i: 1\n        i: 1\n        i: 1\n      }\n    }\n  }\n  attr {\n    key: &quot;use_cudnn_on_gpu&quot;\n    value {\n      b: true\n    }\n  }\n}\nnode {\n  name: &quot;conv1/2/biases&quot;\n  op: &quot;Const&quot;\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;value&quot;\n    value {\n      tensor {\n        dtype: DT_FLOAT\n        tensor_shape {\n          dim {\n            size: 64\n          }\n        }\n        tensor_content: &quot;<stripped 256 bytes>&quot;\n      }\n    }\n  }\n}\nnode {\n  name: &quot;conv1/2/BiasAdd&quot;\n  op: &quot;BiasAdd&quot;\n  input: &quot;conv1/2/Conv2D&quot;\n  input: &quot;conv1/2/biases&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;data_format&quot;\n    value {\n      s: &quot;NHWC&quot;\n    }\n  }\n}\nnode {\n  name: &quot;conv1/2/conv1_2&quot;\n  op: &quot;Relu&quot;\n  input: &quot;conv1/2/BiasAdd&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n}\nnode {\n  name: &quot;pool1&quot;\n  op: &quot;MaxPool&quot;\n  input: &quot;conv1/2/conv1_2&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;data_format&quot;\n    value {\n      s: &quot;NHWC&quot;\n    }\n  }\n  attr {\n    key: &quot;ksize&quot;\n    value {\n      list {\n        i: 1\n        i: 2\n        i: 2\n        i: 1\n      }\n    }\n  }\n  attr {\n    key: &quot;padding&quot;\n    value {\n      s: &quot;SAME&quot;\n    }\n  }\n  attr {\n    key: &quot;strides&quot;\n    value {\n      list {\n        i: 1\n        i: 2\n        i: 2\n        i: 1\n      }\n    }\n  }\n}\nnode {\n  name: &quot;conv2/1/filter&quot;\n  op: &quot;Const&quot;\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;value&quot;\n    value {\n      tensor {\n        dtype: DT_FLOAT\n        tensor_shape {\n          dim {\n            size: 3\n          }\n          dim {\n            size: 3\n          }\n          dim {\n            size: 64\n          }\n          dim {\n            size: 128\n          }\n        }\n        tensor_content: &quot;<stripped 294912 bytes>&quot;\n      }\n    }\n  }\n}\nnode {\n  name: &quot;conv2/1/Conv2D&quot;\n  op: &quot;Conv2D&quot;\n  input: &quot;pool1&quot;\n  input: &quot;conv2/1/filter&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;data_format&quot;\n    value {\n      s: &quot;NHWC&quot;\n    }\n  }\n  attr {\n    key: &quot;padding&quot;\n    value {\n      s: &quot;SAME&quot;\n    }\n  }\n  attr {\n    key: &quot;strides&quot;\n    value {\n      list {\n        i: 1\n        i: 1\n        i: 1\n        i: 1\n      }\n    }\n  }\n  attr {\n    key: &quot;use_cudnn_on_gpu&quot;\n    value {\n      b: true\n    }\n  }\n}\nnode {\n  name: &quot;conv2/1/biases&quot;\n  op: &quot;Const&quot;\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;value&quot;\n    value {\n      tensor {\n        dtype: DT_FLOAT\n        tensor_shape {\n          dim {\n            size: 128\n          }\n        }\n        tensor_content: &quot;<stripped 512 bytes>&quot;\n      }\n    }\n  }\n}\nnode {\n  name: &quot;conv2/1/BiasAdd&quot;\n  op: &quot;BiasAdd&quot;\n  input: &quot;conv2/1/Conv2D&quot;\n  input: &quot;conv2/1/biases&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;data_format&quot;\n    value {\n      s: &quot;NHWC&quot;\n    }\n  }\n}\nnode {\n  name: &quot;conv2/1/conv2_1&quot;\n  op: &quot;Relu&quot;\n  input: &quot;conv2/1/BiasAdd&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n}\nnode {\n  name: &quot;conv2/2/filter&quot;\n  op: &quot;Const&quot;\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;value&quot;\n    value {\n      tensor {\n        dtype: DT_FLOAT\n        tensor_shape {\n          dim {\n            size: 3\n          }\n          dim {\n            size: 3\n          }\n          dim {\n            size: 128\n          }\n          dim {\n            size: 128\n          }\n        }\n        tensor_content: &quot;<stripped 589824 bytes>&quot;\n      }\n    }\n  }\n}\nnode {\n  name: &quot;conv2/2/Conv2D&quot;\n  op: &quot;Conv2D&quot;\n  input: &quot;conv2/1/conv2_1&quot;\n  input: &quot;conv2/2/filter&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;data_format&quot;\n    value {\n      s: &quot;NHWC&quot;\n    }\n  }\n  attr {\n    key: &quot;padding&quot;\n    value {\n      s: &quot;SAME&quot;\n    }\n  }\n  attr {\n    key: &quot;strides&quot;\n    value {\n      list {\n        i: 1\n        i: 1\n        i: 1\n        i: 1\n      }\n    }\n  }\n  attr {\n    key: &quot;use_cudnn_on_gpu&quot;\n    value {\n      b: true\n    }\n  }\n}\nnode {\n  name: &quot;conv2/2/biases&quot;\n  op: &quot;Const&quot;\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;value&quot;\n    value {\n      tensor {\n        dtype: DT_FLOAT\n        tensor_shape {\n          dim {\n            size: 128\n          }\n        }\n        tensor_content: &quot;<stripped 512 bytes>&quot;\n      }\n    }\n  }\n}\nnode {\n  name: &quot;conv2/2/BiasAdd&quot;\n  op: &quot;BiasAdd&quot;\n  input: &quot;conv2/2/Conv2D&quot;\n  input: &quot;conv2/2/biases&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;data_format&quot;\n    value {\n      s: &quot;NHWC&quot;\n    }\n  }\n}\nnode {\n  name: &quot;conv2/2/conv2_2&quot;\n  op: &quot;Relu&quot;\n  input: &quot;conv2/2/BiasAdd&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n}\nnode {\n  name: &quot;pool2&quot;\n  op: &quot;MaxPool&quot;\n  input: &quot;conv2/2/conv2_2&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;data_format&quot;\n    value {\n      s: &quot;NHWC&quot;\n    }\n  }\n  attr {\n    key: &quot;ksize&quot;\n    value {\n      list {\n        i: 1\n        i: 2\n        i: 2\n        i: 1\n      }\n    }\n  }\n  attr {\n    key: &quot;padding&quot;\n    value {\n      s: &quot;SAME&quot;\n    }\n  }\n  attr {\n    key: &quot;strides&quot;\n    value {\n      list {\n        i: 1\n        i: 2\n        i: 2\n        i: 1\n      }\n    }\n  }\n}\nnode {\n  name: &quot;conv3/1/filter&quot;\n  op: &quot;Const&quot;\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;value&quot;\n    value {\n      tensor {\n        dtype: DT_FLOAT\n        tensor_shape {\n          dim {\n            size: 3\n          }\n          dim {\n            size: 3\n          }\n          dim {\n            size: 128\n          }\n          dim {\n            size: 256\n          }\n        }\n        tensor_content: &quot;<stripped 1179648 bytes>&quot;\n      }\n    }\n  }\n}\nnode {\n  name: &quot;conv3/1/Conv2D&quot;\n  op: &quot;Conv2D&quot;\n  input: &quot;pool2&quot;\n  input: &quot;conv3/1/filter&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;data_format&quot;\n    value {\n      s: &quot;NHWC&quot;\n    }\n  }\n  attr {\n    key: &quot;padding&quot;\n    value {\n      s: &quot;SAME&quot;\n    }\n  }\n  attr {\n    key: &quot;strides&quot;\n    value {\n      list {\n        i: 1\n        i: 1\n        i: 1\n        i: 1\n      }\n    }\n  }\n  attr {\n    key: &quot;use_cudnn_on_gpu&quot;\n    value {\n      b: true\n    }\n  }\n}\nnode {\n  name: &quot;conv3/1/biases&quot;\n  op: &quot;Const&quot;\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;value&quot;\n    value {\n      tensor {\n        dtype: DT_FLOAT\n        tensor_shape {\n          dim {\n            size: 256\n          }\n        }\n        tensor_content: &quot;<stripped 1024 bytes>&quot;\n      }\n    }\n  }\n}\nnode {\n  name: &quot;conv3/1/BiasAdd&quot;\n  op: &quot;BiasAdd&quot;\n  input: &quot;conv3/1/Conv2D&quot;\n  input: &quot;conv3/1/biases&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;data_format&quot;\n    value {\n      s: &quot;NHWC&quot;\n    }\n  }\n}\nnode {\n  name: &quot;conv3/1/conv3_1&quot;\n  op: &quot;Relu&quot;\n  input: &quot;conv3/1/BiasAdd&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n}\nnode {\n  name: &quot;conv3/2/filter&quot;\n  op: &quot;Const&quot;\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;value&quot;\n    value {\n      tensor {\n        dtype: DT_FLOAT\n        tensor_shape {\n          dim {\n            size: 3\n          }\n          dim {\n            size: 3\n          }\n          dim {\n            size: 256\n          }\n          dim {\n            size: 256\n          }\n        }\n        tensor_content: &quot;<stripped 2359296 bytes>&quot;\n      }\n    }\n  }\n}\nnode {\n  name: &quot;conv3/2/Conv2D&quot;\n  op: &quot;Conv2D&quot;\n  input: &quot;conv3/1/conv3_1&quot;\n  input: &quot;conv3/2/filter&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;data_format&quot;\n    value {\n      s: &quot;NHWC&quot;\n    }\n  }\n  attr {\n    key: &quot;padding&quot;\n    value {\n      s: &quot;SAME&quot;\n    }\n  }\n  attr {\n    key: &quot;strides&quot;\n    value {\n      list {\n        i: 1\n        i: 1\n        i: 1\n        i: 1\n      }\n    }\n  }\n  attr {\n    key: &quot;use_cudnn_on_gpu&quot;\n    value {\n      b: true\n    }\n  }\n}\nnode {\n  name: &quot;conv3/2/biases&quot;\n  op: &quot;Const&quot;\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;value&quot;\n    value {\n      tensor {\n        dtype: DT_FLOAT\n        tensor_shape {\n          dim {\n            size: 256\n          }\n        }\n        tensor_content: &quot;<stripped 1024 bytes>&quot;\n      }\n    }\n  }\n}\nnode {\n  name: &quot;conv3/2/BiasAdd&quot;\n  op: &quot;BiasAdd&quot;\n  input: &quot;conv3/2/Conv2D&quot;\n  input: &quot;conv3/2/biases&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;data_format&quot;\n    value {\n      s: &quot;NHWC&quot;\n    }\n  }\n}\nnode {\n  name: &quot;conv3/2/conv3_2&quot;\n  op: &quot;Relu&quot;\n  input: &quot;conv3/2/BiasAdd&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n}\nnode {\n  name: &quot;conv3/3/filter&quot;\n  op: &quot;Const&quot;\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;value&quot;\n    value {\n      tensor {\n        dtype: DT_FLOAT\n        tensor_shape {\n          dim {\n            size: 3\n          }\n          dim {\n            size: 3\n          }\n          dim {\n            size: 256\n          }\n          dim {\n            size: 256\n          }\n        }\n        tensor_content: &quot;<stripped 2359296 bytes>&quot;\n      }\n    }\n  }\n}\nnode {\n  name: &quot;conv3/3/Conv2D&quot;\n  op: &quot;Conv2D&quot;\n  input: &quot;conv3/2/conv3_2&quot;\n  input: &quot;conv3/3/filter&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;data_format&quot;\n    value {\n      s: &quot;NHWC&quot;\n    }\n  }\n  attr {\n    key: &quot;padding&quot;\n    value {\n      s: &quot;SAME&quot;\n    }\n  }\n  attr {\n    key: &quot;strides&quot;\n    value {\n      list {\n        i: 1\n        i: 1\n        i: 1\n        i: 1\n      }\n    }\n  }\n  attr {\n    key: &quot;use_cudnn_on_gpu&quot;\n    value {\n      b: true\n    }\n  }\n}\nnode {\n  name: &quot;conv3/3/biases&quot;\n  op: &quot;Const&quot;\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;value&quot;\n    value {\n      tensor {\n        dtype: DT_FLOAT\n        tensor_shape {\n          dim {\n            size: 256\n          }\n        }\n        tensor_content: &quot;<stripped 1024 bytes>&quot;\n      }\n    }\n  }\n}\nnode {\n  name: &quot;conv3/3/BiasAdd&quot;\n  op: &quot;BiasAdd&quot;\n  input: &quot;conv3/3/Conv2D&quot;\n  input: &quot;conv3/3/biases&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;data_format&quot;\n    value {\n      s: &quot;NHWC&quot;\n    }\n  }\n}\nnode {\n  name: &quot;conv3/3/conv3_3&quot;\n  op: &quot;Relu&quot;\n  input: &quot;conv3/3/BiasAdd&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n}\nnode {\n  name: &quot;pool3&quot;\n  op: &quot;MaxPool&quot;\n  input: &quot;conv3/3/conv3_3&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;data_format&quot;\n    value {\n      s: &quot;NHWC&quot;\n    }\n  }\n  attr {\n    key: &quot;ksize&quot;\n    value {\n      list {\n        i: 1\n        i: 2\n        i: 2\n        i: 1\n      }\n    }\n  }\n  attr {\n    key: &quot;padding&quot;\n    value {\n      s: &quot;SAME&quot;\n    }\n  }\n  attr {\n    key: &quot;strides&quot;\n    value {\n      list {\n        i: 1\n        i: 2\n        i: 2\n        i: 1\n      }\n    }\n  }\n}\nnode {\n  name: &quot;conv4/1/filter&quot;\n  op: &quot;Const&quot;\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;value&quot;\n    value {\n      tensor {\n        dtype: DT_FLOAT\n        tensor_shape {\n          dim {\n            size: 3\n          }\n          dim {\n            size: 3\n          }\n          dim {\n            size: 256\n          }\n          dim {\n            size: 512\n          }\n        }\n        tensor_content: &quot;<stripped 4718592 bytes>&quot;\n      }\n    }\n  }\n}\nnode {\n  name: &quot;conv4/1/Conv2D&quot;\n  op: &quot;Conv2D&quot;\n  input: &quot;pool3&quot;\n  input: &quot;conv4/1/filter&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;data_format&quot;\n    value {\n      s: &quot;NHWC&quot;\n    }\n  }\n  attr {\n    key: &quot;padding&quot;\n    value {\n      s: &quot;SAME&quot;\n    }\n  }\n  attr {\n    key: &quot;strides&quot;\n    value {\n      list {\n        i: 1\n        i: 1\n        i: 1\n        i: 1\n      }\n    }\n  }\n  attr {\n    key: &quot;use_cudnn_on_gpu&quot;\n    value {\n      b: true\n    }\n  }\n}\nnode {\n  name: &quot;conv4/1/biases&quot;\n  op: &quot;Const&quot;\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;value&quot;\n    value {\n      tensor {\n        dtype: DT_FLOAT\n        tensor_shape {\n          dim {\n            size: 512\n          }\n        }\n        tensor_content: &quot;<stripped 2048 bytes>&quot;\n      }\n    }\n  }\n}\nnode {\n  name: &quot;conv4/1/BiasAdd&quot;\n  op: &quot;BiasAdd&quot;\n  input: &quot;conv4/1/Conv2D&quot;\n  input: &quot;conv4/1/biases&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;data_format&quot;\n    value {\n      s: &quot;NHWC&quot;\n    }\n  }\n}\nnode {\n  name: &quot;conv4/1/conv4_1&quot;\n  op: &quot;Relu&quot;\n  input: &quot;conv4/1/BiasAdd&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n}\nnode {\n  name: &quot;conv4/2/filter&quot;\n  op: &quot;Const&quot;\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;value&quot;\n    value {\n      tensor {\n        dtype: DT_FLOAT\n        tensor_shape {\n          dim {\n            size: 3\n          }\n          dim {\n            size: 3\n          }\n          dim {\n            size: 512\n          }\n          dim {\n            size: 512\n          }\n        }\n        tensor_content: &quot;<stripped 9437184 bytes>&quot;\n      }\n    }\n  }\n}\nnode {\n  name: &quot;conv4/2/Conv2D&quot;\n  op: &quot;Conv2D&quot;\n  input: &quot;conv4/1/conv4_1&quot;\n  input: &quot;conv4/2/filter&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;data_format&quot;\n    value {\n      s: &quot;NHWC&quot;\n    }\n  }\n  attr {\n    key: &quot;padding&quot;\n    value {\n      s: &quot;SAME&quot;\n    }\n  }\n  attr {\n    key: &quot;strides&quot;\n    value {\n      list {\n        i: 1\n        i: 1\n        i: 1\n        i: 1\n      }\n    }\n  }\n  attr {\n    key: &quot;use_cudnn_on_gpu&quot;\n    value {\n      b: true\n    }\n  }\n}\nnode {\n  name: &quot;conv4/2/biases&quot;\n  op: &quot;Const&quot;\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;value&quot;\n    value {\n      tensor {\n        dtype: DT_FLOAT\n        tensor_shape {\n          dim {\n            size: 512\n          }\n        }\n        tensor_content: &quot;<stripped 2048 bytes>&quot;\n      }\n    }\n  }\n}\nnode {\n  name: &quot;conv4/2/BiasAdd&quot;\n  op: &quot;BiasAdd&quot;\n  input: &quot;conv4/2/Conv2D&quot;\n  input: &quot;conv4/2/biases&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;data_format&quot;\n    value {\n      s: &quot;NHWC&quot;\n    }\n  }\n}\nnode {\n  name: &quot;conv4/2/conv4_2&quot;\n  op: &quot;Relu&quot;\n  input: &quot;conv4/2/BiasAdd&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n}\nnode {\n  name: &quot;conv4/3/filter&quot;\n  op: &quot;Const&quot;\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;value&quot;\n    value {\n      tensor {\n        dtype: DT_FLOAT\n        tensor_shape {\n          dim {\n            size: 3\n          }\n          dim {\n            size: 3\n          }\n          dim {\n            size: 512\n          }\n          dim {\n            size: 512\n          }\n        }\n        tensor_content: &quot;<stripped 9437184 bytes>&quot;\n      }\n    }\n  }\n}\nnode {\n  name: &quot;conv4/3/Conv2D&quot;\n  op: &quot;Conv2D&quot;\n  input: &quot;conv4/2/conv4_2&quot;\n  input: &quot;conv4/3/filter&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;data_format&quot;\n    value {\n      s: &quot;NHWC&quot;\n    }\n  }\n  attr {\n    key: &quot;padding&quot;\n    value {\n      s: &quot;SAME&quot;\n    }\n  }\n  attr {\n    key: &quot;strides&quot;\n    value {\n      list {\n        i: 1\n        i: 1\n        i: 1\n        i: 1\n      }\n    }\n  }\n  attr {\n    key: &quot;use_cudnn_on_gpu&quot;\n    value {\n      b: true\n    }\n  }\n}\nnode {\n  name: &quot;conv4/3/biases&quot;\n  op: &quot;Const&quot;\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;value&quot;\n    value {\n      tensor {\n        dtype: DT_FLOAT\n        tensor_shape {\n          dim {\n            size: 512\n          }\n        }\n        tensor_content: &quot;<stripped 2048 bytes>&quot;\n      }\n    }\n  }\n}\nnode {\n  name: &quot;conv4/3/BiasAdd&quot;\n  op: &quot;BiasAdd&quot;\n  input: &quot;conv4/3/Conv2D&quot;\n  input: &quot;conv4/3/biases&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;data_format&quot;\n    value {\n      s: &quot;NHWC&quot;\n    }\n  }\n}\nnode {\n  name: &quot;conv4/3/conv4_3&quot;\n  op: &quot;Relu&quot;\n  input: &quot;conv4/3/BiasAdd&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n}\nnode {\n  name: &quot;pool4&quot;\n  op: &quot;MaxPool&quot;\n  input: &quot;conv4/3/conv4_3&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;data_format&quot;\n    value {\n      s: &quot;NHWC&quot;\n    }\n  }\n  attr {\n    key: &quot;ksize&quot;\n    value {\n      list {\n        i: 1\n        i: 2\n        i: 2\n        i: 1\n      }\n    }\n  }\n  attr {\n    key: &quot;padding&quot;\n    value {\n      s: &quot;SAME&quot;\n    }\n  }\n  attr {\n    key: &quot;strides&quot;\n    value {\n      list {\n        i: 1\n        i: 2\n        i: 2\n        i: 1\n      }\n    }\n  }\n}\nnode {\n  name: &quot;conv5/1/filter&quot;\n  op: &quot;Const&quot;\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;value&quot;\n    value {\n      tensor {\n        dtype: DT_FLOAT\n        tensor_shape {\n          dim {\n            size: 3\n          }\n          dim {\n            size: 3\n          }\n          dim {\n            size: 512\n          }\n          dim {\n            size: 512\n          }\n        }\n        tensor_content: &quot;<stripped 9437184 bytes>&quot;\n      }\n    }\n  }\n}\nnode {\n  name: &quot;conv5/1/Conv2D&quot;\n  op: &quot;Conv2D&quot;\n  input: &quot;pool4&quot;\n  input: &quot;conv5/1/filter&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;data_format&quot;\n    value {\n      s: &quot;NHWC&quot;\n    }\n  }\n  attr {\n    key: &quot;padding&quot;\n    value {\n      s: &quot;SAME&quot;\n    }\n  }\n  attr {\n    key: &quot;strides&quot;\n    value {\n      list {\n        i: 1\n        i: 1\n        i: 1\n        i: 1\n      }\n    }\n  }\n  attr {\n    key: &quot;use_cudnn_on_gpu&quot;\n    value {\n      b: true\n    }\n  }\n}\nnode {\n  name: &quot;conv5/1/biases&quot;\n  op: &quot;Const&quot;\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;value&quot;\n    value {\n      tensor {\n        dtype: DT_FLOAT\n        tensor_shape {\n          dim {\n            size: 512\n          }\n        }\n        tensor_content: &quot;<stripped 2048 bytes>&quot;\n      }\n    }\n  }\n}\nnode {\n  name: &quot;conv5/1/BiasAdd&quot;\n  op: &quot;BiasAdd&quot;\n  input: &quot;conv5/1/Conv2D&quot;\n  input: &quot;conv5/1/biases&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;data_format&quot;\n    value {\n      s: &quot;NHWC&quot;\n    }\n  }\n}\nnode {\n  name: &quot;conv5/1/conv5_1&quot;\n  op: &quot;Relu&quot;\n  input: &quot;conv5/1/BiasAdd&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n}\nnode {\n  name: &quot;conv5/2/filter&quot;\n  op: &quot;Const&quot;\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;value&quot;\n    value {\n      tensor {\n        dtype: DT_FLOAT\n        tensor_shape {\n          dim {\n            size: 3\n          }\n          dim {\n            size: 3\n          }\n          dim {\n            size: 512\n          }\n          dim {\n            size: 512\n          }\n        }\n        tensor_content: &quot;<stripped 9437184 bytes>&quot;\n      }\n    }\n  }\n}\nnode {\n  name: &quot;conv5/2/Conv2D&quot;\n  op: &quot;Conv2D&quot;\n  input: &quot;conv5/1/conv5_1&quot;\n  input: &quot;conv5/2/filter&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;data_format&quot;\n    value {\n      s: &quot;NHWC&quot;\n    }\n  }\n  attr {\n    key: &quot;padding&quot;\n    value {\n      s: &quot;SAME&quot;\n    }\n  }\n  attr {\n    key: &quot;strides&quot;\n    value {\n      list {\n        i: 1\n        i: 1\n        i: 1\n        i: 1\n      }\n    }\n  }\n  attr {\n    key: &quot;use_cudnn_on_gpu&quot;\n    value {\n      b: true\n    }\n  }\n}\nnode {\n  name: &quot;conv5/2/biases&quot;\n  op: &quot;Const&quot;\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;value&quot;\n    value {\n      tensor {\n        dtype: DT_FLOAT\n        tensor_shape {\n          dim {\n            size: 512\n          }\n        }\n        tensor_content: &quot;<stripped 2048 bytes>&quot;\n      }\n    }\n  }\n}\nnode {\n  name: &quot;conv5/2/BiasAdd&quot;\n  op: &quot;BiasAdd&quot;\n  input: &quot;conv5/2/Conv2D&quot;\n  input: &quot;conv5/2/biases&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;data_format&quot;\n    value {\n      s: &quot;NHWC&quot;\n    }\n  }\n}\nnode {\n  name: &quot;conv5/2/conv5_2&quot;\n  op: &quot;Relu&quot;\n  input: &quot;conv5/2/BiasAdd&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n}\nnode {\n  name: &quot;conv5/3/filter&quot;\n  op: &quot;Const&quot;\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;value&quot;\n    value {\n      tensor {\n        dtype: DT_FLOAT\n        tensor_shape {\n          dim {\n            size: 3\n          }\n          dim {\n            size: 3\n          }\n          dim {\n            size: 512\n          }\n          dim {\n            size: 512\n          }\n        }\n        tensor_content: &quot;<stripped 9437184 bytes>&quot;\n      }\n    }\n  }\n}\nnode {\n  name: &quot;conv5/3/Conv2D&quot;\n  op: &quot;Conv2D&quot;\n  input: &quot;conv5/2/conv5_2&quot;\n  input: &quot;conv5/3/filter&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;data_format&quot;\n    value {\n      s: &quot;NHWC&quot;\n    }\n  }\n  attr {\n    key: &quot;padding&quot;\n    value {\n      s: &quot;SAME&quot;\n    }\n  }\n  attr {\n    key: &quot;strides&quot;\n    value {\n      list {\n        i: 1\n        i: 1\n        i: 1\n        i: 1\n      }\n    }\n  }\n  attr {\n    key: &quot;use_cudnn_on_gpu&quot;\n    value {\n      b: true\n    }\n  }\n}\nnode {\n  name: &quot;conv5/3/biases&quot;\n  op: &quot;Const&quot;\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;value&quot;\n    value {\n      tensor {\n        dtype: DT_FLOAT\n        tensor_shape {\n          dim {\n            size: 512\n          }\n        }\n        tensor_content: &quot;<stripped 2048 bytes>&quot;\n      }\n    }\n  }\n}\nnode {\n  name: &quot;conv5/3/BiasAdd&quot;\n  op: &quot;BiasAdd&quot;\n  input: &quot;conv5/3/Conv2D&quot;\n  input: &quot;conv5/3/biases&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;data_format&quot;\n    value {\n      s: &quot;NHWC&quot;\n    }\n  }\n}\nnode {\n  name: &quot;conv5/3/conv5_3&quot;\n  op: &quot;Relu&quot;\n  input: &quot;conv5/3/BiasAdd&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n}\nnode {\n  name: &quot;pool5&quot;\n  op: &quot;MaxPool&quot;\n  input: &quot;conv5/3/conv5_3&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;data_format&quot;\n    value {\n      s: &quot;NHWC&quot;\n    }\n  }\n  attr {\n    key: &quot;ksize&quot;\n    value {\n      list {\n        i: 1\n        i: 2\n        i: 2\n        i: 1\n      }\n    }\n  }\n  attr {\n    key: &quot;padding&quot;\n    value {\n      s: &quot;SAME&quot;\n    }\n  }\n  attr {\n    key: &quot;strides&quot;\n    value {\n      list {\n        i: 1\n        i: 2\n        i: 2\n        i: 1\n      }\n    }\n  }\n}\nnode {\n  name: &quot;Reshape/shape&quot;\n  op: &quot;Const&quot;\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_INT32\n    }\n  }\n  attr {\n    key: &quot;value&quot;\n    value {\n      tensor {\n        dtype: DT_INT32\n        tensor_shape {\n          dim {\n            size: 2\n          }\n        }\n        tensor_content: &quot;\\377\\377\\377\\377\\000b\\000\\000&quot;\n      }\n    }\n  }\n}\nnode {\n  name: &quot;Reshape&quot;\n  op: &quot;Reshape&quot;\n  input: &quot;pool5&quot;\n  input: &quot;Reshape/shape&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n}\nnode {\n  name: &quot;Const&quot;\n  op: &quot;Const&quot;\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;value&quot;\n    value {\n      tensor {\n        dtype: DT_FLOAT\n        tensor_shape {\n          dim {\n            size: 25088\n          }\n          dim {\n            size: 4096\n          }\n        }\n        tensor_content: &quot;<stripped 411041792 bytes>&quot;\n      }\n    }\n  }\n}\nnode {\n  name: &quot;Const/1&quot;\n  op: &quot;Const&quot;\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;value&quot;\n    value {\n      tensor {\n        dtype: DT_FLOAT\n        tensor_shape {\n          dim {\n            size: 4096\n          }\n        }\n        tensor_content: &quot;<stripped 16384 bytes>&quot;\n      }\n    }\n  }\n}\nnode {\n  name: &quot;MatMul&quot;\n  op: &quot;MatMul&quot;\n  input: &quot;Reshape&quot;\n  input: &quot;Const&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;transpose_a&quot;\n    value {\n      b: false\n    }\n  }\n  attr {\n    key: &quot;transpose_b&quot;\n    value {\n      b: false\n    }\n  }\n}\nnode {\n  name: &quot;fc6&quot;\n  op: &quot;BiasAdd&quot;\n  input: &quot;MatMul&quot;\n  input: &quot;Const/1&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;data_format&quot;\n    value {\n      s: &quot;NHWC&quot;\n    }\n  }\n}\nnode {\n  name: &quot;fc6/relu&quot;\n  op: &quot;Relu&quot;\n  input: &quot;fc6&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n}\nnode {\n  name: &quot;dropout/Shape&quot;\n  op: &quot;Shape&quot;\n  input: &quot;fc6/relu&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n}\nnode {\n  name: &quot;dropout/random/uniform/min&quot;\n  op: &quot;Const&quot;\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;value&quot;\n    value {\n      tensor {\n        dtype: DT_FLOAT\n        tensor_shape {\n        }\n        float_val: 0.0\n      }\n    }\n  }\n}\nnode {\n  name: &quot;dropout/random/uniform/range&quot;\n  op: &quot;Const&quot;\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;value&quot;\n    value {\n      tensor {\n        dtype: DT_FLOAT\n        tensor_shape {\n        }\n        float_val: 1.0\n      }\n    }\n  }\n}\nnode {\n  name: &quot;dropout/random/uniform/RandomUniform&quot;\n  op: &quot;RandomUniform&quot;\n  input: &quot;dropout/Shape&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_INT32\n    }\n  }\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;seed&quot;\n    value {\n      i: 0\n    }\n  }\n  attr {\n    key: &quot;seed2&quot;\n    value {\n      i: 0\n    }\n  }\n}\nnode {\n  name: &quot;dropout/random/uniform/mul&quot;\n  op: &quot;Mul&quot;\n  input: &quot;dropout/random/uniform/RandomUniform&quot;\n  input: &quot;dropout/random/uniform/range&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n}\nnode {\n  name: &quot;dropout/random/uniform&quot;\n  op: &quot;Add&quot;\n  input: &quot;dropout/random/uniform/mul&quot;\n  input: &quot;dropout/random/uniform/min&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n}\nnode {\n  name: &quot;dropout/add/x&quot;\n  op: &quot;Const&quot;\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;value&quot;\n    value {\n      tensor {\n        dtype: DT_FLOAT\n        tensor_shape {\n        }\n        float_val: 0.5\n      }\n    }\n  }\n}\nnode {\n  name: &quot;dropout/add&quot;\n  op: &quot;Add&quot;\n  input: &quot;dropout/add/x&quot;\n  input: &quot;dropout/random/uniform&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n}\nnode {\n  name: &quot;dropout/Floor&quot;\n  op: &quot;Floor&quot;\n  input: &quot;dropout/add&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n}\nnode {\n  name: &quot;dropout/mul/y&quot;\n  op: &quot;Const&quot;\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;value&quot;\n    value {\n      tensor {\n        dtype: DT_FLOAT\n        tensor_shape {\n        }\n        float_val: 2.0\n      }\n    }\n  }\n}\nnode {\n  name: &quot;dropout/mul&quot;\n  op: &quot;Mul&quot;\n  input: &quot;fc6/relu&quot;\n  input: &quot;dropout/mul/y&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n}\nnode {\n  name: &quot;dropout/mul/1&quot;\n  op: &quot;Mul&quot;\n  input: &quot;dropout/mul&quot;\n  input: &quot;dropout/Floor&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n}\nnode {\n  name: &quot;Reshape/1/shape&quot;\n  op: &quot;Const&quot;\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_INT32\n    }\n  }\n  attr {\n    key: &quot;value&quot;\n    value {\n      tensor {\n        dtype: DT_INT32\n        tensor_shape {\n          dim {\n            size: 2\n          }\n        }\n        tensor_content: &quot;\\377\\377\\377\\377\\000\\020\\000\\000&quot;\n      }\n    }\n  }\n}\nnode {\n  name: &quot;Reshape/1&quot;\n  op: &quot;Reshape&quot;\n  input: &quot;dropout/mul/1&quot;\n  input: &quot;Reshape/1/shape&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n}\nnode {\n  name: &quot;Const/2&quot;\n  op: &quot;Const&quot;\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;value&quot;\n    value {\n      tensor {\n        dtype: DT_FLOAT\n        tensor_shape {\n          dim {\n            size: 4096\n          }\n          dim {\n            size: 4096\n          }\n        }\n        tensor_content: &quot;<stripped 67108864 bytes>&quot;\n      }\n    }\n  }\n}\nnode {\n  name: &quot;Const/3&quot;\n  op: &quot;Const&quot;\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;value&quot;\n    value {\n      tensor {\n        dtype: DT_FLOAT\n        tensor_shape {\n          dim {\n            size: 4096\n          }\n        }\n        tensor_content: &quot;<stripped 16384 bytes>&quot;\n      }\n    }\n  }\n}\nnode {\n  name: &quot;MatMul/1&quot;\n  op: &quot;MatMul&quot;\n  input: &quot;Reshape/1&quot;\n  input: &quot;Const/2&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;transpose_a&quot;\n    value {\n      b: false\n    }\n  }\n  attr {\n    key: &quot;transpose_b&quot;\n    value {\n      b: false\n    }\n  }\n}\nnode {\n  name: &quot;fc7&quot;\n  op: &quot;BiasAdd&quot;\n  input: &quot;MatMul/1&quot;\n  input: &quot;Const/3&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;data_format&quot;\n    value {\n      s: &quot;NHWC&quot;\n    }\n  }\n}\nnode {\n  name: &quot;fc7/relu&quot;\n  op: &quot;Relu&quot;\n  input: &quot;fc7&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n}\nnode {\n  name: &quot;dropout/1/Shape&quot;\n  op: &quot;Shape&quot;\n  input: &quot;fc7/relu&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n}\nnode {\n  name: &quot;dropout/1/random_uniform/min&quot;\n  op: &quot;Const&quot;\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;value&quot;\n    value {\n      tensor {\n        dtype: DT_FLOAT\n        tensor_shape {\n        }\n        float_val: 0.0\n      }\n    }\n  }\n}\nnode {\n  name: &quot;dropout/1/random_uniform/range&quot;\n  op: &quot;Const&quot;\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;value&quot;\n    value {\n      tensor {\n        dtype: DT_FLOAT\n        tensor_shape {\n        }\n        float_val: 1.0\n      }\n    }\n  }\n}\nnode {\n  name: &quot;dropout/1/random_uniform/RandomUniform&quot;\n  op: &quot;RandomUniform&quot;\n  input: &quot;dropout/1/Shape&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_INT32\n    }\n  }\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;seed&quot;\n    value {\n      i: 0\n    }\n  }\n  attr {\n    key: &quot;seed2&quot;\n    value {\n      i: 0\n    }\n  }\n}\nnode {\n  name: &quot;dropout/1/random_uniform/mul&quot;\n  op: &quot;Mul&quot;\n  input: &quot;dropout/1/random_uniform/RandomUniform&quot;\n  input: &quot;dropout/1/random_uniform/range&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n}\nnode {\n  name: &quot;dropout/1/random_uniform&quot;\n  op: &quot;Add&quot;\n  input: &quot;dropout/1/random_uniform/mul&quot;\n  input: &quot;dropout/1/random_uniform/min&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n}\nnode {\n  name: &quot;dropout/1/add/x&quot;\n  op: &quot;Const&quot;\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;value&quot;\n    value {\n      tensor {\n        dtype: DT_FLOAT\n        tensor_shape {\n        }\n        float_val: 0.5\n      }\n    }\n  }\n}\nnode {\n  name: &quot;dropout/1/add&quot;\n  op: &quot;Add&quot;\n  input: &quot;dropout/1/add/x&quot;\n  input: &quot;dropout/1/random_uniform&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n}\nnode {\n  name: &quot;dropout/1/Floor&quot;\n  op: &quot;Floor&quot;\n  input: &quot;dropout/1/add&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n}\nnode {\n  name: &quot;dropout/1/mul/y&quot;\n  op: &quot;Const&quot;\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;value&quot;\n    value {\n      tensor {\n        dtype: DT_FLOAT\n        tensor_shape {\n        }\n        float_val: 2.0\n      }\n    }\n  }\n}\nnode {\n  name: &quot;dropout/1/mul&quot;\n  op: &quot;Mul&quot;\n  input: &quot;fc7/relu&quot;\n  input: &quot;dropout/1/mul/y&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n}\nnode {\n  name: &quot;dropout/1/mul_1&quot;\n  op: &quot;Mul&quot;\n  input: &quot;dropout/1/mul&quot;\n  input: &quot;dropout/1/Floor&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n}\nnode {\n  name: &quot;Reshape/2/shape&quot;\n  op: &quot;Const&quot;\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_INT32\n    }\n  }\n  attr {\n    key: &quot;value&quot;\n    value {\n      tensor {\n        dtype: DT_INT32\n        tensor_shape {\n          dim {\n            size: 2\n          }\n        }\n        tensor_content: &quot;\\377\\377\\377\\377\\000\\020\\000\\000&quot;\n      }\n    }\n  }\n}\nnode {\n  name: &quot;Reshape/2&quot;\n  op: &quot;Reshape&quot;\n  input: &quot;dropout/1/mul_1&quot;\n  input: &quot;Reshape/2/shape&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n}\nnode {\n  name: &quot;Const/4&quot;\n  op: &quot;Const&quot;\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;value&quot;\n    value {\n      tensor {\n        dtype: DT_FLOAT\n        tensor_shape {\n          dim {\n            size: 4096\n          }\n          dim {\n            size: 1000\n          }\n        }\n        tensor_content: &quot;<stripped 16384000 bytes>&quot;\n      }\n    }\n  }\n}\nnode {\n  name: &quot;Const/5&quot;\n  op: &quot;Const&quot;\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;value&quot;\n    value {\n      tensor {\n        dtype: DT_FLOAT\n        tensor_shape {\n          dim {\n            size: 1000\n          }\n        }\n        tensor_content: &quot;<stripped 4000 bytes>&quot;\n      }\n    }\n  }\n}\nnode {\n  name: &quot;MatMul/2&quot;\n  op: &quot;MatMul&quot;\n  input: &quot;Reshape/2&quot;\n  input: &quot;Const/4&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;transpose_a&quot;\n    value {\n      b: false\n    }\n  }\n  attr {\n    key: &quot;transpose_b&quot;\n    value {\n      b: false\n    }\n  }\n}\nnode {\n  name: &quot;fc8&quot;\n  op: &quot;BiasAdd&quot;\n  input: &quot;MatMul/2&quot;\n  input: &quot;Const/5&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;data_format&quot;\n    value {\n      s: &quot;NHWC&quot;\n    }\n  }\n}\nnode {\n  name: &quot;prob&quot;\n  op: &quot;Softmax&quot;\n  input: &quot;fc8&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n}\nnode {\n  name: &quot;init&quot;\n  op: &quot;NoOp&quot;\n}\n';
              }
            </script>
            <link rel=&quot;import&quot; href=&quot;https://tensorboard.appspot.com/tf-graph-basic.build.html&quot; onload=load()>
            <div style=&quot;height:600px&quot;>
              <tf-graph-basic id=&quot;graph0.09864523476836828&quot;></tf-graph-basic>
            </div>
        "></iframe>
        



```python
print(names)
```

    ['vgg/images', 'vgg/mul/y', 'vgg/mul', 'vgg/split/split_dim', 'vgg/split', 'vgg/sub/y', 'vgg/sub', 'vgg/sub_1/y', 'vgg/sub_1', 'vgg/sub_2/y', 'vgg/sub_2', 'vgg/concat/concat_dim', 'vgg/concat', 'vgg/conv1_1/filter', 'vgg/conv1_1/Conv2D', 'vgg/conv1_1/biases', 'vgg/conv1_1/BiasAdd', 'vgg/conv1_1/conv1_1', 'vgg/conv1_2/filter', 'vgg/conv1_2/Conv2D', 'vgg/conv1_2/biases', 'vgg/conv1_2/BiasAdd', 'vgg/conv1_2/conv1_2', 'vgg/pool1', 'vgg/conv2_1/filter', 'vgg/conv2_1/Conv2D', 'vgg/conv2_1/biases', 'vgg/conv2_1/BiasAdd', 'vgg/conv2_1/conv2_1', 'vgg/conv2_2/filter', 'vgg/conv2_2/Conv2D', 'vgg/conv2_2/biases', 'vgg/conv2_2/BiasAdd', 'vgg/conv2_2/conv2_2', 'vgg/pool2', 'vgg/conv3_1/filter', 'vgg/conv3_1/Conv2D', 'vgg/conv3_1/biases', 'vgg/conv3_1/BiasAdd', 'vgg/conv3_1/conv3_1', 'vgg/conv3_2/filter', 'vgg/conv3_2/Conv2D', 'vgg/conv3_2/biases', 'vgg/conv3_2/BiasAdd', 'vgg/conv3_2/conv3_2', 'vgg/conv3_3/filter', 'vgg/conv3_3/Conv2D', 'vgg/conv3_3/biases', 'vgg/conv3_3/BiasAdd', 'vgg/conv3_3/conv3_3', 'vgg/pool3', 'vgg/conv4_1/filter', 'vgg/conv4_1/Conv2D', 'vgg/conv4_1/biases', 'vgg/conv4_1/BiasAdd', 'vgg/conv4_1/conv4_1', 'vgg/conv4_2/filter', 'vgg/conv4_2/Conv2D', 'vgg/conv4_2/biases', 'vgg/conv4_2/BiasAdd', 'vgg/conv4_2/conv4_2', 'vgg/conv4_3/filter', 'vgg/conv4_3/Conv2D', 'vgg/conv4_3/biases', 'vgg/conv4_3/BiasAdd', 'vgg/conv4_3/conv4_3', 'vgg/pool4', 'vgg/conv5_1/filter', 'vgg/conv5_1/Conv2D', 'vgg/conv5_1/biases', 'vgg/conv5_1/BiasAdd', 'vgg/conv5_1/conv5_1', 'vgg/conv5_2/filter', 'vgg/conv5_2/Conv2D', 'vgg/conv5_2/biases', 'vgg/conv5_2/BiasAdd', 'vgg/conv5_2/conv5_2', 'vgg/conv5_3/filter', 'vgg/conv5_3/Conv2D', 'vgg/conv5_3/biases', 'vgg/conv5_3/BiasAdd', 'vgg/conv5_3/conv5_3', 'vgg/pool5', 'vgg/Reshape/shape', 'vgg/Reshape', 'vgg/Const', 'vgg/Const_1', 'vgg/MatMul', 'vgg/fc6', 'vgg/fc6_relu', 'vgg/dropout/Shape', 'vgg/dropout/random_uniform/min', 'vgg/dropout/random_uniform/range', 'vgg/dropout/random_uniform/RandomUniform', 'vgg/dropout/random_uniform/mul', 'vgg/dropout/random_uniform', 'vgg/dropout/add/x', 'vgg/dropout/add', 'vgg/dropout/Floor', 'vgg/dropout/mul/y', 'vgg/dropout/mul', 'vgg/dropout/mul_1', 'vgg/Reshape_1/shape', 'vgg/Reshape_1', 'vgg/Const_2', 'vgg/Const_3', 'vgg/MatMul_1', 'vgg/fc7', 'vgg/fc7_relu', 'vgg/dropout_1/Shape', 'vgg/dropout_1/random_uniform/min', 'vgg/dropout_1/random_uniform/range', 'vgg/dropout_1/random_uniform/RandomUniform', 'vgg/dropout_1/random_uniform/mul', 'vgg/dropout_1/random_uniform', 'vgg/dropout_1/add/x', 'vgg/dropout_1/add', 'vgg/dropout_1/Floor', 'vgg/dropout_1/mul/y', 'vgg/dropout_1/mul', 'vgg/dropout_1/mul_1', 'vgg/Reshape_2/shape', 'vgg/Reshape_2', 'vgg/Const_4', 'vgg/Const_5', 'vgg/MatMul_2', 'vgg/fc8', 'vgg/prob', 'vgg/init']


So unlike inception, which has many parallel streams and concatenation operations, this network is much like the network we've created in the last session.  A pretty basic deep convolutional network with a single stream of many convolutions, followed by adding biases, and using relu non-linearities.

<TODO: produce tesnorboard visual>

Let's grab a placeholder for the input and output of the network:


```python
x = g.get_tensor_by_name(names[0] + ':0')
softmax = g.get_tensor_by_name(names[-2] + ':0')
```

We'll grab an image preprocess, add a new dimension to make the image 4-D, then predict the label of this image just like we did with the Inception network:


```python
from skimage.data import coffee
og = coffee()
plt.imshow(og)
```




    <matplotlib.image.AxesImage at 0x1300cd710>




![png](lecture-4_files/lecture-4_112_1.png)



```python
img = vgg16.preprocess(og)
```


```python
plt.imshow(vgg16.deprocess(img))
```




    <matplotlib.image.AxesImage at 0x130274780>




![png](lecture-4_files/lecture-4_114_1.png)



```python
img_4d = img[np.newaxis]

with tf.Session(graph=g) as sess, g.device('/cpu:0'):
    res = softmax.eval(feed_dict={x: img_4d})[0]
    print([(res[idx], net['labels'][idx])
           for idx in res.argsort()[-5:][::-1]])
```

    [(0.99072862, (967, 'n07920052 espresso')), (0.0038241001, (968, 'n07930864 cup')), (0.0016883078, (868, 'n04476259 tray')), (0.0012987712, (960, 'n07836838 chocolate sauce, chocolate syrup')), (0.0011314416, (925, 'n07584110 consomme'))]


<a name="dropout"></a>
## Dropout

If I run this again, I get a different result:


```python
with tf.Session(graph=g) as sess, g.device('/cpu:0'):
    res = softmax.eval(feed_dict={x: img_4d})[0]
print([(res[idx], net['labels'][idx])
       for idx in res.argsort()[-5:][::-1]])
```

    [(0.88728428, (967, 'n07920052 espresso')), (0.077477559, (968, 'n07930864 cup')), (0.018839369, (960, 'n07836838 chocolate sauce, chocolate syrup')), (0.006374768, (925, 'n07584110 consomme')), (0.0063688005, (504, 'n03063599 coffee mug'))]


That's because this network is using something called dropout.  Basically, dropout will randomly drop connections.  This is useful because it allows multiple paths of explanations for a network.  Consider how this might be manifested in an image recognition network.  Perhaps part of the object is occluded.  We would still want the network to be able to describe the object.  That's a very useful thing to do during training to do what's called regularization.  Basically regularization is a fancy term for make sure the activations are within a certain range which I won't get into there.  It turns out there are other very good ways of performing regularization including dropping entire layers instead of indvidual neurons; or performing what's called batch normalization, which I also won't get into here.

To use the VGG network without dropout, we'll have to set the values of the dropout "keep" probability to be 1, meaning don't drop any connections:


```python
[name_i for name_i in names if 'dropout' in name_i]
```




    ['vgg/dropout/Shape',
     'vgg/dropout/random_uniform/min',
     'vgg/dropout/random_uniform/range',
     'vgg/dropout/random_uniform/RandomUniform',
     'vgg/dropout/random_uniform/mul',
     'vgg/dropout/random_uniform',
     'vgg/dropout/add/x',
     'vgg/dropout/add',
     'vgg/dropout/Floor',
     'vgg/dropout/mul/y',
     'vgg/dropout/mul',
     'vgg/dropout/mul_1',
     'vgg/dropout_1/Shape',
     'vgg/dropout_1/random_uniform/min',
     'vgg/dropout_1/random_uniform/range',
     'vgg/dropout_1/random_uniform/RandomUniform',
     'vgg/dropout_1/random_uniform/mul',
     'vgg/dropout_1/random_uniform',
     'vgg/dropout_1/add/x',
     'vgg/dropout_1/add',
     'vgg/dropout_1/Floor',
     'vgg/dropout_1/mul/y',
     'vgg/dropout_1/mul',
     'vgg/dropout_1/mul_1']



Looking at the network, it looks like there are 2 dropout layers.  Let's set these values to 1 by telling the `feed_dict` parameter.


```python
with tf.Session(graph=g) as sess, g.device('/cpu:0'):
    res = softmax.eval(feed_dict={
        x: img_4d,
        'vgg/dropout_1/random_uniform:0': [[1.0]],
        'vgg/dropout/random_uniform:0': [[1.0]]})[0]
print([(res[idx], net['labels'][idx])
       for idx in res.argsort()[-5:][::-1]])
```

    [(0.99999619, (967, 'n07920052 espresso')), (3.8687845e-06, (968, 'n07930864 cup')), (5.4153543e-10, (504, 'n03063599 coffee mug')), (3.8140457e-10, (960, 'n07836838 chocolate sauce, chocolate syrup')), (3.2201272e-10, (925, 'n07584110 consomme'))]


Let's try again to be sure:


```python
with tf.Session(graph=g) as sess, g.device('/cpu:0'):
    res = softmax.eval(feed_dict={
        x: img_4d,
        'vgg/dropout_1/random_uniform:0': [[1.0]],
        'vgg/dropout/random_uniform:0': [[1.0]]})[0]
print([(res[idx], net['labels'][idx])
       for idx in res.argsort()[-5:][::-1]])
```

    [(0.99999619, (967, 'n07920052 espresso')), (3.8687845e-06, (968, 'n07930864 cup')), (5.4153543e-10, (504, 'n03063599 coffee mug')), (3.8140457e-10, (960, 'n07836838 chocolate sauce, chocolate syrup')), (3.2201272e-10, (925, 'n07584110 consomme'))]


Great so we get the exact same probability and it works just like the Inception network!

<a name="defining-the-content-features"></a>
## Defining the Content Features

For the "content" of the image, we're going to need to know what's happening in the image at the broadest spatial scale.  Remember before when we talked about deeper layers having a wider receptive field?  We're going to use that knowledge to say that the later layers are better at representing the overall content of the image.  Let's try using the 4th layer's convolution for the determining the content:


```python
with tf.Session(graph=g) as sess, g.device('/cpu:0'):
    content_layer = 'vgg/conv4_2/conv4_2:0'
    content_features = g.get_tensor_by_name(content_layer).eval(
            session=sess,
            feed_dict={x: img_4d,
                'vgg/dropout_1/random_uniform:0': [[1.0]],
                'vgg/dropout/random_uniform:0': [[1.0]]
            })
print(content_features.shape)
```

    (1, 28, 28, 512)


<a name="defining-the-style-features"></a>
## Defining the Style Features

Great.  We now have a tensor describing the content of our original image.  We're going to stylize it now using another image.  We'll need to grab another image.  I'm going to use Hieronymous Boschs's famous still life painting of sunflowers.


```python
filepath = utils.download('https://upload.wikimedia.org/wikipedia/commons/thumb/a/ae/El_jard%C3%ADn_de_las_Delicias%2C_de_El_Bosco.jpg/640px-El_jard%C3%ADn_de_las_Delicias%2C_de_El_Bosco.jpg')
```


```python
# Note: Unlike in the lecture, I've cropped the image a bit as the borders took over too much...
style_og = plt.imread(filepath)[15:-15, 190:-190, :]
plt.imshow(style_og)
```




    <matplotlib.image.AxesImage at 0x1303137b8>




![png](lecture-4_files/lecture-4_128_1.png)


We'll need to preprocess it just like we've done with the image of the espresso:


```python
style_img = vgg16.preprocess(style_og)
style_img_4d = style_img[np.newaxis]
```

And for fun let's see what VGG thinks of it:


```python
with tf.Session(graph=g) as sess, g.device('/cpu:0'):
    res = softmax.eval(
        feed_dict={
            x: style_img_4d,
            'vgg/dropout_1/random_uniform:0': [[1.0]],
            'vgg/dropout/random_uniform:0': [[1.0]]})[0]
print([(res[idx], net['labels'][idx])
       for idx in res.argsort()[-5:][::-1]])
```

    [(0.99336135, (611, 'n03598930 jigsaw puzzle')), (0.0052440548, (868, 'n04476259 tray')), (0.0011667105, (591, 'n03485794 handkerchief, hankie, hanky, hankey')), (0.00011143424, (865, 'n04462240 toyshop')), (4.7309215e-05, (917, 'n06596364 comic book'))]


So it's not great.  It looks like it thinks it's a jigsaw puzzle. What we're going to do is find features of this image at different layers in the network.


```python
style_layers = ['vgg/conv1_1/conv1_1:0',
                'vgg/conv2_1/conv2_1:0',
                'vgg/conv3_1/conv3_1:0',
                'vgg/conv4_1/conv4_1:0',
                'vgg/conv5_1/conv5_1:0']
style_activations = []

with tf.Session(graph=g) as sess, g.device('/cpu:0'):
    for style_i in style_layers:
        style_activation_i = g.get_tensor_by_name(style_i).eval(
            feed_dict={
                x: style_img_4d,
                'vgg/dropout_1/random_uniform:0': [[1.0]],
                'vgg/dropout/random_uniform:0': [[1.0]]})
        style_activations.append(style_activation_i)
```

Instead of using the raw activations of these layers, what the authors of the StyleNet paper suggest is to use the Gram activation of the layers instead, which mathematically is expressed as the matrix transpose multiplied by itself.  The intuition behind this process is that it measures the similarity between every feature of a matrix.  Or put another way, it is saying how often certain features appear together.

This would seem useful for "style", as what we're trying to do is see what's similar across the image.  To get every feature, we're going to have to reshape our N x H x W x C matrix to have every pixel belonging to each feature in a single column.  This way, when we take the transpose and multiply it against itself, we're measuring the shared direction of every feature with every other feature.  Intuitively, this would be useful as a measure of style, since we're measuring whats in common across all pixels and features.


```python
style_features = []
for style_activation_i in style_activations:
    s_i = np.reshape(style_activation_i, [-1, style_activation_i.shape[-1]])
    gram_matrix = np.matmul(s_i.T, s_i) / s_i.size
    style_features.append(gram_matrix.astype(np.float32))
```

<a name="remapping-the-input"></a>
## Remapping the Input

So now we have a collection of "features", which are basically the activations of our sunflower image at different layers.  We're now going to try and make our coffee image have the same style as this image by trying to enforce these features on the image.  Let's take a look at how we can do that.

We're going to need to create a new graph which replaces the input of the original VGG network with a variable which can be optimized.  So instead of having a placeholder as input to the network, we're going to tell tensorflow that we want this to be a `tf.Variable`.  That's because we're going to try to optimize what this is, based on the objectives which we'll soon create.


```python
reset_default_graph()
g = tf.Graph()
```

And now we'll load up the VGG network again, except unlike before, we're going to map the input of this network to a new variable randomly initialized to our content image.  Alternatively, we could initialize this image noise to see a different result.


```python
net = vgg16.get_vgg_model()
# net_input = tf.get_variable(
#    name='input',
#    shape=(1, 224, 224, 3),
#    dtype=tf.float32,
#    initializer=tf.random_normal_initializer(
#        mean=np.mean(img), stddev=np.std(img)))
with tf.Session(graph=g) as sess, g.device('/cpu:0'):
    net_input = tf.Variable(img_4d)
    tf.import_graph_def(
        net['graph_def'],
        name='vgg',
        input_map={'images:0': net_input})
```

Let's take a look at the graph now:


```python
names = [op.name for op in g.get_operations()]
print(names)
```

    ['Variable/initial_value', 'Variable', 'Variable/Assign', 'Variable/read', 'vgg/images', 'vgg/mul/y', 'vgg/mul', 'vgg/split/split_dim', 'vgg/split', 'vgg/sub/y', 'vgg/sub', 'vgg/sub_1/y', 'vgg/sub_1', 'vgg/sub_2/y', 'vgg/sub_2', 'vgg/concat/concat_dim', 'vgg/concat', 'vgg/conv1_1/filter', 'vgg/conv1_1/Conv2D', 'vgg/conv1_1/biases', 'vgg/conv1_1/BiasAdd', 'vgg/conv1_1/conv1_1', 'vgg/conv1_2/filter', 'vgg/conv1_2/Conv2D', 'vgg/conv1_2/biases', 'vgg/conv1_2/BiasAdd', 'vgg/conv1_2/conv1_2', 'vgg/pool1', 'vgg/conv2_1/filter', 'vgg/conv2_1/Conv2D', 'vgg/conv2_1/biases', 'vgg/conv2_1/BiasAdd', 'vgg/conv2_1/conv2_1', 'vgg/conv2_2/filter', 'vgg/conv2_2/Conv2D', 'vgg/conv2_2/biases', 'vgg/conv2_2/BiasAdd', 'vgg/conv2_2/conv2_2', 'vgg/pool2', 'vgg/conv3_1/filter', 'vgg/conv3_1/Conv2D', 'vgg/conv3_1/biases', 'vgg/conv3_1/BiasAdd', 'vgg/conv3_1/conv3_1', 'vgg/conv3_2/filter', 'vgg/conv3_2/Conv2D', 'vgg/conv3_2/biases', 'vgg/conv3_2/BiasAdd', 'vgg/conv3_2/conv3_2', 'vgg/conv3_3/filter', 'vgg/conv3_3/Conv2D', 'vgg/conv3_3/biases', 'vgg/conv3_3/BiasAdd', 'vgg/conv3_3/conv3_3', 'vgg/pool3', 'vgg/conv4_1/filter', 'vgg/conv4_1/Conv2D', 'vgg/conv4_1/biases', 'vgg/conv4_1/BiasAdd', 'vgg/conv4_1/conv4_1', 'vgg/conv4_2/filter', 'vgg/conv4_2/Conv2D', 'vgg/conv4_2/biases', 'vgg/conv4_2/BiasAdd', 'vgg/conv4_2/conv4_2', 'vgg/conv4_3/filter', 'vgg/conv4_3/Conv2D', 'vgg/conv4_3/biases', 'vgg/conv4_3/BiasAdd', 'vgg/conv4_3/conv4_3', 'vgg/pool4', 'vgg/conv5_1/filter', 'vgg/conv5_1/Conv2D', 'vgg/conv5_1/biases', 'vgg/conv5_1/BiasAdd', 'vgg/conv5_1/conv5_1', 'vgg/conv5_2/filter', 'vgg/conv5_2/Conv2D', 'vgg/conv5_2/biases', 'vgg/conv5_2/BiasAdd', 'vgg/conv5_2/conv5_2', 'vgg/conv5_3/filter', 'vgg/conv5_3/Conv2D', 'vgg/conv5_3/biases', 'vgg/conv5_3/BiasAdd', 'vgg/conv5_3/conv5_3', 'vgg/pool5', 'vgg/Reshape/shape', 'vgg/Reshape', 'vgg/Const', 'vgg/Const_1', 'vgg/MatMul', 'vgg/fc6', 'vgg/fc6_relu', 'vgg/dropout/Shape', 'vgg/dropout/random_uniform/min', 'vgg/dropout/random_uniform/range', 'vgg/dropout/random_uniform/RandomUniform', 'vgg/dropout/random_uniform/mul', 'vgg/dropout/random_uniform', 'vgg/dropout/add/x', 'vgg/dropout/add', 'vgg/dropout/Floor', 'vgg/dropout/mul/y', 'vgg/dropout/mul', 'vgg/dropout/mul_1', 'vgg/Reshape_1/shape', 'vgg/Reshape_1', 'vgg/Const_2', 'vgg/Const_3', 'vgg/MatMul_1', 'vgg/fc7', 'vgg/fc7_relu', 'vgg/dropout_1/Shape', 'vgg/dropout_1/random_uniform/min', 'vgg/dropout_1/random_uniform/range', 'vgg/dropout_1/random_uniform/RandomUniform', 'vgg/dropout_1/random_uniform/mul', 'vgg/dropout_1/random_uniform', 'vgg/dropout_1/add/x', 'vgg/dropout_1/add', 'vgg/dropout_1/Floor', 'vgg/dropout_1/mul/y', 'vgg/dropout_1/mul', 'vgg/dropout_1/mul_1', 'vgg/Reshape_2/shape', 'vgg/Reshape_2', 'vgg/Const_4', 'vgg/Const_5', 'vgg/MatMul_2', 'vgg/fc8', 'vgg/prob', 'vgg/init']


So notice now the first layers of the network have everything prefixed by input, our new variable which we've just created.  This will initialize a variable with the content image upon initialization.  And then as we run whatever our optimizer ends up being, it will slowly become the a stylized image.

<a name="defining-the-content-loss"></a>
## Defining the Content Loss

We now need to define a loss function which tries to optimize the distance between the net's output at our content layer, and the content features which we have built from the coffee image:


```python
with tf.Session(graph=g) as sess, g.device('/cpu:0'):
    content_loss = tf.nn.l2_loss((g.get_tensor_by_name(content_layer) -
                                 content_features) /
                                 content_features.size)
```

<a name="defining-the-style-loss"></a>
## Defining the Style Loss

For our style loss, we'll compute the gram matrix of the current network output, and then measure the l2 loss with our precomputed style image's gram matrix.  So most of this is the same as when we compute the gram matrix for the style image, except now, we're doing this in tensorflow's computational graph, so that we can later connect these operations to an optimizer. Refer to the lecture for a more in depth explanation of this.


```python
with tf.Session(graph=g) as sess, g.device('/cpu:0'):
    style_loss = np.float32(0.0)
    for style_layer_i, style_gram_i in zip(style_layers, style_features):
        layer_i = g.get_tensor_by_name(style_layer_i)
        layer_shape = layer_i.get_shape().as_list()
        layer_size = layer_shape[1] * layer_shape[2] * layer_shape[3]
        layer_flat = tf.reshape(layer_i, [-1, layer_shape[3]])
        gram_matrix = tf.matmul(tf.transpose(layer_flat), layer_flat) / layer_size
        style_loss = tf.add(style_loss, tf.nn.l2_loss((gram_matrix - style_gram_i) / np.float32(style_gram_i.size)))
```

<a name="defining-the-total-variation-loss"></a>
## Defining the Total Variation Loss

Lastly, we'll create a third loss value which will simply measure the difference between neighboring pixels.  By including this as a loss, we're saying that we want neighboring pixels to be similar.


```python
def total_variation_loss(x):
    h, w = x.get_shape().as_list()[1], x.get_shape().as_list()[1]
    dx = tf.square(x[:, :h-1, :w-1, :] - x[:, :h-1, 1:, :])
    dy = tf.square(x[:, :h-1, :w-1, :] - x[:, 1:, :w-1, :])
    return tf.reduce_sum(tf.pow(dx + dy, 1.25))

with tf.Session(graph=g) as sess, g.device('/cpu:0'):
    tv_loss = total_variation_loss(net_input)
```

<a name="training"></a>
## Training

With both content and style losses, we can combine the two, optimizing our loss function, and creating a stylized coffee cup.


```python
with tf.Session(graph=g) as sess, g.device('/cpu:0'):
    loss = 0.1 * content_loss + 5.0 * style_loss + 0.01 * tv_loss
    optimizer = tf.train.AdamOptimizer(0.01).minimize(loss)
```


```python
with tf.Session(graph=g) as sess, g.device('/cpu:0'):
    sess.run(tf.initialize_all_variables())
    # map input to noise
    n_iterations = 100
    og_img = net_input.eval()
    imgs = []
    for it_i in range(n_iterations):
        _, this_loss, synth = sess.run([optimizer, loss, net_input],
                feed_dict={
                    'vgg/dropout_1/random_uniform:0':
                        np.ones(g.get_tensor_by_name(
                        'vgg/dropout_1/random_uniform:0').get_shape().as_list()),
                    'vgg/dropout/random_uniform:0':
                        np.ones(g.get_tensor_by_name(
                        'vgg/dropout/random_uniform:0').get_shape().as_list())})
        print("%d: %f, (%f - %f)" %
            (it_i, this_loss, np.min(synth), np.max(synth)))
        if it_i % 5 == 0:
            imgs.append(np.clip(synth[0], 0, 1))
            fig, ax = plt.subplots(1, 3, figsize=(22, 5))
            ax[0].imshow(vgg16.deprocess(img))
            ax[0].set_title('content image')
            ax[1].imshow(vgg16.deprocess(style_img))
            ax[1].set_title('style image')
            ax[2].set_title('current synthesis')
            ax[2].imshow(vgg16.deprocess(synth[0]))
            plt.show()
            fig.canvas.draw()
    gif.build_gif(imgs, saveto='stylenet-bosch.gif')
```

    0: 47.247391, (-0.010000 - 1.010000)



![png](lecture-4_files/lecture-4_151_1.png)


    1: 41.527805, (-0.019984 - 1.020013)
    2: 36.726242, (-0.029970 - 1.030036)
    3: 33.132301, (-0.039943 - 1.040036)
    4: 30.825277, (-0.049814 - 1.049982)
    5: 29.472797, (-0.059614 - 1.059739)



![png](lecture-4_files/lecture-4_151_3.png)


    6: 28.482155, (-0.069084 - 1.069270)
    7: 27.398533, (-0.078122 - 1.078805)
    8: 26.078959, (-0.086861 - 1.088219)
    9: 24.599209, (-0.096156 - 1.097538)
    10: 23.106552, (-0.104688 - 1.106785)



![png](lecture-4_files/lecture-4_151_5.png)


    11: 21.728117, (-0.112565 - 1.116050)
    12: 20.532927, (-0.121970 - 1.125387)
    13: 19.527969, (-0.132242 - 1.134813)
    14: 18.680058, (-0.142458 - 1.144342)
    15: 17.944118, (-0.152462 - 1.153988)



![png](lecture-4_files/lecture-4_151_7.png)


    16: 17.281269, (-0.162253 - 1.163744)
    17: 16.672457, (-0.171585 - 1.173564)
    18: 16.113823, (-0.179913 - 1.183393)
    19: 15.612597, (-0.186531 - 1.193166)
    20: 15.175312, (-0.190814 - 1.202783)



![png](lecture-4_files/lecture-4_151_9.png)


    21: 14.799221, (-0.193791 - 1.212068)
    22: 14.471707, (-0.202589 - 1.220960)
    23: 14.174005, (-0.210879 - 1.229360)
    24: 13.889018, (-0.218521 - 1.237175)
    25: 13.608907, (-0.225438 - 1.244390)



![png](lecture-4_files/lecture-4_151_11.png)


    26: 13.335546, (-0.231610 - 1.251005)
    27: 13.076498, (-0.237120 - 1.257057)
    28: 12.839410, (-0.242065 - 1.262578)
    29: 12.627560, (-0.246453 - 1.267609)
    30: 12.438450, (-0.250359 - 1.272232)



![png](lecture-4_files/lecture-4_151_13.png)


    31: 12.266417, (-0.253853 - 1.276482)
    32: 12.105091, (-0.256894 - 1.280326)
    33: 11.950727, (-0.264070 - 1.283617)
    34: 11.802845, (-0.273506 - 1.286333)
    35: 11.663670, (-0.282830 - 1.291337)



![png](lecture-4_files/lecture-4_151_15.png)


    36: 11.535501, (-0.291971 - 1.297106)
    37: 11.419305, (-0.300841 - 1.302230)
    38: 11.313501, (-0.309364 - 1.306639)
    39: 11.214956, (-0.317482 - 1.310291)
    40: 11.120373, (-0.325157 - 1.313208)



![png](lecture-4_files/lecture-4_151_17.png)


    41: 11.028181, (-0.332392 - 1.315448)
    42: 10.938995, (-0.339203 - 1.317093)
    43: 10.854811, (-0.345624 - 1.318280)
    44: 10.776907, (-0.351696 - 1.319143)
    45: 10.705124, (-0.357446 - 1.319765)



![png](lecture-4_files/lecture-4_151_19.png)


    46: 10.638214, (-0.362892 - 1.320251)
    47: 10.574605, (-0.368022 - 1.320665)
    48: 10.513269, (-0.372818 - 1.321053)
    49: 10.454495, (-0.377236 - 1.321387)
    50: 10.398926, (-0.381224 - 1.321625)



![png](lecture-4_files/lecture-4_151_21.png)


    51: 10.346942, (-0.384715 - 1.322742)
    52: 10.298336, (-0.387667 - 1.327233)
    53: 10.252294, (-0.390061 - 1.331752)
    54: 10.207791, (-0.391895 - 1.339223)
    55: 10.164473, (-0.393215 - 1.346594)



![png](lecture-4_files/lecture-4_151_23.png)


    56: 10.122510, (-0.394063 - 1.353848)
    57: 10.082338, (-0.394505 - 1.361039)
    58: 10.044279, (-0.394610 - 1.368173)
    59: 10.008223, (-0.394434 - 1.375257)
    60: 9.973778, (-0.394023 - 1.382314)



![png](lecture-4_files/lecture-4_151_25.png)


    61: 9.940480, (-0.393409 - 1.389303)
    62: 9.908276, (-0.392608 - 1.396200)
    63: 9.877314, (-0.391608 - 1.402997)
    64: 9.847815, (-0.390402 - 1.409703)
    65: 9.819736, (-0.388995 - 1.416255)



![png](lecture-4_files/lecture-4_151_27.png)


    66: 9.792754, (-0.387381 - 1.422586)
    67: 9.766596, (-0.385586 - 1.428678)
    68: 9.741144, (-0.384339 - 1.434477)
    69: 9.716408, (-0.396202 - 1.439959)
    70: 9.692583, (-0.408149 - 1.445198)



![png](lecture-4_files/lecture-4_151_29.png)


    71: 9.669689, (-0.420201 - 1.450209)
    72: 9.647569, (-0.432363 - 1.455014)
    73: 9.626028, (-0.444627 - 1.459660)
    74: 9.605102, (-0.456983 - 1.464140)
    75: 9.584824, (-0.469422 - 1.468347)



![png](lecture-4_files/lecture-4_151_31.png)


    76: 9.565211, (-0.481931 - 1.472192)
    77: 9.546281, (-0.494502 - 1.475690)
    78: 9.527858, (-0.507132 - 1.478889)
    79: 9.509901, (-0.519812 - 1.481811)
    80: 9.492346, (-0.532554 - 1.484594)



![png](lecture-4_files/lecture-4_151_33.png)


    81: 9.475199, (-0.545363 - 1.487275)
    82: 9.458512, (-0.558239 - 1.489857)
    83: 9.442279, (-0.571179 - 1.492366)
    84: 9.426536, (-0.584182 - 1.494845)
    85: 9.411205, (-0.597234 - 1.497312)



![png](lecture-4_files/lecture-4_151_35.png)


    86: 9.396239, (-0.610334 - 1.501728)
    87: 9.381626, (-0.623471 - 1.509634)
    88: 9.367407, (-0.636638 - 1.517822)
    89: 9.353575, (-0.649831 - 1.525973)
    90: 9.340040, (-0.663042 - 1.533923)



![png](lecture-4_files/lecture-4_151_37.png)


    91: 9.326790, (-0.676278 - 1.541619)
    92: 9.313822, (-0.689538 - 1.549024)
    93: 9.301157, (-0.702835 - 1.556171)
    94: 9.288795, (-0.716173 - 1.563234)
    95: 9.276705, (-0.729558 - 1.570207)



![png](lecture-4_files/lecture-4_151_39.png)


    96: 9.264882, (-0.742992 - 1.577089)
    97: 9.253311, (-0.756464 - 1.583826)
    98: 9.241951, (-0.769972 - 1.590389)
    99: 9.230789, (-0.783518 - 1.596775)



![png](lecture-4_files/lecture-4_151_41.png)



```python
ipyd.Image(url='stylenet-bosch.gif', height=300, width=300)
```




<img src="stylenet-bosch.gif" width="300" height="300"/>



We can play with a lot of the parameters involved to produce wildly different results.  There are also a lot of extensions to what I've presented here currently in the literature including incorporating structure, temporal constraints, variational constraints, and other regularizing methods including making use of the activations in the content image to help infer what features in the gram matrix are relevant.

There is also no reason I can see why this approach wouldn't work with using different sets of layers or different networks entirely such as the Inception network we started with in this session.  Perhaps after exploring deep representations a bit more, you might find intuition towards which networks, layers, or neurons in particular represent the aspects of the style you want to bring out.  You might even try blending different sets of neurons to produce interesting results.  Play with different motions.  Try blending the results as you produce the deep dream with other content.

Also, there is no reason you have to start with an image of noise, or an image of the content.  Perhaps you can start with an entirely different image which tries to reflect the process you are interested in.  There are also a lot of interesting published extensions to this technique including image analogies, neural doodle, incorporating structure, and incorporating temporal losses from optical flow to stylize video.

There is certainly a lot of room to explore within technique.  A good starting place for the possibilities with the basic version of style net I've shown here is Kyle McDonald's Style Studies:

http://www.kylemcdonald.net/stylestudies/

If you find other interesting applications of the technique, feel free to post them on the forums.

<TODO: Neural Doodle, Semantic Style Transfer>

<a name="homework"></a>
# Homework

* GIF of deep dream
* GIF of stylenet
* Combine a Deep Dream and a Style Net together.

- Peer assessed
- Gallery commenting

<a name="reading"></a>
# Reading

Matthew D Zeiler, Rob Fergus.  Visualizing and Understanding Convolutional Networks.  2013.
https://arxiv.org/abs/1311.2901

Karen Simonyan, Andrea Vedaldi, Andrew Zisserman.  Deep Inside Convolutional Networks: Visualising Image Classification Models and Saliency Maps. 2013.

Aravindh Mahendran, Andrea Vedaldi.  Understanding Deep Image Representations by Inverting Them.  2014.
https://arxiv.org/abs/1412.0035

Mordvintsev, Alexander; Olah, Christopher; Tyka, Mike (2015). "Inceptionism: Going Deeper into Neural Networks". Google Research. Archived from the original on 2015-07-03.

Szegedy, Christian; Liu, Wei; Jia, Yangqing; Sermanet, Pierre; Reed, Scott; Anguelov, Dragomir; Erhan, Dumitru; Vanhoucke, Vincent; Rabinovich, Andrew (2014). "Going Deeper with Convolutions". Computing Research Repository. arXiv:1409.4842.

Leon A. Gatys, Alexander S. Ecker, Matthias Bethge. 
A Neural Algorithm of Artistic Style.  2015.  https://arxiv.org/abs/1508.06576

Texture Networks.
http://jmlr.org/proceedings/papers/v48/ulyanov16.pdf

Jost Tobias Springenberg, Alexey Dosovitskiy, Thomas Brox, Martin Riedmiller.  Striving for Simplicity: The All Convolutional Net.  2015.
https://arxiv.org/abs/1412.6806

Yosinski, J., Clune, J., Nguyen, A., Fuchs, T., Lipson, H.  Understanding Neural Networks Through Deep Visualization.  ICML.  2015.
http://yosinski.com/media/papers/Yosinski__2015__ICML_DL__Understanding_Neural_Networks_Through_Deep_Visualization__.pdf
