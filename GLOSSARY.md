# Table of Contents

<!-- MarkdownTOC autolink=true -->

- [1x1 Convolutions](#1x1-convolutions)
- [2-D Gaussian Kernel](#2-d-gaussian-kernel)
- [Activation Function](#activation-function)
- [Autoencoders](#autoencoders)
- [Back-propagation](#back-propagation)
- [Batch Dimension](#batch-dimension)
- [Batch Normalization](#batch-normalization)
- [Batches](#batches)
- [Blur](#blur)
- [Celeb Net](#celeb-net)
- [Char-RNN](#char-rnn)
- [Character Language Model](#character-language-model)
- [Checkpoint](#checkpoint)
- [Classification Network](#classification-network)
- [Clip](#clip)
- [Content Features](#content-features)
- [Content Loss](#content-loss)
- [Context Managers](#context-managers)
- [Convolution](#convolution)
- [Convolutional Autoencoder](#convolutional-autoencoder)
- [Convolutional Networks](#convolutional-networks)
- [Convolve](#convolve)
- [Cost](#cost)
- [Dataset](#dataset)
- [DCGAN](#dcgan)
- [Decoder](#decoder)
- [Deep Convolutional Networks](#deep-convolutional-networks)
- [Deep Dream](#deep-dream)
- [Deep Dreaming](#deep-dreaming)
- [Deep Learning vs. Machine Learning](#deep-learning-vs-machine-learning)
- [Denoising Autoencoder](#denoising-autoencoder)
- [Deprocessing](#deprocessing)
- [Deviation](#deviation)
- [DRAW](#draw)
- [Dropout](#dropout)
- [Embedding](#embedding)
- [Encoder](#encoder)
- [Equilibrium](#equilibrium)
- [Error](#error)
- [Filter](#filter)
- [Fully Connected](#fully-connected)
- [GAN](#gan)
- [Gaussian](#gaussian)
- [Gaussian Kernel](#gaussian-kernel)
- [Generalized Matrix Multiplication](#generalized-matrix-multiplication)
- [Generative Adversarial Networks](#generative-adversarial-networks)
- [Gradient](#gradient)
- [Gradient Descent](#gradient-descent)
- [Graph Definition](#graph-definition)
- [Graphs](#graphs)
- [GRU](#gru)
- [Guided Hallucinations](#guided-hallucinations)
- [Histogram Equalization](#histogram-equalization)
- [Histograms](#histograms)
- [Hyperparameters](#hyperparameters)
- [Image Inpainting](#image-inpainting)
- [Image Labels](#image-labels)
- [Inception Module](#inception-module)
- [Inception Network](#inception-network)
- [Inference](#inference)
- [Input's Representation](#inputs-representation)
- [Invariances](#invariances)
- [LAPGAN](#lapgan)
- [Laplacian Pyramid](#laplacian-pyramid)
- [Latent Encoding](#latent-encoding)
- [Latent Feature Arithmetic](#latent-feature-arithmetic)
- [Latent-Space](#latent-space)
- [Learning From Data](#learning-from-data)
- [Learning Rate](#learning-rate)
- [Loading a Pretrained Network](#loading-a-pretrained-network)
- [Local Minima/Optima](#local-minimaoptima)
- [Loss](#loss)
- [LSTM](#lstm)
- [Matrix](#matrix)
- [Matrix Multiplication](#matrix-multiplication)
- [Mean](#mean)
- [Mini Batch Gradient Descent](#mini-batch-gradient-descent)
- [MNIST](#mnist)
- [Models](#models)
- [Network](#network)
- [Network Labels](#network-labels)
- [Neural Network](#neural-network)
- [Nonlinearities](#nonlinearities)
- [Normalization](#normalization)
- [Objective](#objective)
- [One-Hot Encoding](#one-hot-encoding)
- [Operations](#operations)
- [Optimizers](#optimizers)
- [Over vs. Underfitting](#over-vs-underfitting)
- [Preprocess](#preprocess)
- [Preprocessing](#preprocessing)
- [Pretrained Networks](#pretrained-networks)
- [Priming](#priming)
- [Probabilistic Sampling](#probabilistic-sampling)
- [Protobuf](#protobuf)
- [Recurrent Neural Networks](#recurrent-neural-networks)
- [Regression](#regression)
- [Reinforcement Learning](#reinforcement-learning)
- [ReLu](#relu)
- [RNN](#rnn)
- [Sessions](#sessions)
- [Sigmoid](#sigmoid)
- [Softmax Layer](#softmax-layer)
- [Standard Deviation](#standard-deviation)
- [Stochastic](#stochastic)
- [Style Features](#style-features)
- [Style Loss](#style-loss)
- [Style Net](#style-net)
- [Supervised Learning](#supervised-learning)
- [TanH](#tanh)
- [Temperature](#temperature)
- [Tensor](#tensor)
- [Tensor Shapes](#tensor-shapes)
- [Tensorboard](#tensorboard)
- [Tensorflow Basics](#tensorflow-basics)
- [Tensors](#tensors)
- [Testing](#testing)
- [Total Variation Loss](#total-variation-loss)
- [Training](#training)
- [Training Parameters](#training-parameters)
- [Training vs. Testing](#training-vs-testing)
- [Transpose](#transpose)
- [Unsupervised Learning](#unsupervised-learning)
- [Unsupervised vs. Supervised Learning](#unsupervised-vs-supervised-learning)
- [VAEGAN](#vaegan)
- [Variable](#variable)
- [Variance](#variance)
- [Variational Auto-Encoding Generative Adversarial Network](#variational-auto-encoding-generative-adversarial-network)
- [Variational Autoencoders](#variational-autoencoders)
- [Variational Layer](#variational-layer)
- [Vector](#vector)
- [VGG Network](#vgg-network)

<!-- /MarkdownTOC -->


<a name="1x1-convolutions"></a>
# 1x1 Convolutions

This defines an operation where the height and width of the kernel of a [convolution](#convolution) operation are set to 1.  This is useful because the depth dimension of the convolution operation can still be used to reduce the dimensionality (or increase it).  So for instance, if we have batch number of 100 x 100 images w/ 3 color channels, we can define a 1x1 convolution which reduces the 3 color channels to just 1 channel of information.  This is often applied before a much more expensive operation to reduce the number of overall parameters.

<a name="2-d-gaussian-kernel"></a>
# 2-D Gaussian Kernel

A Gaussian Kernel in 2-dimensions has its peak in the middle and curves outwards.  The image below depicts a 1-dimensional Gaussian.

![imgs/1d-gaussian.png](imgs/1d-gaussian)

When matrix multiplied with the transpose of itself, the 1-d Gaussian can be depicted in 2-dimensions as such:

![imgs/2d-gaussian.png](imgs/2d-gaussian)

<a name="activation-function"></a>
# Activation Function

The activation function, also known as the non-linearity, or sometimes transfer function, describes the non-linear operation in a Neural Network.  Typical activation functions include the [sigmoid](#sigmoid), [TanH](#tanh), or [ReLu](#relu).

<a name="autoencoders"></a>
# Autoencoders

An autoencoder describes a network which [encodes](#encoder) its input to some [latent encoding](#latent-encoding) layer of smaller dimensions, and then [decodes](#decoder) this [latent layer](#latent-layer) back to the original input space dimensions.  The purpose of such a network is usually to compress the information in a large dataset such that the inner most, or the layer just following the encoder, is capable of retaining as much of the information necessary to reconstitute the original dataset.  For instance, an image of 256 x 256 x 3 dimensions may be encoded to merely 2 values describing any image's latent encoding. The decoder is then capable of taking these 2 values and creating an image resembling the original image, depending on how well the network is trained/performs.

<a name="back-propagation"></a>
# Back-propagation

This describes the process of the backwards propagation of the training signal, or error, from a neural network, to each of the gradients in a network using the [chain rule of calculus](https://en.wikipedia.org/wiki/Chain_rule).  This process is used with an optimization technique such as Gradient Descent.

<a name="batch-dimension"></a>
# Batch Dimension

<a name="batch-normalization"></a>
# Batch Normalization

<a name="batches"></a>
# Batches

<a name="blur"></a>
# Blur

<a name="celeb-net"></a>
# Celeb Net

<a name="char-rnn"></a>
# Char-RNN

<a name="character-language-model"></a>
# Character Language Model

<a name="checkpoint"></a>
# Checkpoint

<a name="classification-network"></a>
# Classification Network

<a name="clip"></a>
# Clip

<a name="content-features"></a>
# Content Features

<a name="content-loss"></a>
# Content Loss

<a name="context-managers"></a>
# Context Managers

<a name="convolution"></a>
# Convolution

<a name="convolutional-autoencoder"></a>
# Convolutional Autoencoder

<a name="convolutional-networks"></a>
# Convolutional Networks

<a name="convolve"></a>
# Convolve

<a name="cost"></a>
# Cost

<a name="dataset"></a>
# Dataset

<a name="dcgan"></a>
# DCGAN

<a name="decoder"></a>
# Decoder

<a name="deep-convolutional-networks"></a>
# Deep Convolutional Networks

<a name="deep-dream"></a>
# Deep Dream

<a name="deep-dreaming"></a>
# Deep Dreaming

<a name="deep-learning-vs-machine-learning"></a>
# Deep Learning vs. Machine Learning

Deep Learning is a type of Machine Learning algorithm that uses Neural Networks to learn. The type of learning is "Deep" because it is composed of many layers of Neural Networks.

<a name="denoising-autoencoder"></a>
# Denoising Autoencoder

<a name="deprocessing"></a>
# Deprocessing

<a name="deviation"></a>
# Deviation

<a name="draw"></a>
# DRAW

<a name="dropout"></a>
# Dropout

<a name="embedding"></a>
# Embedding

<a name="encoder"></a>
# Encoder

<a name="equilibrium"></a>
# Equilibrium

<a name="error"></a>
# Error

<a name="filter"></a>
# Filter

<a name="fully-connected"></a>
# Fully Connected

<a name="gan"></a>
# GAN

<a name="gaussian"></a>
# Gaussian

<a name="gaussian-kernel"></a>
# Gaussian Kernel

<a name="generalized-matrix-multiplication"></a>
# Generalized Matrix Multiplication

<a name="generative-adversarial-networks"></a>
# Generative Adversarial Networks

<a name="gradient"></a>
# Gradient

<a name="gradient-descent"></a>
# Gradient Descent

<a name="graph-definition"></a>
# Graph Definition

<a name="graphs"></a>
# Graphs

<a name="gru"></a>
# GRU

<a name="guided-hallucinations"></a>
# Guided Hallucinations

<a name="histogram-equalization"></a>
# Histogram Equalization

<a name="histograms"></a>
# Histograms

<a name="hyperparameters"></a>
# Hyperparameters

<a name="image-inpainting"></a>
# Image Inpainting

<a name="image-labels"></a>
# Image Labels

<a name="inception-module"></a>
# Inception Module

<a name="inception-network"></a>
# Inception Network

<a name="inference"></a>
# Inference

<a name="inputs-representation"></a>
# Input's Representation

<a name="invariances"></a>
# Invariances

We usually describe the factors which represent something "invariances". That just means we are trying not to vary based on some factor. We are invariant to it. For instance, an object could appear to one side of an image, or another. We call that translation invariance. Or it could be from one angle or another. That's called rotation invariance. Or it could be closer to the camera, or farther. and That would be scale invariance. There are plenty of other types of invariances, such as perspective or brightness or exposure in the case of photographic images.  Many researchers/scientists/philosophers will have other definitions of this term.

<a name="lapgan"></a>
# LAPGAN

<a name="laplacian-pyramid"></a>
# Laplacian Pyramid

<a name="latent-encoding"></a>
# Latent Encoding

<a name="latent-feature-arithmetic"></a>
# Latent Feature Arithmetic

<a name="latent-space"></a>
# Latent-Space

<a name="learning-from-data"></a>
# Learning From Data

<a name="learning-rate"></a>
# Learning Rate

<a name="loading-a-pretrained-network"></a>
# Loading a Pretrained Network

<a name="local-minimaoptima"></a>
# Local Minima/Optima

<a name="loss"></a>
# Loss

<a name="lstm"></a>
# LSTM

<a name="matrix"></a>
# Matrix

<a name="matrix-multiplication"></a>
# Matrix Multiplication

<a name="mean"></a>
# Mean

<a name="mini-batch-gradient-descent"></a>
# Mini Batch Gradient Descent

<a name="mnist"></a>
# MNIST

<a name="models"></a>
# Models

<a name="network"></a>
# Network

<a name="network-labels"></a>
# Network Labels

<a name="neural-network"></a>
# Neural Network

<a name="nonlinearities"></a>
# Nonlinearities

<a name="normalization"></a>
# Normalization

<a name="objective"></a>
# Objective

<a name="one-hot-encoding"></a>
# One-Hot Encoding

<a name="operations"></a>
# Operations

<a name="optimizers"></a>
# Optimizers

<a name="over-vs-underfitting"></a>
# Over vs. Underfitting

<a name="preprocess"></a>
# Preprocess

<a name="preprocessing"></a>
# Preprocessing

<a name="pretrained-networks"></a>
# Pretrained Networks

<a name="priming"></a>
# Priming

<a name="probabilistic-sampling"></a>
# Probabilistic Sampling

<a name="protobuf"></a>
# Protobuf

<a name="recurrent-neural-networks"></a>
# Recurrent Neural Networks

<a name="regression"></a>
# Regression

<a name="reinforcement-learning"></a>
# Reinforcement Learning

<a name="relu"></a>
# ReLu

<a name="rnn"></a>
# RNN

<a name="sessions"></a>
# Sessions

<a name="sigmoid"></a>
# Sigmoid

<a name="softmax-layer"></a>
# Softmax Layer

<a name="standard-deviation"></a>
# Standard Deviation

<a name="stochastic"></a>
# Stochastic

<a name="style-features"></a>
# Style Features

<a name="style-loss"></a>
# Style Loss

<a name="style-net"></a>
# Style Net

<a name="supervised-learning"></a>
# Supervised Learning

<a name="tanh"></a>
# TanH

<a name="temperature"></a>
# Temperature

<a name="tensor"></a>
# Tensor

<a name="tensor-shapes"></a>
# Tensor Shapes

<a name="tensorboard"></a>
# Tensorboard

<a name="tensorflow-basics"></a>
# Tensorflow Basics

<a name="tensors"></a>
# Tensors

<a name="testing"></a>
# Testing

<a name="total-variation-loss"></a>
# Total Variation Loss

<a name="training"></a>
# Training

<a name="training-parameters"></a>
# Training Parameters

<a name="training-vs-testing"></a>
# Training vs. Testing

<a name="transpose"></a>
# Transpose

<a name="unsupervised-learning"></a>
# Unsupervised Learning

<a name="unsupervised-vs-supervised-learning"></a>
# Unsupervised vs. Supervised Learning

<a name="vaegan"></a>
# VAEGAN

<a name="variable"></a>
# Variable

<a name="variance"></a>
# Variance

<a name="variational-auto-encoding-generative-adversarial-network"></a>
# Variational Auto-Encoding Generative Adversarial Network

<a name="variational-autoencoders"></a>
# Variational Autoencoders

<a name="variational-layer"></a>
# Variational Layer

<a name="vector"></a>
# Vector

<a name="vgg-network"></a>
# VGG Network

---

Thanks to Golan Levin for [suggesting the idea](https://twitter.com/golan/status/798619471199883264).
