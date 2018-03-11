import numpy as np
import tensorflow as tf
import matplotlib
import matplotlib.pyplot as plt
from MyCapsNetwork import *
from MyCapsNetwork.CapsNetwork import *

from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("/tmp/data/")

X = tf.placeholder(shape=[None, 28, 28, 1], dtype=tf.float32, name="X")

caps1_n_maps = 32
caps1_n_caps = caps1_n_maps * 6 * 6  # 1152 primary capsules
caps1_n_dims = 8

conv1_params = {
    "filters": 256,
    "kernel_size": 9,
    "strides": 1,
    "padding": "valid",
    "activation": tf.nn.relu,
}

conv2_params = {
    "filters": caps1_n_maps * caps1_n_dims, # 256 convolutional filters
    "kernel_size": 9,
    "strides": 2,
    "padding": "valid",
    "activation": tf.nn.relu
}

conv1 = tf.layers.conv2d(X, name="conv1", **conv1_params)
conv2 = tf.layers.conv2d(conv1, name="conv2", **conv2_params)

caps1_raw = tf.reshape(conv2, [-1, caps1_n_caps, caps1_n_dims], name="caps1_raw")

caps_network = CapsNetwork(caps1_raw, X, "./my_capsule_network")
caps_network.train(mnist, epochs=10)
caps_network.eval(mnist)

