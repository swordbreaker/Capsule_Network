import numpy as np
import tensorflow as tf
import os
import matplotlib
import matplotlib.pyplot as plt
from MyCapsNetwork.ImageCapsNetwork import *
from MyCapsNetwork.DataSet import *

from tensorflow.examples.tutorials.mnist import input_data

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

data = input_data.read_data_sets('data/fashion')

labels = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot"]

#mnist = input_data.read_data_sets("/tmp/data/")
data_set = DataSet.fromtf(data)
img_caps_net = ImageCapsNetwork(data_set, "./mnist_fashion")
img_caps_net.train(epochs=2, batch_size=100, restore_checkpoint=True)
img_caps_net.eval()
img_caps_net.plot_solution(labels, n_samples=10)


#X = tf.placeholder(shape=[None, 28, 28, 1], dtype=tf.float32, name="X")

#caps1_n_maps = 32
#caps1_n_caps = caps1_n_maps * 6 * 6  # 1152 primary capsules
#caps1_n_dims = 8

#conv1_params = {
#    "filters": 256,
#    "kernel_size": 9,
#    "strides": 1,
#    "padding": "valid",
#    "activation": tf.nn.relu,
#}

#conv2_params = {
#    "filters": caps1_n_maps * caps1_n_dims, # 256 convolutional filters
#    "kernel_size": 9,
#    "strides": 2,
#    "padding": "valid",
#    "activation": tf.nn.relu
#}

#conv1 = tf.layers.conv2d(X, name="conv1", **conv1_params)
#conv2 = tf.layers.conv2d(conv1, name="conv2", **conv2_params)

#caps1_raw = tf.reshape(conv2, [-1, caps1_n_caps, caps1_n_dims], name="caps1_raw")

#caps_network = CapsNetwork(caps1_raw, X, "./my_capsule_network")

#x_train = mnist.train.images
#y_train = mnist.train.labels

#x_val = mnist.validation.images
#y_val = mnist.validation.labels

#x_test = mnist.test.images
#y_test = mnist.test.labels

#caps_network.train(x_train, y_train, x_val, y_val, epochs=1, batch_size=300)
#caps_network.eval(x_test, y_test)

#n_samples = 5
#sample_images = mnist.test.images[:n_samples].reshape([-1, 28, 28, 1])

#caps2_output_value, decoder_output_value, y_pred_value = caps_network.predict_and_reconstruct(sample_images)

#sample_images = sample_images.reshape(-1, 28, 28)
#reconstructions = decoder_output_value.reshape([-1, 28, 28])

#plt.figure(figsize=(n_samples * 2, 3))
#for index in range(n_samples):
#    plt.subplot(1, n_samples, index + 1)
#    plt.imshow(sample_images[index], cmap="binary")
#    plt.title("Label:" + str(mnist.test.labels[index]))
#    plt.axis("off")

#plt.show()

#plt.figure(figsize=(n_samples * 2, 3))
#for index in range(n_samples):
#    plt.subplot(1, n_samples, index + 1)
#    plt.title("Predicted:" + str(y_pred_value[index]))
#    plt.imshow(reconstructions[index], cmap="binary")
#    plt.axis("off")

#plt.show()
