import numpy as np
from MyCapsNetwork.CapsNetwork import *
from MyCapsNetwork.DataSet import *
import tensorflow as tf
import matplotlib
import matplotlib.pyplot as plt

class ImageCapsNetwork(object):
    """description of class"""

    def __init__(self, ds: DataSet, checkpoint_path, *args, **kwargs):
        self.caps_network = self.__creatCapsNetwork(checkpoint_path)
        self.ds = ds

        return super().__init__(*args, **kwargs)

    def __creatCapsNetwork(self, checkpoint_path):
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
        return CapsNetwork(caps1_raw, X, checkpoint_path)

    def train(self, epochs = 10, batch_size = 100, restore_checkpoint = True):
        self.caps_network.train(self.ds.x_train, self.ds.y_train, self.ds.x_val, self.ds.y_val, epochs=epochs, batch_size=batch_size, restore_checkpoint=restore_checkpoint)

    def eval(self):
        self.caps_network.eval(self.ds.x_test, self.ds.y_test)

    def plot_solution(self, labels: [str], n_samples = 5): 
        sample_images = self.ds.x_test[:n_samples].reshape([-1, 28, 28, 1])

        caps2_output_value, decoder_output_value, y_pred_value = self.caps_network.predict_and_reconstruct(sample_images)

        sample_images = sample_images.reshape(-1, 28, 28)
        reconstructions = decoder_output_value.reshape([-1, 28, 28])

        plt.ion()

        plt.figure(figsize=(n_samples * 2, 3))
        for index in range(n_samples):
            plt.subplot(1, n_samples, index + 1)
            plt.imshow(sample_images[index], cmap="binary")
            plt.title("Label:" + labels[self.ds.y_test[index]])
            plt.axis("off")

        plt.show()

        plt.figure(figsize=(n_samples * 2, 3))
        for index in range(n_samples):
            plt.subplot(1, n_samples, index + 1)
            plt.title("Predicted:" + labels[y_pred_value[index]])
            plt.imshow(reconstructions[index], cmap="binary")
            plt.axis("off")

        plt.show()