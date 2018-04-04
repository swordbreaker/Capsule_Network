from MyCapsNetwork.CapsNetwork import *
import tensorflow as tf


class CapsDecoder(object):
    n_hidden1 = 512
    n_hidden2 = 1024
    #n_output = 28 * 28

    def __init__(self, caps_network: "CapsNetwork", n_output : int):
        self.n_output = n_output
        self.caps_network = caps_network
        self.mask_with_labels = tf.placeholder_with_default(False, shape=(), name="mask_with_labels")
        self.decoder_input = self.build_decoder_input()
        self.decoder_output = self.build_decoder_output()

    def build_decoder_input(self):
        with tf.name_scope("build_decoder_input"):
            reconstruction_targets = tf.cond(self.mask_with_labels,  # condition
                                             lambda: self.caps_network.y,  # if True use y
                                             lambda: self.caps_network.y_pred,  # if False use y_pred
                                             name="reconstruction_targets")
            reconstruction_mask = tf.one_hot(reconstruction_targets, depth=self.caps_network.caps2_caps, name="reconstruction_mask")
            reconstruction_mask_reshaped = tf.reshape(reconstruction_mask, [-1, 1, self.caps_network.caps2_caps, 1, 1], name="reconstruction_mask_reshaped")
            caps2_output_masked = tf.multiply(self.caps_network.caps2_output, reconstruction_mask_reshaped, name="caps2_output_masked")
            return tf.reshape(caps2_output_masked, [-1, self.caps_network.caps2_caps * self.caps_network.caps2_vec_norm], name="decoder_input")

    def build_decoder_output(self):
        with tf.name_scope("decoder"):
            hidden1 = tf.layers.dense(self.decoder_input, self.n_hidden1, activation=tf.nn.relu, name="hidden1")
            hidden2 = tf.layers.dense(hidden1, self.n_hidden2, activation=tf.nn.relu, name="hidden2")
            return tf.layers.dense(hidden2, self.n_output, activation=tf.nn.sigmoid, name="decoder_output")

    def calc_reconstruction_loss(self, x):
        with tf.name_scope("calc_reconstruction"):
            X_flat = tf.reshape(x, [-1, self.n_output], name="X_flat")
            squared_difference = tf.square(X_flat - self.decoder_output, name="squared_difference")
            return tf.reduce_mean(squared_difference, name="reconstruction_loss")
