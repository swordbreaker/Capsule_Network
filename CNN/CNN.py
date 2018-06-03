import numpy as np
import tensorflow as tf
import os
import matplotlib
import matplotlib.pyplot as plt
import keras
from tensorflow.examples.tutorials.mnist import input_data
from  ConvNetwork import ConvNetwork

def load_minst():
    dataset = input_data.read_data_sets("/tmp/data/")
    x_train = dataset.train.images
    y_train = dataset.train.labels
    x_val = dataset.validation.images
    y_val = dataset.validation.labels
    x_test = dataset.test.images
    y_test = dataset.test.labels

    x_train = x_train.reshape(-1, 28, 28, 1)
    x_val = x_val.reshape(-1, 28, 28, 1)
    x_test = x_test.reshape(-1, 28, 28, 1)
    y_train = keras.utils.to_categorical(y_train, 10)
    y_val = keras.utils.to_categorical(y_val, 10)
    y_test = keras.utils.to_categorical(y_test, 10)

    return (x_train, y_train, x_val, y_val, x_test, y_test)

def load_mnist_fashion():
    dataset = input_data.read_data_sets('data/fashion')
    x_train = dataset.train.images
    y_train = dataset.train.labels
    x_val = dataset.validation.images
    y_val = dataset.validation.labels
    x_test = dataset.test.images
    y_test = dataset.test.labels

    x_train = x_train.reshape(-1, 28, 28, 1)
    x_val = x_val.reshape(-1, 28, 28, 1)
    x_test = x_test.reshape(-1, 28, 28, 1)
    y_train = keras.utils.to_categorical(y_train, 10)
    y_val = keras.utils.to_categorical(y_val, 10)
    y_test = keras.utils.to_categorical(y_test, 10)

    return (x_train, y_train, x_val, y_val, x_test, y_test)

def mnist():
    x_train, y_train, x_val, y_val, x_test, y_test = load_minst()
    labels = [str(i) for i in range(10)]

    #cnn = ConvNetwork(saved_model_path="model.hdf5")
    cnn = ConvNetwork(input_shape=(28,28,1))
    cnn.fit(x_train, y_train, x_val, y_val, epochs=10)
    cnn.eval(x_val, y_val)
    #cnn.save("model.hdf5")

def mnist_fashion():
    x_train, y_train, x_val, y_val, x_test, y_test = load_mnist_fashion()
    labels = [str(i) for i in range(10)]

    #cnn = ConvNetwork(saved_model_path="model.hdf5")
    cnn = ConvNetwork(input_shape=(28,28,1))
    cnn.fit(x_train, y_train, x_val, y_val, epochs=16)
    cnn.eval(x_val, y_val)
    #cnn.save("model.hdf5")

mnist()