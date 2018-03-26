import numpy as np
import tensorflow as tf
import os
import matplotlib
import matplotlib.pyplot as plt
from MyCapsNetwork.ImageCapsNetwork import *
from MyCapsNetwork.DataSet import *

from tensorflow.examples.tutorials.mnist import input_data


def mnist(train=True, eval=True, restore_checkpoint=True):
    labels = [str(i) for i in range(10)]
    data = input_data.read_data_sets("/tmp/data/")
    data_set = DataSet.fromtf(data)
    img_caps_net = ImageCapsNetwork(data_set, "./my_capsule_network")
    if train:
        img_caps_net.train(epochs=2, batch_size=100, restore_checkpoint=restore_checkpoint)
    if eval:
        img_caps_net.eval()
    img_caps_net.plot_solution(labels, n_samples=10)
    plt.show()


def mnist_fashion(train=True, eval=True, restore_checkpoint=True, epochs=2):
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

    data_set = DataSet.fromtf(data)
    img_caps_net = ImageCapsNetwork(data_set, "./mnist_fashion")
    if train:
        img_caps_net.train(epochs=epochs, batch_size=100, restore_checkpoint=True)
    if eval:
        img_caps_net.eval()
    img_caps_net.plot_solution(labels, n_samples=10)
    for i in range(10):
        img_caps_net.plot_from_category(labels, i, n_samples=10)
