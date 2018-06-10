import numpy as np
from MyCapsNetwork.CapsNetwork import *
from MyCapsNetwork.DataSet import *
import tensorflow as tf
import matplotlib
import matplotlib.pyplot as plt
import cv2
import time
import os
import keras

def transform(img, angle=30):
    n_samples = 3
    sample_images = np.zeros(shape=(n_samples, 28, 28, 1))

    angle_step = angle

    angle = -angle_step
    for i in range(3):
        M = cv2.getRotationMatrix2D((28/2,28/2),angle,1)
        dst = cv2.warpAffine(img,M,(28,28))
        dst = dst.reshape([28, 28, 1])
        sample_images[i] = dst
        angle += angle_step

    #angle = -angle_step
    #for i in range(4):
    #    M = cv2.getRotationMatrix2D((28/2,28/2),angle,0.5)
    #    dst = cv2.warpAffine(img,M,(28,28))
    #    dst = dst.reshape([28, 28, 1])
    #    sample_images[i+4] = dst
    #    angle += angle_step

    return sample_images

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
        tf.summary.tensor_summary("conv1", conv1)
        conv2 = tf.layers.conv2d(conv1, name="conv2", **conv2_params)
        tf.summary.tensor_summary("conv2", conv2)

        caps1_raw = tf.reshape(conv2, [-1, caps1_n_caps, caps1_n_dims], name="caps1_raw")
        return CapsNetwork(caps1_raw, X, checkpoint_path)

    def train(self, epochs = 10, batch_size = 100, restore_checkpoint = True):
        x_train = self.ds.x_train.reshape([-1, 28, 28, 1])
        x_val = self.ds.x_val.reshape([-1, 28, 28, 1])
        self.caps_network.train(x_train, self.ds.y_train, x_val, self.ds.y_val, epochs=epochs, batch_size=batch_size, restore_checkpoint=restore_checkpoint)

    def eval(self):
        self.caps_network.eval(self.ds.x_test.reshape([-1, 28, 28, 1]), self.ds.y_test)

    def plot_from_category(self, labels: [str], category, n_samples = 5):
        idx = self.ds.y_test == category

        sample_images = self.ds.x_test[idx]
        sample_ys = (self.ds.y_test[idx])[:n_samples]
        sample_images = sample_images[:n_samples].reshape([-1, 28, 28, 1])

        caps2_output_value, decoder_output_value, y_pred_value = self.caps_network.predict_and_reconstruct(sample_images)

        sample_images = sample_images.reshape(-1, 28, 28)
        reconstructions = decoder_output_value.reshape([-1, 28, 28])

        plt.figure(figsize=(n_samples * 2, 6))
        for i in range(n_samples):
            plt.subplot(2, n_samples, i + 1)
            plt.imshow(sample_images[i], cmap="binary")
            plt.title("Label:" + labels[sample_ys[i]])
            plt.axis("off")

        for i in range(n_samples):
            plt.subplot(2, n_samples, n_samples + i + 1)
            plt.title(labels[y_pred_value[i]])
            plt.imshow(reconstructions[i], cmap="binary")
            plt.axis("off")

        plt.show(block=False)
        plt.show()

    def plot_solution(self, labels: [str], n_samples = 5): 
        sample_images = self.ds.x_test[:n_samples].reshape([-1, 28, 28, 1])

        caps2_output_value, decoder_output_value, y_pred_value = self.caps_network.predict_and_reconstruct(sample_images)

        sample_images = sample_images.reshape(-1, 28, 28)
        reconstructions = decoder_output_value.reshape([-1, 28, 28])

        plt.figure(figsize=(n_samples * 2, 6))
        for i in range(n_samples):
            plt.subplot(2, n_samples, i + 1)
            plt.imshow(sample_images[i], cmap="binary")
            plt.title("Label:" + labels[self.ds.y_test[i]])
            plt.axis("off")

        for i in range(n_samples):
            plt.subplot(2, n_samples, n_samples + i + 1)
            plt.title(labels[y_pred_value[i]])
            plt.imshow(reconstructions[i], cmap="binary")
            plt.axis("off")

        plt.show(block=False)
        plt.show()

    def manipulated_and_reconstruct(self, labels: [str], log_name, id = 0):
        x = self.ds.x_test[id].reshape([1, 28, 28, 1])
        y = self.ds.y_test[id].reshape([-1])

        str_time = f"{time.strftime('%Y_%m_%d_%H_%M_%S')}"
        path = f"Results/{log_name}_{labels[y[0]]}_{str_time}"
        os.makedirs(path)

        for k in range(16):
            shift = np.zeros((16))
            shift[k] = -0.20

            plt.figure()

            for i in range(5):
                caps2_output_value, decoder_output_value, y_pred_value = self.caps_network.recunstruct_shifted(x, y, shift)
            
                reconstructions = decoder_output_value.reshape([-1, 28, 28])
                plt.subplot(1, 5, i + 1)
                plt.title(f"{shift[k]:1.2f}")
                plt.imshow(reconstructions[0], cmap="binary")
                plt.axis("off")

                shift[k] += 0.10
            plt.savefig(f"{path}/{k}.png")

    def transform_images_and_plot(self, labels: [str], id = 0):
        img = self.ds.x_test[id]
        img = img.reshape(28, 28)

        n_samples = 6
        sample_images = np.zeros(shape=(n_samples, 28, 28, 1))

        angle = -30
        for i in range(3):
            M = cv2.getRotationMatrix2D((28/2,28/2),angle,1)
            dst = cv2.warpAffine(img,M,(28,28))
            dst = dst.reshape([28, 28, 1])
            sample_images[i] = dst
            angle += 30

        angle = -30
        for i in range(3):
            M = cv2.getRotationMatrix2D((28/2,28/2),angle,0.5)
            dst = cv2.warpAffine(img,M,(28,28))
            dst = dst.reshape([28, 28, 1])
            sample_images[i+3] = dst
            angle += 30

        caps2_output_value, decoder_output_value, y_pred_value = self.caps_network.predict_and_reconstruct(sample_images)

        sample_images = sample_images.reshape(-1, 28, 28)
        reconstructions = decoder_output_value.reshape([-1, 28, 28])

        plt.figure(figsize=(n_samples * 2, 6))
        for i in range(n_samples):
            plt.subplot(2, n_samples, i + 1)
            plt.imshow(sample_images[i], cmap="binary")
            plt.title("Label:" + labels[self.ds.y_test[0]])
            plt.axis("off")

        for i in range(n_samples):
            plt.subplot(2, n_samples, n_samples + i + 1)
            plt.title(labels[y_pred_value[i]])
            plt.imshow(reconstructions[i], cmap="binary")
            plt.axis("off")

        plt.show()

    def mnist_transform_all(self, angle):
        num_samples = 3
        sample_images = np.zeros(shape=(self.ds.x_test.shape[0] * num_samples, 28, 28, 1))
        labels = np.zeros((self.ds.x_test.shape[0] * num_samples))

        k = 0
        for i in range(0, self.ds.x_test.shape[0]):
            img = self.ds.x_test[i]
            img = img.reshape(28, 28)

            sample_images[k:k+num_samples] = transform(img, angle)
            labels[k:k+num_samples] = np.repeat(self.ds.y_test[i], num_samples)
            k += num_samples

        self.caps_network.eval(sample_images, labels, 500)


    def noise_all(self):
        mean = 0
        var = 0.2
        sigma = var**0.5
        gauss = np.random.normal(mean,sigma,(28,28,1))
        gauss = gauss.reshape(28*28)

        sample_images = self.ds.x_test + gauss
        sample_images = sample_images.reshape(-1, 28,28,1)

        self.caps_network.eval(sample_images, self.ds.y_test, 500)


    def noise_images_and_plot(self, labels: [str]):
        n_samples = 3
        img = self.ds.x_test[:n_samples]

        mean = 0
        var = 0.1
        sigma = var**0.5
        gauss = np.random.normal(mean,sigma,(28,28,1))
        gauss = gauss.reshape(28*28)

        for i in range(n_samples):
            img[i] = img[i] + gauss

        
        img = img.reshape(-1, 28, 28, 1)

        caps2_output_value, decoder_output_value, y_pred_value = self.caps_network.predict_and_reconstruct(img)

        sample_images = img.reshape(-1, 28, 28)
        reconstructions = decoder_output_value.reshape([-1, 28, 28])

        plt.figure(figsize=(n_samples * 2, 6))
        for i in range(n_samples):
            plt.subplot(2, n_samples, i + 1)
            plt.imshow(sample_images[i], cmap="binary")
            plt.title("Label:" + labels[self.ds.y_test[i]])
            plt.axis("off")

        for i in range(n_samples):
            plt.subplot(2, n_samples, n_samples + i + 1)
            plt.title(labels[y_pred_value[i]])
            plt.imshow(reconstructions[i], cmap="binary")
            plt.axis("off")

        plt.show()