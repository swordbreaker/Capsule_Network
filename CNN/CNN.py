import numpy as np
import tensorflow as tf
import os
import matplotlib
import matplotlib.pyplot as plt
import keras
from tensorflow.examples.tutorials.mnist import input_data
from  ConvNetwork import ConvNetwork

import cv2

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
    cnn.fit(x_train, y_train, x_val, y_val, epochs=50)
    cnn.eval(x_val, y_val)
    cnn.save("checkpoints/cnn_mnist/model.hdf5")

def mnist_fashion():
    x_train, y_train, x_val, y_val, x_test, y_test = load_mnist_fashion()
    labels = [str(i) for i in range(10)]
    
    x_train = x_train[:5000]
    y_train = y_train[:5000]

    cnn = ConvNetwork(saved_model_path="checkpoints/cnn_small/epoch_22-val_los_0.43.hdf5")
    #cnn = ConvNetwork(input_shape=(28,28,1))
    cnn.fit(x_train, y_train, x_val, y_val, epochs=20)
    cnn.eval(x_val, y_val)
    #cnn.save("model_small.hdf5")

def transform(img, angle):
    n_samples = 4
    sample_images = np.zeros(shape=(n_samples, 28, 28, 1))

    angle_step = angle

    angle = -angle_step
    for i in range(4):
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

def mnist_transform(id:int = 0):
    x_train, y_train, x_val, y_val, x_test, y_test = load_minst()
    labels = [str(i) for i in range(10)]

    cnn = ConvNetwork(saved_model_path="checkpoints/cnn_mnist/model.hdf5")

    img = x_test[id]
    img = img.reshape(28, 28)

    sample_images = transform(img, 30)

    #n_samples = 8
    #sample_images = np.zeros(shape=(n_samples, 28, 28, 1))

    #angle = -30
    #for i in range(4):
    #    M = cv2.getRotationMatrix2D((28/2,28/2),angle,1)
    #    dst = cv2.warpAffine(img,M,(28,28))
    #    dst = dst.reshape([28, 28, 1])
    #    sample_images[i] = dst
    #    angle += 30

    #angle = -30
    #for i in range(4):
    #    M = cv2.getRotationMatrix2D((28/2,28/2),angle,0.5)
    #    dst = cv2.warpAffine(img,M,(28,28))
    #    dst = dst.reshape([28, 28, 1])
    #    sample_images[i+4] = dst
    #    angle += 30
    
    y_pred_value = cnn.predict(sample_images)
    y_pred_value = np.argmax(y_pred_value, axis=1)

    print(y_pred_value)
    sample_images = sample_images.reshape(-1, 28, 28)

    plt.figure(figsize=(n_samples, 6))
    for i in range(n_samples):
        plt.subplot(2, n_samples, i + 1)
        plt.imshow(sample_images[i], cmap="binary")
        plt.title("Label:" + labels[y_pred_value[i]])
        plt.axis("off")

    plt.show()


def mnist_transform_all(angle):
    x_train, y_train, x_val, y_val, x_test, y_test = load_minst()

    cnn = ConvNetwork(saved_model_path="checkpoints/cnn_mnist/model.hdf5")

    num_samples = 4
    sample_images = np.zeros(shape=(x_test.shape[0] * num_samples, 28, 28, 1))
    labels = np.zeros((x_test.shape[0] * num_samples, 10))

    k = 0
    for i in range(0, x_test.shape[0]):
        img = x_test[i]
        img = img.reshape(28, 28)

        sample_images[k:k+num_samples] = transform(img, angle)

        for j in range(num_samples):
            labels[k+j] = y_test[i]
        k += num_samples

    cnn.eval(sample_images, labels)

def noise_minst():
    x_train, y_train, x_val, y_val, x_test, y_test = load_minst()
    cnn = ConvNetwork(saved_model_path="checkpoints/cnn_mnist/model.hdf5")

    mean = 0
    var = 0.2
    sigma = var**0.5
    gauss = np.random.normal(mean,sigma,(28,28,1))
    
    sample_images = x_test + gauss

    sample_images = np.zeros(x_test.shape)
    for i in range(x_test.shape[0]):
        sample_images[i] = x_test[i] + gauss

    sample_images = sample_images.reshape(-1, 28,28,1)

    cnn.eval(sample_images, y_test)

x_train, y_train, x_val, y_val, x_test, y_test = load_minst()
cnn = ConvNetwork(saved_model_path="checkpoints/cnn_mnist/model.hdf5")
cnn.eval(x_test, y_test)

#mnist_fashion()
#mnist()
#mnist_transform()

#print("rotation: 30")
#mnist_transform_all(30)
#print("rotation: 40")
#mnist_transform_all(40)
#print("rotation: 50")
#mnist_transform_all(50)
#print("rotation: 90")
#mnist_transform_all(90)

#noise_minst()