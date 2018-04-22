import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate

class DataSet(object):
    """description of class"""

    def __init__(self, x_train, y_train, x_val, y_val, x_test, y_test, class_names: [str] = None):
        self.x_train = x_train
        self.y_train = y_train
        self.x_val = x_val
        self.y_val = y_val
        self.x_test = x_test
        self.y_test = y_test
        self.class_names = class_names

    @staticmethod
    def fromtf(dataset):
        x_train = dataset.train.images
        y_train = dataset.train.labels
        x_val = dataset.validation.images
        y_val = dataset.validation.labels
        x_test = dataset.test.images
        y_test = dataset.test.labels
        return DataSet(x_train, y_train, x_val, y_val, x_test, y_test)

    @staticmethod
    def from_np_array(x: np.ndarray, y: np.ndarray, class_names: [str] = None, p_train=0.6, p_val=0.2):
        n = x.shape[0]
        n_train = int(n * p_train)
        n_val = int(n * p_val)

        x_train = x[:n_train]
        y_train = y[:n_train]
        x_val = x[n_train:n_train + n_val]
        y_val = y[n_train:n_train + n_val]
        x_test = x[n_train + n_val:]
        y_test = y[n_train + n_val:]

        return DataSet(x_train, y_train, x_val, y_val, x_test, y_test, class_names)

    def __str__(self):
        return f"Train set: {self.x_train.shape[0]} samples \nValidation set: {self.x_val.shape[0]} samples \nTest set: {self.x_test.shape[0]} samples"