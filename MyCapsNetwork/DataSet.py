class DataSet(object):
    """description of class"""

    def __init__(self, x_train, y_train, x_val, y_val, x_test, y_test):
        self.x_train = x_train
        self.y_train = y_train
        self.x_val = x_val
        self.y_val = y_val
        self.x_test = x_test
        self.y_test = y_test

    @staticmethod
    def fromtf(dataset):
        x_train = dataset.train.images
        y_train = dataset.train.labels
        x_val = dataset.validation.images
        y_val = dataset.validation.labels
        x_test = dataset.test.images
        y_test = dataset.test.labels
        return DataSet(x_train, y_train, x_val, y_val, x_test, y_test)