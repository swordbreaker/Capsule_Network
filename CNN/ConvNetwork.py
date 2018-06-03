import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, Activation
from keras.optimizers import RMSprop, SGD
import matplotlib.pyplot as plt


class ConvNetwork(object):
    """description of class"""


    def __init__(self, input_shape=(28,28,1), num_classes=10, saved_model_path=None):
        if(saved_model_path is None):
            self.input_shape = input_shape
            self.num_classes = num_classes
            self.model = self.compile_model()
        else:
            self.model = keras.models.load_model(saved_model_path)
        print(self.model.summary())

    def compile_model(self):
        model = Sequential()
        model.add(Conv2D(filters=32, kernel_size=(5,5), activation='relu', input_shape=self.input_shape))
        model.add(MaxPooling2D(pool_size=(2,2), strides=2))
        model.add(Conv2D(filters=64, kernel_size=(5,5), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2,2), strides=2))
        model.add(Flatten())
        model.add(Dense(1024, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(self.num_classes, activation='softmax'))
        model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(), metrics=['accuracy'])

        return model

    def fit(self, x_train, y_train, x_val, y_val, batch_size=200, epochs=5, verbose=1):
        self.model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=verbose, validation_data=(x_val, y_val))

    def eval(self, x_val, y_val):
        score = self.model.evaluate(x_val, y_val, verbose=0)
        print('Test loss:', score[0])
        print('Test accuracy:', score[1])

    def predict(self, x):
        return self.model.predict(x)

    def save(self, path):
        self.model.save(path)