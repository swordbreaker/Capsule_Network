import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, Activation
from keras.optimizers import RMSprop, SGD
import matplotlib.pyplot as plt
import time

class TimeLogger(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.start = time.time()

    def on_epoch_end(self, epoch, logs={}):
        diff = time.time() - self.start
        print(f"elapsed time: {diff}")

class MetricsLogger(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.i = 0
        self.x = []
        self.losses = []
        self.val_losses = []
        self.acc = []
        self.val_acc = []
        self.fig = plt.figure()
        self.start = time.time()
        self.time = []
        
        self.logs = []

    def on_epoch_end(self, epoch, logs={}):
        diff = time.time() - self.start
        self.time.append(diff)
        self.logs.append(logs)
        self.x.append(self.i)
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))
        self.acc.append(logs.get('acc'))
        self.val_acc.append(logs.get('val_acc'))
        self.i += 1

    def print(self):
        print("losses")
        print(self.losses)
        print("val_losses")
        print(self.val_losses)
        print("acc")
        print(self.acc)
        print("val_acc")
        print(self.val_acc)
        print("time")
        print(self.time)
        

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

    def fit(self, x_train, y_train, x_val, y_val, batch_size=100, epochs=5, verbose=2):
        #chekpoint_path = "./checkpoints/cnn/"
        #model_checkpoint_callback = keras.callbacks.ModelCheckpoint(chekpoint_path + "epoch_{epoch:02d}-val_los_{val_loss:.2f}.hdf5", monitor='val_loss', verbose=0, save_best_only=False, save_weights_only=False, mode='auto', period=1)

        chekpoint_path = "./checkpoints/cnn_small/"
        model_checkpoint_callback = keras.callbacks.ModelCheckpoint(chekpoint_path + "epoch_{epoch:02d}-val_los_{val_loss:.2f}.hdf5", monitor='val_loss', verbose=0, save_best_only=False, save_weights_only=False, mode='auto', period=1)
        time_logger = TimeLogger()
        metrics_logger = MetricsLogger()
        self.model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=verbose, validation_data=(x_val, y_val),  callbacks=[model_checkpoint_callback, time_logger, metrics_logger])
        metrics_logger.print()

    def eval(self, x_val, y_val):
        score = self.model.evaluate(x_val, y_val, verbose=0, batch_size=200)
        print('Test loss:', score[0])
        print('Test accuracy:', score[1])

    def predict(self, x):
        return self.model.predict(x)

    def save(self, path):
        self.model.save(path)