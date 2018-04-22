import numpy as np
import MyCapsNetwork
import itertools
from MyCapsNetwork.DataSet import *
import keras
import os.path
from Ticketing.ticketing_data import *
import tensorflow as tf
from MyCapsNetwork.ImageCapsNetwork import *


def pre_processing()->"DataSet":
    labels, class_names = get_merged_labels_three()

    x = get_fast_text_tickets_message()
    y = labels

    max = 0
    l = []

    for words in x:
        n = words.shape[0]
        if n > max:
            max = n

    new_x = np.zeros((x.shape[0], max, x[0].shape[1]))

    i = 0
    for words in x:
        new_x[i,:words.shape[0],:] = words
        i += 1

    x = new_x

    y = keras.utils.to_categorical(y)

    data_set = DataSet.from_np_array(x, y, class_names=class_names)

    return data_set;

def ticketing(train=True, eval=True, restore_checkpoint=True, epochs=2, batch_size = 100):
    ds = pre_processing()

    # shape (batch_size, time_step, vec_length)
    x = ds.x_train

    X = tf.placeholder(shape=[None, x.shape[1], x.shape[2]], dtype=tf.float32, name="X")
    
    
    #caps1_n_caps = 1
    #caps1_n_dims = x.shape[1]
    caps_net = CapsNetwork(X, X, "./ticketing", caps1_vec_norm=x.shape[2], caps2_caps=len(ds.class_names), caps1_caps=x.shape[1], decoder_output=x.shape[1]*x.shape[2])

    if train:
        caps_net.train(ds.x_train, ds.y_train, ds.x_val, ds.y_val, epochs=epochs, batch_size=batch_size, restore_checkpoint=restore_checkpoint)
    if eval:
        caps_net.eval(ds.x_test, ds.y_test)

