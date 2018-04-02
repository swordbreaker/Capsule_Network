import numpy as np
import tensorflow as tf
import os
import matplotlib
import matplotlib.pyplot as plt
from MyCapsNetwork.ImageCapsNetwork import *
from MyCapsNetwork.DataSet import *
import pandas as pd

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
    #img_caps_net.plot_solution(labels, n_samples=10)
    img_caps_net.transform_images_and_plot(labels)

def mnist_fashion(train=True, eval=True, restore_checkpoint=True, epochs=2):
    data = input_data.read_data_sets('data/fashion')

    labels = ["T-shirt/top",
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

def mushroom_example(train=True, eval=True, restore_checkpoint=True, epochs=2, batch_size = 100):
    mushrooms = pd.read_csv('data/mushrooms/mushrooms.csv')
        
    x = np.zeros(shape=(mushrooms.shape[0],mushrooms.shape[1] - 1))
    y = np.zeros(shape=(mushrooms.shape[0]))
    # 1 + 21 collums
    # classes
    #   e = edible
    #   p = poisonous
    classes = {'e' : 0, 'p': 1}
    y[:] = [classes[m] for m in mushrooms['class']]

    # cap-shape:
    #   bell=b,
    #   conical=c,
    #   convex=x,
    #   flat=f,
    #   knobbed=k,
    #   sunken=s
    cap_shapes = {'b' : 0, 'c' : 1, 'x': 2, 'f' : 3, 'k' : 4, 's' : 5}
    x[:, 0] = [cap_shapes[m] for m in mushrooms['cap-shape']]

    # cap-surface:
    #   fibrous=f,
    #   grooves=g,
    #   scaly=y,
    #   smooth=s
    cap_surface = {'f': 0, 'g': 1, 'y': 2, 's': 3}
    x[:, 1] = [cap_surface[m] for m in mushrooms['cap-surface']]

    # cap-color:
    #   brown=n,
    #   buff=b,
    #   cinnamon=c,
    #   gray=g,
    #   green=r,
    #   pink=p,
    #   purple=u,
    #   red=e,
    #   white=w,
    #   yellow=y
    cap_color = {'n': 0, 'b': 1, 'c': 2, 'g': 3, 'r': 4, 'p': 5, 'u': 6, 'e': 7, 'w': 8, 'y': 9}
    x[:, 2] = [cap_color[m] for m in mushrooms['cap-color']]

    # bruises:
    #   bruises=t,
    #   no=f
    bruises = {'t': 0, 'f': 1}
    x[:, 3] = [bruises[m] for m in mushrooms['bruises']]

    # odor:
    #   almond=a,
    #   anise=l,
    #   creosote=c,
    #   fishy=y,
    #   foul=f,
    #   musty=m,
    #   none=n,
    #   pungent=p,
    #   spicy=s
    odor = {'a':0,'l':1,'c':2,'y':3,'f':4,'m':5,'n':6,'p':7,'s':8 }
    x[:, 4] = [odor[m] for m in mushrooms['odor']]

    # gill-attachment:
    #   attached=a,
    #   descending=d,
    #   free=f,
    #   notched=n
    gill_attachment = {'a':0,'d':1,'f':2,'n':3}
    x[:, 5] = [gill_attachment[m] for m in mushrooms['gill-attachment']]

    # gill-spacing:
    #   close=c,
    #   crowded=w,
    #   distant=d
    gill_spacing = {'c':0,'w':1,'d':2}
    x[:, 6] = [gill_spacing[m] for m in mushrooms['gill-spacing']]

    # gill-size:
    #   broad=b,
    #   narrow=n
    grill_size = {'b':0,'n':1}
    x[:, 7] = [grill_size[m] for m in mushrooms['gill-size']]

    # gill-color:
    #   black=k,
    #   brown=n,
    #   buff=b,
    #   chocolate=h,
    #   gray=g,
    #   green=r,
    #   orange=o,
    #   pink=p,
    #   purple=u,
    #   red=e,
    #   white=w,
    #   yellow=y
    gill_color = {'k':0,'n':1,'b':2,'h':3,'g':4,'r':5,'o':6,'p':7,'u':8,'e':9,'w':10,'y':11}
    x[:, 8] = [gill_color[m] for m in mushrooms['gill-color']]

    # stalk-shape:
    #   enlarging=e,
    #   tapering=t
    stalk_shape = {'e':0,'t':1}
    x[:, 9] = [stalk_shape[m] for m in mushrooms['stalk-shape']]

    # stalk-root:
    #   bulbous=b,
    #   club=c,
    #   cup=u,
    #   equal=e,
    #   rhizomorphs=z,
    #   rooted=r,
    #   missing=?
    stalk_root = {'b':0,'c':1,'u':2,'e':3,'z':4,'r':5,'?':6}
    x[:, 10] = [stalk_root[m] for m in mushrooms['stalk-root']]

    # stalk-surface-above-ring:
    #   fibrous=f,
    #   scaly=y,
    #   silky=k,
    #   smooth=s
    stalk_surface_above_ring = {'f':0,'y':1,'k':2,'s':3}
    x[:, 11] = [stalk_surface_above_ring[m] for m in mushrooms['stalk-surface-above-ring']]

    # stalk-surface-below-ring:
    #   fibrous=f,
    #   scaly=y,
    #   silky=k,
    #   smooth=s
    stalk_surface_below_ring = {'f':0,'y':1,'k':2,'s':3}
    x[:, 12] = [stalk_surface_below_ring[m] for m in mushrooms['stalk-surface-below-ring']]

    # stalk-color-above-ring:
    #   brown=n,
    #   buff=b,
    #   cinnamon=c,
    #   gray=g,
    #   orange=o,
    #   pink=p,
    #   red=e,
    #   white=w,
    #   yellow=y
    stalk_color_above_ring = {'n':0,'b':1,'c':2,'g':3,'o':4,'p':5,'e':6,'w':7,'y':8}
    x[:, 13] = [stalk_color_above_ring[m] for m in mushrooms['stalk-color-above-ring']]

    # stalk-color-below-ring:
    #   brown=n,
    #   buff=b,
    #   cinnamon=c,
    #   gray=g,
    #   orange=o,
    #   pink=p,
    #   red=e,
    #   white=w,
    #   yellow=y
    stalk_color_below_ring = {'n':0,'b':1,'c':2,'g':3,'o':4,'p':5,'e':6,'w':7,'y':8}
    x[:, 14] = [stalk_color_below_ring[m] for m in mushrooms['stalk-color-below-ring']]

    # veil-type:
    #   partial=p,
    #   universal=u
    veil_type = {'p':0,'u':1}
    x[:, 15] = [veil_type[m] for m in mushrooms['veil-type']]

    # veil-color:
    #   brown=n,
    #   orange=o,
    #   white=w,
    #   yellow=y
    veil_color = {'n':0,'o':1,'w':2,'y':3}
    x[:, 16] = [veil_color[m] for m in mushrooms['veil-color']]

    # ring-number:
    #   none=n,
    #   one=o,
    #   two=t
    ring_number = {'n':0,'o':1,'t':2}
    x[:, 17] = [ring_number[m] for m in mushrooms['ring-number']]

    # ring-type:
    #   cobwebby=c,
    #   evanescent=e,
    #   flaring=f,
    #   large=l,
    #   none=n,
    #   pendant=p,
    #   sheathing=s,
    #   zone=z
    ring_type = {'c':0,'e':1,'f':2,'l':3,'n':4,'p':5,'s':6,'z':7}
    x[:, 18] = [ring_type[m] for m in mushrooms['ring-type']]

    # spore-print-color:
    #   black=k,
    #   brown=n,
    #   buff=b,
    #   chocolate=h,
    #   green=r,
    #   orange=o,
    #   purple=u,
    #   white=w,
    #   yellow=y
    spore_print_color = {'k':0,'n':1,'b':2,'h':3,'r':4,'o':5,'u':6,'w':7,'y':8}
    x[:, 19] = [spore_print_color[m] for m in mushrooms['spore-print-color']]

    # population:
    #   abundant=a,
    #   clustered=c,
    #   numerous=n,
    #   scattered=s,
    #   several=v,
    #   solitary=y
    population = {'a':0,'c':1,'n':2,'s':3,'v':4,'y':5}
    x[:, 20] = [population[m] for m in mushrooms['population']]

    # habitat:
    #   grasses=g,
    #   leaves=l,
    #   meadows=m,
    #   paths=p,
    #   urban=u,
    #   waste=w,
    #   woods=d
    habitat = {'g':0,'l':1,'m':2,'p':3,'u':4,'w':5,'d':6}
    x[:, 21] = [habitat[m] for m in mushrooms['habitat']]

    #shuffle
    np.random.seed(42)
    idx = np.arange(x.shape[0])
    np.random.shuffle(idx)
    x = x[idx]
    y = y[idx]

    #splitt
    n = x.shape[0]
    n_train = int(n * 0.6)
    n_val = int(n * 0.2)

    x_train = x[:n_train]
    y_train = y[:n_train]
    x_val   = x[n_train:n_train+n_val]
    y_val   = y[n_train:n_train+n_val]
    x_test  = x[n_train+n_val:]
    y_test  = y[n_train+n_val:]

    ds = DataSet(x_train, y_train, x_val, y_val, x_test, y_test)

    X = tf.placeholder(shape=[None, x.shape[1], 1], dtype=tf.float32, name="X")

    caps1_n_maps = 32
    caps1_n_caps = caps1_n_maps * 6 * 6  # 1152 primary capsules
    caps1_n_dims = 7

    caps1_raw = tf.reshape(X, [-1, caps1_n_caps, caps1_n_dims], name="caps1_raw")
    caps_net.caps1_vec_norm = caps1_n_dims
    caps_net.caps2_output = 2
    caps_net = CapsNetwork(caps1_raw, X, "./mushrooms")

    if train:
        caps_net.train(ds.x_train.reshape(-1, x.shape[1], 1), ds.y_train, ds.x_val.reshape(-1, x.shape[1], 1), ds.y_val, epochs=epochs, batch_size=batch_size, restore_checkpoint=restore_checkpoint)
    if eval:
        caps_net.eval(ds.x_test.reshape(-1, x.shape[1], 1), ds.y_test)