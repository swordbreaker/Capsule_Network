import numpy as np
import tensorflow as tf
import os
import matplotlib
import matplotlib.pyplot as plt
from MyCapsNetwork.ImageCapsNetwork import *
from MyCapsNetwork.DataSet import *
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

from tensorflow.examples.tutorials.mnist import input_data

def one_hot_encode(data) -> np.ndarray:
    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(data)
    onehot_encoder = OneHotEncoder(sparse=False)
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    return onehot_encoder.fit_transform(integer_encoded)

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
    img_caps_net = ImageCapsNetwork(data_set, "./mnist_fashion2")
    if train:
        img_caps_net.train(epochs=epochs, batch_size=100, restore_checkpoint=True)
    if eval:
        img_caps_net.eval()
    #img_caps_net.plot_solution(labels, n_samples=10)
    #for i in range(10):
    #    img_caps_net.plot_from_category(labels, i, n_samples=10)

    #img_caps_net.transform_images_and_plot(labels, 2)
    #img_caps_net.manipulated_and_reconstruct(labels, 5)


def mushroom_example(train=True, eval=True, restore_checkpoint=True, epochs=2, batch_size = 100):
    mushrooms = pd.read_csv('data/mushrooms/mushrooms.csv')
        
    #x = np.zeros(shape=(mushrooms.shape[0],mushrooms.shape[1] - 1))
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
    # integer encode
    cap_shapes = one_hot_encode(mushrooms['cap-shape'])

    # cap-surface:
    #   fibrous=f,
    #   grooves=g,
    #   scaly=y,
    #   smooth=s
    cap_surface = one_hot_encode(mushrooms['cap-surface'])

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
    cap_color = one_hot_encode(mushrooms['cap-color'])

    # bruises:
    #   bruises=t,
    #   no=f
    bruises_keys = {'t': 0, 'f': 1}
    bruises = [bruises_keys[m] for m in mushrooms['bruises']]

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
    odor = one_hot_encode(mushrooms['odor'])

    # gill-attachment:
    #   attached=a,
    #   descending=d,
    #   free=f,
    #   notched=n
    gill_attachment = one_hot_encode(mushrooms['gill-attachment'])

    # gill-spacing:
    #   close=c,
    #   crowded=w,
    #   distant=d
    gill_spacing = one_hot_encode(mushrooms['gill-spacing'])

    # gill-size:
    #   broad=b,
    #   narrow=n
    grill_size_keys = {'b':0,'n':1}
    grill_size = [grill_size_keys[m] for m in mushrooms['gill-size']]

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
    gill_color = one_hot_encode(mushrooms['gill-color'])

    # stalk-shape:
    #   enlarging=e,
    #   tapering=t
    stalk_shape_key = {'e':0,'t':1}
    stalk_shape = [stalk_shape_key[m] for m in mushrooms['stalk-shape']]

    # stalk-root:
    #   bulbous=b,
    #   club=c,
    #   cup=u,
    #   equal=e,
    #   rhizomorphs=z,
    #   rooted=r,
    #   missing=?
    stalk_root = one_hot_encode(mushrooms['stalk-root'])

    # stalk-surface-above-ring:
    #   fibrous=f,
    #   scaly=y,
    #   silky=k,
    #   smooth=s
    stalk_surface_above_ring = one_hot_encode(mushrooms['stalk-surface-above-ring'])

    # stalk-surface-below-ring:
    #   fibrous=f,
    #   scaly=y,
    #   silky=k,
    #   smooth=s
    stalk_surface_below_ring = one_hot_encode(mushrooms['stalk-surface-below-ring'])

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
    stalk_color_above_ring = one_hot_encode(mushrooms['stalk-color-above-ring'])

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
    stalk_color_below_ring = one_hot_encode(mushrooms['stalk-color-below-ring'])

    # veil-type:
    #   partial=p,
    #   universal=u
    veil_type_key = {'p':0,'u':1}
    veil_type = [veil_type_key[m] for m in mushrooms['veil-type']]

    # veil-color:
    #   brown=n,
    #   orange=o,
    #   white=w,
    #   yellow=y
    veil_color = one_hot_encode(mushrooms['veil-color'])

    # ring-number:
    #   none=n,
    #   one=o,
    #   two=t
    ring_number_key = {'n':0,'o':1,'t':2}
    ring_number = [ring_number_key[m] for m in mushrooms['ring-number']]

    # ring-type:
    #   cobwebby=c,
    #   evanescent=e,
    #   flaring=f,
    #   large=l,
    #   none=n,
    #   pendant=p,
    #   sheathing=s,
    #   zone=z
    ring_type = one_hot_encode(mushrooms['ring-type'])

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
    spore_print_color = one_hot_encode(mushrooms['spore-print-color'])

    # population:
    #   abundant=a,
    #   clustered=c,
    #   numerous=n,
    #   scattered=s,
    #   several=v,
    #   solitary=y
    population_keys = {'a':5,'c':4,'n':3,'s':2,'v':1,'y':0}
    population = [population_keys[m] for m in mushrooms['population']]

    # habitat:
    #   grasses=g,
    #   leaves=l,
    #   meadows=m,
    #   paths=p,
    #   urban=u,
    #   waste=w,
    #   woods=d
    habitat = one_hot_encode(mushrooms['habitat'])

    #8124x108
    x = np.column_stack((cap_shapes, cap_surface, cap_color, bruises, odor, gill_attachment, gill_spacing, grill_size, gill_color, stalk_shape, stalk_root, stalk_surface_above_ring, stalk_surface_below_ring, stalk_color_above_ring, stalk_color_below_ring, veil_type, veil_color, ring_number, ring_type, spore_print_color, population, habitat, np.zeros(shape=(8124,1))))

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

    caps1_n_caps = x.shape[1] // 9  # 12 primary capsules
    caps1_n_dims = 9

    caps1_raw = tf.reshape(X, [-1, caps1_n_caps, caps1_n_dims], name="caps1_raw")
    caps_net = CapsNetwork(caps1_raw, X, "./mushrooms", caps1_vec_norm=caps1_n_dims, caps2_caps=2, caps1_caps=caps1_n_caps, decoder_output=x.shape[1])

    if train:
        caps_net.train(ds.x_train.reshape(-1, x.shape[1], 1), ds.y_train, ds.x_val.reshape(-1, x.shape[1], 1), ds.y_val, epochs=epochs, batch_size=batch_size, restore_checkpoint=restore_checkpoint)
    if eval:
        caps_net.eval(ds.x_test.reshape(-1, x.shape[1], 1), ds.y_test)

def mushroom_example2(train=True, eval=True, restore_checkpoint=True, epochs=2, batch_size = 100):
    mushrooms = pd.read_csv('data/mushrooms/mushrooms.csv')
        
    #x = np.zeros(shape=(mushrooms.shape[0],mushrooms.shape[1] - 1))
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
    # integer encode
    cap_shapes = one_hot_encode(mushrooms['cap-shape'])

    # cap-surface:
    #   fibrous=f,
    #   grooves=g,
    #   scaly=y,
    #   smooth=s
    cap_surface = one_hot_encode(mushrooms['cap-surface'])

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
    cap_color = one_hot_encode(mushrooms['cap-color'])

    # bruises:
    #   bruises=t,
    #   no=f
    bruises_keys = {'t': 0, 'f': 1}
    bruises = [bruises_keys[m] for m in mushrooms['bruises']]

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
    odor = one_hot_encode(mushrooms['odor'])

    # gill-attachment:
    #   attached=a,
    #   descending=d,
    #   free=f,
    #   notched=n
    gill_attachment = one_hot_encode(mushrooms['gill-attachment'])

    # gill-spacing:
    #   close=c,
    #   crowded=w,
    #   distant=d
    gill_spacing = one_hot_encode(mushrooms['gill-spacing'])

    # gill-size:
    #   broad=b,
    #   narrow=n
    grill_size_keys = {'b':0,'n':1}
    grill_size = [grill_size_keys[m] for m in mushrooms['gill-size']]

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
    gill_color = one_hot_encode(mushrooms['gill-color'])

    # stalk-shape:
    #   enlarging=e,
    #   tapering=t
    stalk_shape_key = {'e':0,'t':1}
    stalk_shape = [stalk_shape_key[m] for m in mushrooms['stalk-shape']]

    # stalk-root:
    #   bulbous=b,
    #   club=c,
    #   cup=u,
    #   equal=e,
    #   rhizomorphs=z,
    #   rooted=r,
    #   missing=?
    stalk_root = one_hot_encode(mushrooms['stalk-root'])

    # stalk-surface-above-ring:
    #   fibrous=f,
    #   scaly=y,
    #   silky=k,
    #   smooth=s
    stalk_surface_above_ring = one_hot_encode(mushrooms['stalk-surface-above-ring'])

    # stalk-surface-below-ring:
    #   fibrous=f,
    #   scaly=y,
    #   silky=k,
    #   smooth=s
    stalk_surface_below_ring = one_hot_encode(mushrooms['stalk-surface-below-ring'])

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
    stalk_color_above_ring = one_hot_encode(mushrooms['stalk-color-above-ring'])

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
    stalk_color_below_ring = one_hot_encode(mushrooms['stalk-color-below-ring'])

    # veil-type:
    #   partial=p,
    #   universal=u
    veil_type_key = {'p':0,'u':1}
    veil_type = [veil_type_key[m] for m in mushrooms['veil-type']]

    # veil-color:
    #   brown=n,
    #   orange=o,
    #   white=w,
    #   yellow=y
    veil_color = one_hot_encode(mushrooms['veil-color'])

    # ring-number:
    #   none=n,
    #   one=o,
    #   two=t
    ring_number_key = {'n':0,'o':1,'t':2}
    ring_number = [ring_number_key[m] for m in mushrooms['ring-number']]

    # ring-type:
    #   cobwebby=c,
    #   evanescent=e,
    #   flaring=f,
    #   large=l,
    #   none=n,
    #   pendant=p,
    #   sheathing=s,
    #   zone=z
    ring_type = one_hot_encode(mushrooms['ring-type'])

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
    spore_print_color = one_hot_encode(mushrooms['spore-print-color'])

    # population:
    #   abundant=a,
    #   clustered=c,
    #   numerous=n,
    #   scattered=s,
    #   several=v,
    #   solitary=y
    population_keys = {'a':5,'c':4,'n':3,'s':2,'v':1,'y':0}
    population = [population_keys[m] for m in mushrooms['population']]

    # habitat:
    #   grasses=g,
    #   leaves=l,
    #   meadows=m,
    #   paths=p,
    #   urban=u,
    #   waste=w,
    #   woods=d
    habitat = one_hot_encode(mushrooms['habitat'])

    #8124x107
    x = np.column_stack((cap_shapes, cap_surface, cap_color, bruises, odor, gill_attachment, gill_spacing, grill_size, gill_color, stalk_shape, stalk_root, stalk_surface_above_ring, stalk_surface_below_ring, stalk_color_above_ring, stalk_color_below_ring, veil_type, veil_color, ring_number, ring_type, spore_print_color, population, habitat))

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

    caps1_n_caps = 1
    caps1_n_dims = 107

    caps1_raw = tf.reshape(X, [-1, caps1_n_caps, caps1_n_dims], name="caps1_raw")
    caps_net = CapsNetwork(caps1_raw, X, "./mushrooms2", caps1_vec_norm=caps1_n_dims, caps2_caps=2, caps1_caps=caps1_n_caps, decoder_output=x.shape[1])

    if train:
        caps_net.train(ds.x_train.reshape(-1, x.shape[1], 1), ds.y_train, ds.x_val.reshape(-1, x.shape[1], 1), ds.y_val, epochs=epochs, batch_size=batch_size, restore_checkpoint=restore_checkpoint)
    if eval:
        caps_net.eval(ds.x_test.reshape(-1, x.shape[1], 1), ds.y_test)
