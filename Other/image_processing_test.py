import cv2
import os
import matplotlib
import matplotlib.pyplot as plt
from MyCapsNetwork.DataSet import *
import numpy as np

from tensorflow.examples.tutorials.mnist import input_data


labels = [str(i) for i in range(10)]
data = input_data.read_data_sets("/tmp/data/")
data_set = DataSet.fromtf(data)

img = data_set.x_test[0]
img = img.reshape(28, 28)

print(img)

plt.imshow(img)
plt.show()

M = cv2.getRotationMatrix2D((28/2,28/2),90,1)
dst = cv2.warpAffine(img,M,(28,28))

plt.imshow(dst)
plt.show()