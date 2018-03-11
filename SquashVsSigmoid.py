import numpy as np
import numpy.linalg
import matplotlib
import matplotlib.pyplot as plt
from math import *


def sigmoid(x: float) -> float:
    return 1/(1 + exp(-x))


def squash(s: np.ndarray) -> np.ndarray:
    magnitude2 = (s ** 2).sum()
    print(magnitude2)
    if magnitude2 == 0:
        return np.array([0, 0])
    return (magnitude2 / (1 + magnitude2)) * (s/sqrt(magnitude2))


start = -10
end = 50
sigs = np.zeros(shape=end - start)
squashes = np.zeros(shape=end - start)
idx = np.zeros(shape=end - start)

for i in range(start, end):
    sigs[i + -start] = sigmoid(i)
    squashes[i + -start] = np.linalg.norm(squash(np.array([i, 0])))
    idx[i + -start] = i


figure = plt.figure()
plt.axvline(0, color='grey')
sigplot = plt.plot(idx, sigs, label="sigma")
sqplot = plt.plot(idx, squashes, label="squash")
plt.legend(["sigma", "squash"])

plt.show()
