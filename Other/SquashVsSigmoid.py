import numpy as np
import numpy.linalg
import matplotlib
import matplotlib.pyplot as plt
from math import *
#%matplotlib inline

def sigmoid(x: float) -> float:
    return 1/(1 + exp(-x))


def squash(s: np.ndarray) -> np.ndarray:
    s = np.array(s)
    magnitude2 = (s ** 2).sum()
    if magnitude2 == 0:
        return np.zeros(s.shape)
    return (magnitude2 * s)/(1 + magnitude2*sqrt(magnitude2))
    #return (magnitude2 / (1 + magnitude2)) * (s/sqrt(magnitude2))


start = -20
end = 20
sigs = np.zeros(shape=end - start)
squashes = np.zeros(shape=end - start)
idx = np.zeros(shape=end - start)

for i in range(start, end):
    sigs[i + -start] = sigmoid(i)
    squashes[i + -start] = np.linalg.norm(squash(np.array([i])))
    idx[i + -start] = i


figure = plt.figure()
sigplot = plt.plot(idx, sigs, label="sigmoid")
sqplot = plt.plot(idx, squashes, label="squash")
plt.axvline(0, color='grey')
plt.legend(["sigmoid", "squash"])
plt.xlim([start, end])

plt.show()