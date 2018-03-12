import numpy as np
from Other.Softmax import softmax
from math import *

m_plus = 0.9
m_minus = 0.1
lambda_ = 0.5


def margin_loss(vk: np.ndarray, tk: float):
    v_norm = np.linalg.norm(vk)
    l = tk * max(0., m_plus - v_norm)**2 + lambda_ * (1 - tk) * max(0., v_norm - m_minus)**2
    return l


# false positive werden weniger bestraft als false negative

v = np.array([0, 0, 0, 1])
# v = softmax(v)
t = 0

print(f"v: {v}")
print(f"T: {t}")

l = margin_loss(v, t)

print(f"L: {l}")
