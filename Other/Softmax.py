import numpy as np

def softmax(z: np.ndarray):
    z_exp = np.exp(z)
    sum_z_exp = z_exp.sum()
    return np.array([round(i / sum_z_exp, 3) for i in z_exp])


z = np.array([0., 1., 0.])
print(softmax(z))