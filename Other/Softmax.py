import numpy as np

z = np.array([0., 0., 0., 0., 1., 0., 0.])
z_exp = np.exp(z)
print(f"z_exp: {z_exp.round(2)}")

sum_z_exp = z_exp.sum()
print(f"sum z_exp: {sum_z_exp}")

softmax = [round(i / sum_z_exp, 3) for i in z_exp]
print(f"softmax: {softmax}")
