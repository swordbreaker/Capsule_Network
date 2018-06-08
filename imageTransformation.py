import cv2
import numpy as np

def rotate(img):
    angle = -22.5
    for i in range(4):
        M = cv2.getRotationMatrix2D((28/2,28/2),angle,1)
        dst = cv2.warpAffine(img,M,(28,28))
        dst = dst.reshape([28, 28, 1])
        sample_images[i] = dst
        angle += 22.5


def rotated_and_scale(img):
    angle = -22.5
    for i in range(4):
        M = cv2.getRotationMatrix2D((28/2,28/2),angle,0.5)
        dst = cv2.warpAffine(img,M,(28,28))
        dst = dst.reshape([28, 28, 1])
        sample_images[i+4] = dst
        angle += 22.5

def brighten(img, amount):
    return np.clip(img * (1 + amount), 0, 1)


def add_random_noise(img, seed, damp=1):
    np.random.seed(seed)
    r = np.random.rand(1,28,28,1)
    r * damp
    return img * r


def plot(img, y, labels):
    for i in range(n_samples):
        plt.imshow(sample_images[i], cmap="binary")
        plt.title("Label:" + labels[y[0]])
        plt.axis("off")
