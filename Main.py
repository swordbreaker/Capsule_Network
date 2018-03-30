#import Examples
from Examples import *

tf.logging.set_verbosity(v)

mnist_fashion(epochs=3, train=True, eval=True)