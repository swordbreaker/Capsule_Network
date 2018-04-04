#import Examples
from Examples import *

tf.logging.set_verbosity(tf.logging.ERROR)

mnist_fashion(epochs=5, train=True, eval=True)

#mnist(train=False, eval=False)

#mushroom_example(train=True, eval=True, epochs=4)