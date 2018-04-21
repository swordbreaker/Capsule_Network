#import Examples
from Examples import *

tf.logging.set_verbosity(tf.logging.ERROR)

mnist_fashion(epochs=1, train=True, eval=False)

#mnist(train=False, eval=False)

#mushroom_example(train=True, eval=True, epochs=4)
#mushroom_example2(train=True, eval=True, epochs=4)