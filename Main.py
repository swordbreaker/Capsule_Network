#import Examples
from Examples import *
from ticketing_examples import *

tf.logging.set_verbosity(tf.logging.ERROR)

#ticketing(train=True, eval=True, restore_checkpoint=True, epochs=10, batch_size = 50)

#mnist_fashion(epochs=10, train=True, eval=True, reconstruct=False, transform=False)

mnist(train=True, eval=True, reconstruct=False, epochs=10)

#mushroom_example(train=True, eval=True, epochs=4)
#mushroom_example2(train=True, eval=True, epochs=4)