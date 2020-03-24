import numpy as np

x = np.arange(24).reshape(8,3)
print(np.argmax(x, axis=1))
