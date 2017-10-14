import em
import numpy as np
import random


data = np.random.rand(20, 30, 5)
F = np.random.rand(10, 10)
B =np.random.rand(20, 30)
print(np.ones_like(B) - B.max())
