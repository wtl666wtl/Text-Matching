import numpy as np
import torch

a = np.array([1,1,1,1,0,0])
b = a[:,None] < a[None,:]
b = b.astype(float)
print((1-b)*1e12)