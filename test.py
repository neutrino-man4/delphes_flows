from scipy.stats import wasserstein_distance
import numpy as np
import tensorflow as tf
a=np.random.rand(100,300)
b=np.random.rand(100,300)
wd=[]
for i in range(b.shape[-1]):
    wd.append(wasserstein_distance(a[:,i],b[:,i]))
wd=tf.stack(wd)
print(wd.shape)