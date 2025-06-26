import random as rd 
import numpy as np
import math 
import matplotlib.pyplot as plt
np.random.seed(51)

### Script for constructing data for KF experiments

# x's are sampled randomly from a uniform[-3,3] distribution 
n = 1000
x = np.random.uniform(-3.0, 3.0, size = n)
x = np.sort(x)

# smooth y 
y_smooth = np.cos(math.pi * x)

# smooth but high frequency y
y_hfreq = np.sin(6*math.pi*x) + 0.3*np.sin(20*math.pi*x) + 0.25*np.cos(50*math.pi*x)

# y with bump
y_bump = np.exp(-10*(x)**2)


# rough y 
y_rough = abs(x)

