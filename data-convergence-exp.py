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

# random error term of ~2% total variation
epsilon = np.random.normal(-0.04, 0.04, size = n)

# smooth y 
y_smooth_true = np.cos(0.5*math.pi + 0.5* math.pi * x)
y_smooth = y_smooth_true + epsilon

# smooth but high frequency y
y_hfreq_true = np.sin(6*math.pi*x) + 0.3*np.sin(20*math.pi*x) + 0.25*np.cos(50*math.pi*x)
y_hfreq = y_hfreq_true + epsilon

# y with bump
y_bump_true = np.exp(-10*(x)**2)
y_bump = y_bump_true + epsilon

# rough y 
y_rough_true = abs(x)
y_rough = y_rough_true + epsilon

# plotting functions 
plt.figure()
plt.plot(x, y_smooth, linestyle='-')
plt.xlabel("x")
plt.ylabel("rough y (true)")
plt.title("Test")
plt.grid(True)
plt.show()