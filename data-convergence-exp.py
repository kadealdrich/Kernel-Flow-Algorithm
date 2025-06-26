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
y_smooth = np.sin(math.pi * x) + 0.5 * np.cos(3 * math.pi * x)

# y with bump
y_bump = np.exp(-10*(x)**2)

# random analytic y 
m = 30
a_vec   = np.random.uniform(-1.0, 1.0, size=m)
w_vec   = np.random.uniform(-1.0, 1.0, size=m)
phi_vec = np.random.uniform(-np.pi, np.pi, size=m) # phase in radians

y_RF = np.zeros_like(x) # initialize y 
for i in range(n):
    new_el = 0
    for j in range(m): # build y iteritively 
        new_el += a_vec[j] * np.cos(w_vec[j] * x[i] + phi_vec[j]) # use np.cos for elementwise operation
    y_RF[i] = new_el

# Plot
plt.plot(x, y_RF)
plt.title("Random Fourier Function")
plt.xlabel("x")
plt.ylabel("f(x)")
plt.grid(True)
plt.show()

# rough y 
y_rough = abs(x)

