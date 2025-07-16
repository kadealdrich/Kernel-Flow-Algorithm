import random as rd 
import numpy as np
import math 
import matplotlib.pyplot as plt
import pandas as pd
np.random.seed(51)

### Script for constructing data for KF experiments

# x's are sampled randomly from a uniform[-1,1] distribution 
n = 1000
x = np.random.uniform(-1.0, 1.0, size = n)
x = np.sort(x)

# defining random error terms for each function 
# Balancing magnitude of error and complexity of the function

# random error term for smooth function 
epsilon_smooth = np.random.normal(0, 0.25, size = n)

# random error term for high frequency function 
epsilon_hfreq = np.random.normal(0, 0.1, size = n)

# random error term for y with bump 
epsilon_bump = np.random.normal(0, 0.25, size = n)

# random error term for y rough 
epsilon_rough = np.random.normal(0, 0.15, size = n)

# smooth y 
y_smooth_true = np.cos(0.5*math.pi + math.pi * x)
y_smooth = y_smooth_true + epsilon_smooth

# smooth but high frequency y
y_hfreq_true = np.sin(6*math.pi*x) + 0.3*np.sin(20*math.pi*x) + 0.25*np.cos(50*math.pi*x)
y_hfreq = y_hfreq_true + epsilon_hfreq

# y with bump
y_bump_true = np.exp(-10*(x)**2)
y_bump = y_bump_true + epsilon_bump

# rough y 
y_rough_true = abs(x)
y_rough = y_rough_true + epsilon_rough

## Smooth function
# plotting y smooth with error
plt.figure()
plt.plot(x, y_smooth, linestyle='-')
plt.xlabel("x (1000 rand unif samples)")
plt.ylabel("y")
plt.title("y_smooth with Normal error SD = 0.25")
plt.grid(True)
plt.show()

## Rough function 
# plotting y rough with error
plt.figure()
plt.plot(x, y_rough, linestyle='-')
plt.xlabel("x (1000 rand unif samples)")
plt.ylabel("y")
plt.title("y_rough with Normal error SD = 0.15")
plt.grid(True)
plt.show()

## High frequency function 
# plotting y rough with error
plt.figure()
plt.plot(x, y_hfreq, linestyle='-')
plt.xlabel("x (1000 rand unif samples)")
plt.ylabel("y")
plt.title("y_hfreq with Normal error SD = 0.1")
plt.grid(True)
plt.show()

## Bump function 
# plotting y bump with error
plt.figure()
plt.plot(x, y_bump, linestyle='-')
plt.xlabel("x (1000 rand unif samples)")
plt.ylabel("y")
plt.title("y_bump with Normal error SD = 0.25")
plt.grid(True)
plt.show()


## Exporting data as csv
data = pd.DataFrame({
    "x":             x,
    "y_smooth":      y_smooth,
    "y_hfreq":       y_hfreq,
    "y_bump":        y_bump,
    "y_rough":       y_rough,
})
data.to_csv("test-functions.csv", index=False)