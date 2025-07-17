### Python script for validating code for kernel ridge regression in KF algorithm
# Goal is to confirm that results from manual kernel ridge regression matches those from the Sci-kit learn package
# Using same validation data that is used in FL algorithm

# Kade ALDRICH
# Internship 2025

import jax.numpy as jnp
import numpy as np
import jax
from jax import grad, make_jaxpr, jit
import random
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import mean_squared_error


######## Option for importing a csv #########
# Loading in experiment data

df_experiment = pd.read_csv("test-functions.csv")
## columns (in order)
# x
# y_smooth
# y_hfreq
# y_bump
# y_rough

################ Specify Y ##################
y = df_experiment['y_smooth']
#############################################
# loading in x for the sake of interpretibility 
x = df_experiment['x']

### Separating data 
# using scikit learn

## Training data (80% of total data)
x_train, x_test, y_train, y_test = train_test_split(
    x, y, # inputing original x and y from csv
    test_size = 0.2, # 80% train, 20% test
    random_state = 51, # setting random seed for this
    shuffle = True # shuffling data because no time dependency
)



## train2 and validation data (80/20 split of training data)
# validation data to be exclusive to the mse prediction penalty at each iteration
# train2 to be exclusive to the rho calculation at each iteration
x_train2, x_validation, y_train2, y_validation = train_test_split(
    x_train, y_train, # inputing original x and y from csv
    test_size = 0.2, # 80% train, 20% test
    random_state = 51, # setting random seed for this
    shuffle = True # shuffling data because no time dependency
)



totalSampleSize = len(df_experiment['x']) # have to set totalsamplesize for rho calculation function

# Setting up data which is used only in rho calculation 
X_1D = jnp.asarray(x_train2.to_numpy(), dtype=jnp.float32)
Y = jnp.asarray(y_train2.to_numpy(), dtype=jnp.float32)

# Setting up data for use in mse calculation 
x_validation = jnp.asarray(x_validation.to_numpy(), dtype=jnp.float32)
y_validation = jnp.asarray(y_validation.to_numpy(), dtype=jnp.float32)


############## Global Parameters  ################

# Kernel Ridge Regression Regularization parameter:
lam = 200  

# RBF kernel parameter:
w = 2 

##################################################


# function for calculating mean squared error of KRR prediction on validation data
def calc_mse(w):
    # calculating kernel gram matrix
    diffs = X_1D[:, None] - X_1D[None, :]
    sqdf = diffs**2  # square diff matrix (nf, nf)
    K_train = jnp.exp(-w * sqdf)
    K_train_reg = K_train + lam * jnp.eye(len(X_1D)) 

    # solving for the weights 
    weights = jnp.linalg.solve(K_train_reg, Y) # here Y and X_1D are the training sample of the training sample

    # predicting unseen validation data
    x_validation_sq = jnp.sum(x_validation**2, axis=1)[:, None] 
    x_train2_sq = np.sum(X_1D**2, axis=1)[None, :]
    sq_dists = x_validation_sq + x_train2_sq - 2 * x_validation @ x_train2.T
    K_test_train = np.exp(-w * sq_dists) # kernel gram matrix of the train2 and validation x's 

    y_validation_pred = K_test_train @ weights # KRR prediction of y validation

    # calculate mean squared error between predicted y and y validation 
    mse = jnp.mean((y_validation - y_validation_pred) ** 2)

    return mse

# function for predicting unseen data using kernel ridge regression 
# isolating this from calc_mse function above 
@jit
def KRR(w, x_train, y_train, x_test):
    # calculating kernel gram matrix
    train_diffs = x_train[:, None] - x_train[None, :]
    train_sqdf = train_diffs**2  # square diff matrix (nf, nf)
    K_train = jnp.exp(-w * train_sqdf) # kernel gram matrix of training data
    K_train_reg = K_train + lam * jnp.eye(len(x_train))  # regularized kernel gram matrix 

    # solving for the weights 
    weights = jnp.linalg.solve(K_train_reg, y_train) 

    # predicting unseen validation data
    x_test_sq = jnp.sum(x_test**2, axis=1)[:, None] 
    x_train2_sq = jnp.sum(x_train**2, axis=1)[None, :]
    sq_dists = x_test_sq + x_train2_sq - 2 * x_test @ x_train.T
    K_test_train = jnp.exp(-w * sq_dists) # kernel gram matrix of the train2 and validation x's 

    y_pred = K_test_train @ weights # KRR prediction of y validation

    return y_pred 