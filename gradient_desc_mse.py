# Kade ALDRICH
# Internship 2025

import jax.numpy as jnp
import numpy as np
import jax
from jax import grad, make_jaxpr
from jax.scipy.linalg import solve
import random
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import mean_squared_error

random.seed(51)

# loading in the data 
df_experiment = pd.read_csv("test-functions.csv")
y = df_experiment['y_smooth'] # using smooth cosine y 
x = df_experiment['x']

# original training and testing split
# testing data not to be touched until validation the results from the gradient descent 
x_train_first, x_test_first, y_train_first, y_test_first = train_test_split(
    x, y, # inputing original x and y from csv
    test_size = 0.2, # 80% train, 20% test
    random_state = 51, # setting random seed for this
    shuffle = True # shuffling data because no time dependency
)


def get_validation_split():
    # use global variable for training data to be split
    # return a new training and validation split
    return train_test_split(
        x_train_first, y_train_first, # splitting the original training data
        test_size = 0.2,
        shuffle=True
    )


# function for predicting unseen data using kernel ridge regression 
# isolating this from calc_mse function above 
def KRR(w, lam, x_train, y_train, x_test):
    # calculating kernel gram matrix
    train_diffs = x_train[:, None] - x_train[None, :]
    train_sqdf = train_diffs**2  # square diff matrix (nf, nf)
    K_train = jnp.exp(-w * train_sqdf) # kernel gram matrix of training data
    K_train_reg = K_train + lam * jnp.eye(len(x_train))  # regularized kernel gram matrix 

    # solving for the weights 
    weights = solve(K_train_reg, y_train, assume_a = 'pos', lower = True) # positive definite and symmetric
    # set argurments in solve for optimized solve: 
    #   assume_a = 'pos', lower=True
      
    # sym_pos: sym mean symmetric, pos means positive definite --> jax uses Cholesky decomp which is fast and should help with numerical issues

    # predicting unseen validation data
    train_test_diffs = x_test[:, None] - x_train[None, :] # difference matrix between test and train matrices
    sq_dists = train_test_diffs**2
    K_test_train = jnp.exp(-w * sq_dists) # kernel gram matrix of the train2 and validation x's 

    y_pred = K_test_train @ weights # KRR prediction of y validation

    return y_pred 

# function for calculating mean squared error of KRR prediction on validation data
# uses global variables for test y values and predicted y values so that jax can be used for gradient calculation in the future
# all the same as the KRR function except for it returns the prediction mean squared error
lam = 100
gamma = 5
desc_parameters_init = [lam, gamma]
def calc_mse(desc_parameters):

    # get training and validation split
    x_train, x_validation, y_train, y_validation = get_validation_split()
    # calculating kernel gram matrix
    train_diffs = x_train[:, None] - x_train[None, :]
    train_sqdf = train_diffs**2  # square diff matrix (nf, nf)
    K_train = jnp.exp(-gamma * train_sqdf) # kernel gram matrix of training data
    K_train_reg = K_train + lam * jnp.eye(len(x_train))  # regularized kernel gram matrix 

    # solving for the weights 
    weights = solve(K_train_reg, y_train, assume_a = 'pos', lower=True) # positive definite and symmetric  
    # sym_pos: sym mean symmetric, pos means positive definite --> jax uses Cholesky decomp which is fast and should help with numerical issues

    # predicting unseen validation data
    train_test_diffs = x_validation[:, None] - x_train[None, :] # difference matrix between test and train matrices
    sq_dists = train_test_diffs**2
    K_test_train = jnp.exp(-gamma * sq_dists) # kernel gram matrix of the train2 and validation x's 

    y_pred = K_test_train @ weights # KRR prediction of y validation

    mse = jnp.mean((y_validation - y_pred)**2) # calculate mean squared error of kernel ridge regression prediction
    
    return mse

    