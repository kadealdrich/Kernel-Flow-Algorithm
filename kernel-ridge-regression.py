### Python script for validating code for kernel ridge regression in KF algorithm
# Goal is to confirm that results from manual kernel ridge regression matches those from the Sci-kit learn package
# Using same validation data that is used in FL algorithm

# Kade ALDRICH
# Internship 2025

import jax.numpy as jnp
import numpy as np
from jax import grad, make_jaxpr, jit
from jax.scipy.linalg import solve
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import mean_squared_error, r2_score

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
    x_train, y_train, # inputing training split to be split again
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
lam = 100

# RBF kernel parameter:
w = 4

##################################################



# function for predicting unseen data using kernel ridge regression 
# isolating this from calc_mse function above 
@jit
def KRR(w, lam, x_train, y_train, x_test):
    # calculating kernel gram matrix
    train_diffs = x_train[:, None] - x_train[None, :]
    train_sqdf = train_diffs**2  # square diff matrix (nf, nf)
    K_train = jnp.exp(-w * train_sqdf) # kernel gram matrix of training data
    K_train_reg = K_train + lam * jnp.eye(len(x_train))  # regularized kernel gram matrix 

    # solving for the weights 
    weights = solve(K_train_reg, y_train, assume_a = 'pos', lower=True) # positive definite and symmetric  
    # sym_pos: sym mean symmetric, pos means positive definite --> jax uses Cholesky decomp which is fast and should help with numerical issues

    # predicting unseen validation data
    train_test_diffs = x_test[:, None] - x_train[None, :] # difference matrix between test and train matrices
    sq_dists = train_test_diffs**2
    K_test_train = jnp.exp(-w * sq_dists) # kernel gram matrix of the train2 and validation x's 

    y_pred = K_test_train @ weights # KRR prediction of y validation

    return y_pred 


KRR_pred = KRR(w = 4, lam = 200, x_train = X_1D, y_train = Y, x_test = x_validation)


# function for calculating mean squared error of KRR prediction on validation data
# uses global variables for test y values and predicted y values so that jax can be used for gradient calculation in the future
# all the same as the KRR function except for it returns the prediction mean squared error
@jit
def calc_mse(w):
    # calculating kernel gram matrix
    train_diffs = X_1D[:, None] - X_1D[None, :]
    train_sqdf = train_diffs**2  # square diff matrix (nf, nf)
    K_train = jnp.exp(-w * train_sqdf) # kernel gram matrix of training data
    K_train_reg = K_train + lam * jnp.eye(len(X_1D))  # regularized kernel gram matrix 

    # solving for the weights 
    weights = solve(K_train_reg, Y, assume_a = 'pos', lower=True) # positive definite and symmetric  
    # sym_pos: sym mean symmetric, pos means positive definite --> jax uses Cholesky decomp which is fast and should help with numerical issues

    # predicting unseen validation data
    train_test_diffs = x_validation[:, None] - X_1D[None, :] # difference matrix between test and train matrices
    sq_dists = train_test_diffs**2
    K_test_train = jnp.exp(-w * sq_dists) # kernel gram matrix of the train2 and validation x's 

    y_pred = K_test_train @ weights # KRR prediction of y validation

    mse = jnp.mean((y_validation - y_pred)**2) # calculate mean squared error of kernel ridge regression prediction
    
    return mse




#####################  Experimentation  ######################
# use x train and y train to train the model and calc r2 
# compare y pred to y test via mse and graph 
# make sure scikit learn mse and my mse match up 

mse_manual = calc_mse(w)



## Validating using Sklearn 
mod = KernelRidge(kernel = 'rbf', alpha= lam, gamma = w)

x_train_reshaped = X_1D.reshape(-1,1)
x_validation_reshaped = x_validation.reshape(-1,1)

mod.fit(x_train_reshaped, Y)

y_pred_sklearn = mod.predict(x_validation_reshaped)

mse_sklearn = mean_squared_error(y_validation, y_pred_sklearn)
print("MSE from SKlearn= ", mse_sklearn)
print("MSE manual= ", mse_manual)


# plotting 
plt.figure()                       # new figure
plt.plot(x_validation, y_validation, 'o', label='True y', markersize=5)
plt.plot(x_validation, KRR_pred, 'o', label='Predicted y Manual', markersize=5)
plt.plot(x_validation, y_pred_sklearn, 'o', label='Prediected y SKlearn', markersize=5)
plt.plot(X_1D, Y, 'o', label='Training data', markersize=5)
plt.xlabel('x_test')              # label axes
plt.ylabel('y')
plt.title('True Y vs KRR prediction on Test Set')
plt.legend(title=f'w = {w}, λ = {lam}, mse = {mse_manual:.5f})') # rounding mse to 5 digits
plt.tight_layout()
plt.show()