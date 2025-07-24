# Kade ALDRICH
# Internship 2025

import jax.numpy as jnp
import numpy as np
import jax
from jax import grad, make_jaxpr, jit 
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

    # converting to numpy arrays
    x_train = jnp.asarray(x_train)
    y_train = jnp.asarray(y_train)
    x_test  = jnp.asarray(x_test)

    # calculating kernel gram matrix
    train_diffs = x_train[:, None] - x_train[None, :]
    train_sqdf = train_diffs**2  # square diff matrix (nf, nf)
    K_train = jnp.exp(-w * train_sqdf) # kernel gram matrix of training data
    K_train_reg = K_train + lam * jnp.eye(len(x_train), dtype = K_train.dtype)  # regularized kernel gram matrix 

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


def calc_mse_rbf(params, x_tr, x_val, y_tr, y_val):
    lam, gamma = params # two hyperparameters

    # convert to jax arrays
    x_tr, x_val, y_tr, y_val = map(jnp.asarray, (x_tr, x_val, y_tr, y_val))
    
    # calculating kernel gram matrix
    train_diffs = x_tr[:, None] - x_tr[None, :]
    train_sqdf = train_diffs**2  # square diff matrix (nf, nf)
    K_train = jnp.exp(-gamma * train_sqdf) # kernel gram matrix of training data
    K_train_reg = K_train + lam * jnp.eye(len(x_tr))  # regularized kernel gram matrix 

    # solving for the weights 
    weights = solve(K_train_reg, y_tr, assume_a = 'pos', lower=True) # positive definite and symmetric  
    # sym_pos: sym mean symmetric, pos means positive definite --> jax uses Cholesky decomp which is fast and should help with numerical issues

    # predicting unseen validation data
    train_test_diffs = x_val[:, None] - x_tr[None, :] # difference matrix between test and train matrices
    sq_dists = train_test_diffs**2
    K_test_train = jnp.exp(-gamma * sq_dists) # kernel gram matrix of the train2 and validation x's 

    y_pred = K_test_train @ weights # KRR prediction of y validation

    mse = jnp.mean((y_val - y_pred)**2) # calculate mean squared error of kernel ridge regression prediction
    
    return mse

# jit wrappers for functions 
calc_mse_rbf_jit = jit(calc_mse_rbf)
value_and_grad = jit(jax.value_and_grad(calc_mse_rbf_jit))
KRR_jit = jit(KRR)


#################### GRADIENT DESCENT FUNCTION USING JIT ##########################

# Splitting into a gradient step function and gradient descent function 
@jit
def make_gd_step_rbf_mse(params, x_tr, x_val, y_tr, y_val, step_size = 100, step_style = 'fs'):
    
    ###########################################################################################
    #                                                                                         #
    # step_style:                                                                             #
    #   'fs'    |   Fixed step size to be specified with function call (default = 0.2)        #
    #   'ls'    |   Line search for finding optimal step size at each iteration automatically #
    #                                                                                         #
    ###########################################################################################
        
    mse, grad = value_and_grad(params, x_tr, x_val, y_tr, y_val)
    
    if step_style == 'fs':
        params = params - step_size*grad

    return params, mse
    
def run_gd(max_iter, params_init, split_thresh = 1, step_size = 0.2, step_style = 'fs', kernel = 'rbf'):
    traj = np.zeros((max_iter + 1, len(params_init)))
    mse_trace = np.zeros(max_iter + 1)

    params = jnp.asarray(params_init, dtype = jnp.float32) # initializing parameters as np array

    # initial validation split
    x_tr, x_val, y_tr, y_val = get_validation_split()
    x_tr = jnp.asarray(x_tr, dtype = jnp.float32)
    y_tr = jnp.asarray(y_tr, dtype = jnp.float32)
    x_val = jnp.asarray(x_val, dtype = jnp.float32)
    y_val = jnp.asarray(y_val, dtype = jnp.float32) 
    
    for i in range(max_iter):
        # run descent over max_iter iterations
        if i % split_thresh == 0: # if threshold met for splitting the validation data 

            x_tr, x_val, y_tr, y_val = get_validation_split()

            x_tr = jnp.asarray(x_tr)
            y_tr = jnp.asarray(y_tr)
            x_val = jnp.asarray(x_val)
            y_val = jnp.asarray(y_val) 

        params, mse = make_gd_step_rbf_mse(params, x_tr, x_val, y_tr, y_val)

        traj[i+1] = np.asarray(params)
        mse_trace[i+1] = float(mse)


    # return data frame of the results from the gradient descent
    return pd.DataFrame({
        "iteration": np.arange(max_iter + 1), # column for each iteration
        "lambda": traj[:,0], # column for the lambda value at each iteration
        "gamma": traj[:,1], # column for the gamma value at each iteration
        "mse": mse_trace, # column for the calculated mean squared error at each iteration
    })

###################################################################################


# first implementation of gradient descent algorithm using mse as criterion 
# does not use jit 
def grad_desc_fs_2d_rbf(max_iter, params_init, step = 0.2, resample_iter = 1):
    # gradient descent on lambda (Ridge regularization parameter) and gamma (rbf kernel parameter)
    traj = np.zeros((max_iter + 1, 2)) # parameter values across iterations
    mse_trace = np.zeros(max_iter + 1) # mse loss values to be minimized via gradient descent

    # initial values
    traj[0] = np.asarray(params_init)

    print("Initial gamma: ", traj[0, 1])
    print("Initial lambda: ", traj[0, 0])

    # initial validation split
    x_train, x_validation, y_train, y_validation = get_validation_split()

    # make mse calculation function a function of only parameter vector
    def mse_fxn(params):
        return calc_mse_rbf(params, x_tr = x_train, y_tr = y_train, x_val = x_validation, y_val = y_validation)
    
    # setting up jax gradient calculation
    grad_fxn = jax.grad(mse_fxn)
    
    # gradient calculation with initial parameters
    mse_trace[0] = float(mse_fxn(traj[0]))
    
    print("Initial MSE: ", mse_trace[0])


    for i in range(max_iter):
        print("Iteration ", i)
        # for loop controlling gradient descent
        # runs until the maximum number of iterations is reached 

        if i % resample_iter == 0:
            
            # print statement for testing
            print("Resampling valdation data")

            # getting new validation and training split amongst the training data every resample_iter iterations
            # only controls how often we resample the data for the mse calculation
            # larger resample_iter gives nicer trajectory but introduces more bias
            x_train, x_validation, y_train, y_validation = get_validation_split() # gets new split

            # rebuilding mse function to recalculate using the new validation split 
            def mse_fxn(params):
                return calc_mse_rbf(params, x_tr = x_train, y_tr = y_train, x_val = x_validation, y_val = y_validation)

            # rebuilding jax gradient function
            grad_fxn = jax.grad(mse_fxn)

        # compute gradient
        grad = np.asarray(grad_fxn(traj[i]))
        print("gradient: ", grad)
        # compute gradient step
        traj[i+1] = traj[i] - step*grad*1000 # increasing scale by 2 orders of magnitude 

        # get the new mse related to the new parameters from the gradient step
        mse_trace[i+1] = float(mse_fxn(traj[i+1]))

        print(f"gamma {i+1}: {traj[i+1, 1]}")
        print(f"lambda {i+1}: {traj[i+1, 0]}")
    # return data frame of the results from the gradient descent
    return pd.DataFrame({
        "iteration": np.arange(max_iter + 1), # column for each iteration
        "lambda": traj[:,0], # column for the lambda value at each iteration
        "gamma": traj[:,1], # column for the gamma value at each iteration
        "mse": mse_trace, # column for the calculated mean squared error at each iteration
    })




max_iterations = 1000
lam_init = 10
gamma_init = 100
desc_parameters_init = jnp.array([lam_init, gamma_init], dtype = jnp.float32)
split_threshold = 5

# test to see if run_gd works
df1 = run_gd(max_iter = max_iterations, params_init=desc_parameters_init, split_thresh = split_threshold)
df1.to_csv("grad_desc_fs_2d_rbf.csv", index=False)


## running KRR using the gd results
# using testing data set aside at beginning
gamma_gd = df1["gamma"].tail(1).item()
lam_gd = df1["lambda"].tail(1).item()

y_pred = jnp.asarray(KRR(w = gamma_gd, lam = lam_gd, x_train = x_train_first, y_train = y_train_first, x_test = x_test_first)) # converting to numpy array
y_test_first = jnp.asarray(y_test_first)
final_mse = jnp.mean((y_pred - y_test_first)**2)

print(f"Final MSE: {final_mse}")

# plotting 
plt.figure()                       # new figure
plt.plot(x_test_first, y_test_first, 'o', label='True y', markersize=5)
plt.plot(x_test_first, y_pred, 'o', label='Predicted y', markersize=5)
plt.xlabel('x_test')              # label axes
plt.ylabel('y')
plt.title('True Y vs KRR prediction on Test Set using RBF Kernel')
plt.legend(title=f'gamma = {gamma_gd:.3f}, λ = {lam_gd:.3f}, mse = {final_mse:.5f})') 
plt.tight_layout()
plt.show()


# plot for looking at mse trace 
fig, ax = plt.subplots()
ax.plot(df1["iteration"], df1["mse"])
ax.set_xlabel("Iteration")
ax.set_ylabel("Mean Squared Error")
ax.set_title(f"MSE Trace over {max_iterations} steps")
plt.legend(title=f'initial gamma = {gamma_init:.1f}, initial λ = {lam_init:.1f}, split threshold = 5') 
ax.grid(True)  # optional, but often helpful
plt.show()
