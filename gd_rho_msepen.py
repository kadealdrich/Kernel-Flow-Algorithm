### Gradient desecent algorithm for kernel parameter optimzation using rho criterion with validation KRR MSE as penalty 

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
from functools import partial

random.seed(51)
key_init = jax.random.PRNGKey(51) # jax key for getting random permutation of coarse indices


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


# function for getting new split of validation and training data for mse criterion calculation 
def get_validation_split():
    # use global variable for training data to be split
    # return a new training and validation split
    return train_test_split(
        x_train_first, y_train_first, # splitting the original training data
        test_size = 0.2,
        shuffle=True
    )


# function for getting new coarse and fine samples for rho criterion
def get_coarse_indices(key, n_train, coarse_prop = 0.5):
    # ASSUMES FINE SAMPLE IS ALL TRAINING DATA 
    # gets a random coarse sample equal to a random selection of round(0.5 * len(fine_sample)) indices
    key, subkey = jax.random.split(key)
    n_coarse = jnp.maximum(1, jnp.round(coarse_prop*n_train).astype(jnp.int32))
    perm = jax.random.permutation(subkey, n_train) # gets random permutation 
    coarse_indices = perm[:n_coarse]
    return coarse_indices, key # anytime this is called have to update key value


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


# function for calculating Normlized and bounded mse 
@jit
def calc_nmse_rbf(params, x_tr, x_val, y_tr, y_val):
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
    
    var = jnp.var(y_tr) + 1e-8 # adds epsilon because this goes in denominator
    nmse = mse / var
    bounded_nmse = 1.0 - jnp.exp(-nmse)

    return bounded_nmse


# function for calculating rho criterion 
def calc_rho_rbf(params, x_fine, x_coarse, y_fine, y_coarse):
    lam, gamma = params # two hyperparameters

    # converting to jax numpy arrays 
    x_f, x_c, y_f, y_c = map(jnp.asarray, (x_fine, x_coarse, y_fine, y_coarse))

    ## constructing kernel Gram matrices

    # calculating difference matrices
    diffs_f = x_f[:, None] - x_f[None:,]
    diffs_c = x_c[:, None] - x_c[None:,]

    sqdf = jnp.square(diffs_f)
    sqdc = jnp.square(diffs_c)

    Kf = jnp.exp(-gamma * sqdf) # fine sample kernel Gram matrix
    Kc = jnp.exp(-gamma * sqdc) # coarse sample kernel Gram matrix

    Kf_reg = Kf + lam * jnp.eye(len(x_fine)) # regularized kernel Gram matrices
    Kc_reg = Kc + lam * jnp.eye(len(x_coarse))

    # calculating rho 
    # compute (Reg)^{-1} and then its square
    inv_f = jnp.linalg.inv(Kf_reg)
    inv_c = jnp.linalg.inv(Kc_reg)

    Kf_inv2 = inv_f @ inv_f
    Kc_inv2 = inv_c @ inv_c

    # quadratic forms
    N = y_c.T @ (Kc @ Kc_inv2) @ y_c # numerator
    D = y_f.T @ (Kf @ Kf_inv2) @ y_f # denominator

    # final rho
    ### Not sure if should multiply by nf/nc or not 
    rho = 1.0 - (N / D)
    #print("rho=", rho, ", w=", w)

    return rho


# function for getting rho penalized by mse 
def calc_pen_crit(params, x_fine, x_coarse, y_fine, y_coarse, x_tr, x_val, y_tr, y_val, mse_weight = 0.5):
    rho = calc_rho_rbf(params, x_fine, x_coarse, y_fine, y_coarse)
    nmse = calc_nmse_rbf(params, x_tr, x_val, y_tr, y_val)

    print(f"rho={rho} | nmse={nmse}")
    return 0.5*rho + mse_weight*nmse
    

# jit wrappers for functions 
calc_crit_rbf_jit = jit(calc_pen_crit)
value_and_grad = jit(jax.value_and_grad(calc_crit_rbf_jit))
KRR_jit = jit(KRR)


# function for getting gradient step using rho criterion penalized by mse
@partial(jit, static_argnames=("step_style",))
def make_gd_step_rbf_pen_crit(params, x_tr, x_val, y_tr, y_val, x_fine, x_coarse, y_fine, y_coarse, mse_weight, step_size = 100, step_style = 'fs'):
    
    ###########################################################################################
    #                                                                                         #
    # step_style:                                                                             #
    #   'fs'    |   Fixed step size to be specified with function call (default = 0.2)        #
    #   'ls'    |   Line search for finding optimal step size at each iteration automatically #
    #                                                                                         #
    ###########################################################################################
        
    crit, grad = value_and_grad(params, x_fine, x_coarse, y_fine, y_coarse, x_tr, x_val, y_tr, y_val, mse_weight)
    
    if step_style == 'fs':
        params = params - step_size*grad

    return params, crit


# function for running gradient descent
def run_gd(max_iter, params_init, key, mse_weight, step_size, split_thresh = 1, step_style = 'fs', kernel = 'rbf'):
    traj = np.zeros((max_iter + 1, len(params_init)))
    crit_trace = np.zeros(max_iter + 1)
    params = jnp.asarray(params_init, dtype = jnp.float32) # initializing parameters as np array

    # initial validation split
    x_tr, x_val, y_tr, y_val = get_validation_split()
    x_fine = x
    y_fine = y

    x_tr = jnp.asarray(x_tr, dtype = jnp.float32)
    y_tr = jnp.asarray(y_tr, dtype = jnp.float32)
    x_val = jnp.asarray(x_val, dtype = jnp.float32)
    y_val = jnp.asarray(y_val, dtype = jnp.float32) 
    x_fine = jnp.asarray(x_fine, dtype=jnp.float32)
    y_fine = jnp.asarray(y_fine, dtype=jnp.float32)
    

    coarse_indices, key = get_coarse_indices(key=key, n_train = len(x_fine))

    # get the coarse subset for calculating rho 
    x_coarse = x_fine[coarse_indices]
    y_coarse = y_fine[coarse_indices]

    # ensure they are jnp arrays of specific float datatype
    x_coarse = jnp.asarray(x_coarse, dtype=jnp.float32)
    y_coarse = jnp.asarray(y_coarse, dtype=jnp.float32)
    
    # set initial parameter values in trajectory
    traj[0,:] = jnp.asarray(params_init, dtype=jnp.float32)
    
    # calculate initial crit
    init_crit = calc_pen_crit(params= params_init, x_fine=x_fine, x_coarse=x_coarse, y_fine = y_fine, y_coarse = y_coarse, x_tr = x_tr, x_val = x_val, y_tr = y_tr, y_val = y_val)
    crit_trace[0] = init_crit

    for i in range(1, max_iter):
    # run descent over max_iter iterations
        if i % split_thresh == 0: # if threshold met for splitting the validation data 

            x_tr, x_val, y_tr, y_val = get_validation_split()

            x_tr = jnp.asarray(x_tr)
            y_tr = jnp.asarray(y_tr)
            x_val = jnp.asarray(x_val)
            y_val = jnp.asarray(y_val) 

        if kernel == 'rbf':
            params, crit = make_gd_step_rbf_pen_crit(params, x_tr, x_val, y_tr, y_val, x_fine, x_coarse, y_fine, y_coarse, mse_weight, step_size, step_style)

        traj[i+1] = np.asarray(params)
        crit_trace[i+1] = float(crit)


    # return data frame of the results from the gradient descent
    return pd.DataFrame({
        "iteration": np.arange(max_iter + 1), # column for each iteration
        "lambda": traj[:,0], # column for the lambda value at each iteration
        "gamma": traj[:,1], # column for the gamma value at each iteration
        "criterion": crit_trace, # column for the calculated mean squared error at each iteration
    })


lam_init = 1
gamma_init = 1
desc_parameters_init = jnp.array([lam_init, gamma_init], dtype = jnp.float32)
split_threshold = 5

# running gradient descent algorithm
df = run_gd(max_iter = 1000,
       params_init = desc_parameters_init, 
       key = key_init, 
       mse_weight = 1,  
       step_size = 0.2,
       split_thresh = 1,
       step_style='fs',
       kernel = 'rbf'
       )   

# df.to_csv("gd_mse_pen.csv", index=False)

# plot for looking at mse trace 
fig, ax = plt.subplots()
ax.plot(df["iteration"], df["criterion"])
ax.set_xlabel("Iteration")
ax.set_ylabel("Rho with MSE Penalty")
ax.set_title(f"Criterion Trace")
plt.legend(title=f'initial gamma = {gamma_init:.1f}, initial λ = {lam_init:.1f}, split threshold = 1') 
ax.grid(True)  # optional, but often helpful
plt.show()

# plot for looking at mse trace 
fig, ax = plt.subplots()
ax.plot(df["iteration"], df["lambda"])
ax.plot(df["iteration"], df["gamma"])
ax.set_xlabel("Iteration")
ax.set_ylabel("Parameter Values")
ax.set_title(f"Trace of Parameters")
plt.legend(title=f'initial gamma = {gamma_init:.1f}, initial λ = {lam_init:.1f}, split threshold = 1') 
ax.grid(True)  # optional, but often helpful
plt.show()

# assessing gradient blow up or NaN
finite = np.isfinite(df['criterion'])
last_good = df.loc[finite, 'iteration'].max()
# Looking for 17-18
print(f"Last finite criterion at iteration {last_good}")
print("Number of NaNs:", (~finite).sum())

# looking at fit of final KRR
## running KRR using the gd results
# using testing data set aside at beginning
gamma_gd = df["gamma"].tail(1).item()
lam_gd = df["lambda"].tail(1).item()

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