### Python file for kernel flow implementation 
# Kade ALDRICH
# Internship 2025

import jax.numpy as jnp
import numpy as np
import jax
from jax import grad, make_jaxpr
import random
import pandas as pd 
import matplotlib.pyplot as plt

random.seed(51)

#######################################################################
## Parameters of Interest
lam = 0.1

#######################################################################



#######################################################################
## Initializing data
# Creating experimental data with quadratic relationship 
# data needs to be visible to calc rho function at all times

# starting with 1 dimensional input data 
# generating x's from N(0,1)
#totalSampleSize = 200
#X_1D = np.random.normal(loc = 0, scale = 1, size = totalSampleSize)

# creating output data 
# Y is quadratically related to X

a = 2.0
b = -1.0
c = 0.5

# Noise is N(0,1)
#epsilon = np.random.normal(loc = 0.0, scale = 1, size = X_1D.shape)

# calculate Y vector
#Y = a * X_1D**2 + b * X_1D + c + epsilon


######## Option for importing a csv #########
# Loading in experiment data

df_experiment = pd.read_csv("kf_experiment_data.csv")
## columns (in order)
# x
# y_smooth_true
# y_smooth
# y_hfreq_true
# y_hfreq
# y_bump_true
# y_bump
# y_rough_true
# y_rough

totalSampleSize = len(df_experiment['x']) # have to set totalsamplesize for rho calculation function
X_1D = df_experiment['x']
Y = df_experiment['smooth'] # take 'column' from above as appropriate
tv = max(Y) - min(Y) # calculate total variation for constructing error term
epsilon = np.random.normal(0, 0.05 * tv, size = len(Y))
Y += epsilon
#######################################################################


## Function for calculating rho
# Needs to only take kernel parameters w as an input
# matrix multiplication if w a vector and multiplication if w a scalar
# make data globally accessible so that it doesn't have to be passed in
# define kernel matrices as a function of w

def calc_rho(w):
    # sampling done within for loop
    
    # getting fine sample
    fineProportion = 1.0 # proportion of pool to use for fine sample (default is 1)
    fineIndices = sorted(np.random.choice(range(totalSampleSize), size = round(totalSampleSize * fineProportion), replace = False))
    fineSample = X_1D[fineIndices]

    #print("fine indices", fineIndices)
    #print("fine length", len(fineIndices))

    # coarse sample
    coarseProportion = 0.5 # testing if coarse and samples the same, if rho = 0
    #coarseIndices = fineIndices
    #coarseSample = fineSample
    coarseIndices = sorted(np.random.choice(fineIndices, size = round(len(fineIndices) * coarseProportion), replace = False))
    coarseSample = X_1D[coarseIndices] # get directly from total sample pool because indices are necessarily in fine sample
    
    # Debugging
    #print(coarseIndices[0])
    #print("Coarse Indices", coarseIndices)
    #print("coarse length", len(coarseIndices))

    # get correponding Y samples
    # Y fine sample
    Yf = Y[fineIndices]

    # Y coarse sample
    Yc = Y[coarseIndices]

    ## Calculate kernel matrices using RBF kernel
    # fine sample 
    diffs_f = fineSample[:, None] - fineSample[None, :]
    sqdf = diffs_f**2  # square diff matrix (nf, nf)
    Kf = jnp.exp(-w * sqdf)

    # coarse sample 
    diffs_c = coarseSample[:, None] - coarseSample[None, :]
    sqdc = diffs_c**2  # square diff matrix (nf, nf)
    Kc = jnp.exp(-w * sqdc)

    # Calculate rho 
    # sizes
    nf = len(fineIndices)
    nc = len(coarseIndices)
    #print("coarse ind len: ", len(coarseIndices)) debugging
    # print(Kc.shape) So Kc is the size of the fine matrix
    # build regularized matrices
    Kf_reg = Kf + lam * jnp.eye(nf) 
    Kc_reg = Kc + lam * jnp.eye(nc)

    # compute (Reg)^{-1} and then its square
    inv_f = jnp.linalg.inv(Kf_reg)
    inv_c = jnp.linalg.inv(Kc_reg)

    Kf_inv2 = inv_f @ inv_f
    Kc_inv2 = inv_c @ inv_c

    # quadratic forms
    N = Yc.T @ (Kc @ Kc_inv2) @ Yc # numerator
    D = Yf.T @ (Kf @ Kf_inv2) @ Yf # denominator

    # final rho
    ### Not sure if should multiply by nf/nc or not 
    rho = 1.0 - (N / D)

    return rho
    #return rho**2


## Handling multiple samples 
## Kernel flow algorithm for gradient descent across multiple samples of data 

# function for calculating vector of rhos for multiple samples 
def calc_rho_avg(w):

    # setting number of subsamples
    n_subsamples = 30

    # initializing array of rhos 
    rho_vec = jnp.zeros(n_subsamples)

    # sampling rules
    fineProportion = 1.0 # proportion of pool to use for fine sample (default is 1)
    coarseProportion = 0.5 # testing if coarse and samples the same, if rho = 0

    for i in range(n_subsamples):
        # sampling done within for loop
        # getting fine sample
        fineIndices = sorted(np.random.choice(range(totalSampleSize), size = round(totalSampleSize * fineProportion), replace = False))
        fineSample = X_1D[fineIndices]

        #print("fine indices", fineIndices)
        #print("fine length", len(fineIndices))

        # getting coarse sample
        #coarseIndices = fineIndices
        #coarseSample = fineSample
        coarseIndices = sorted(np.random.choice(fineIndices, size = round(len(fineIndices) * coarseProportion), replace = False))
        coarseSample = X_1D[coarseIndices] # get directly from total sample pool because indices are necessarily in fine sample
        
        # Debugging
        #print(coarseIndices[0])
        #print("Coarse Indices", coarseIndices)
        #print("coarse length", len(coarseIndices))

        # get correponding Y samples
        # Y fine sample
        Yf = Y[fineIndices]

        # Y coarse sample
        Yc = Y[coarseIndices]

        ## Calculate kernel matrices using RBF kernel
        # fine sample 
        diffs_f = fineSample[:, None] - fineSample[None, :]
        sqdf = diffs_f**2  # square diff matrix (nf, nf)
        Kf = jnp.exp(-w * sqdf)

        # coarse sample 
        diffs_c = coarseSample[:, None] - coarseSample[None, :]
        sqdc = diffs_c**2  # square diff matrix (nf, nf)
        Kc = jnp.exp(-w * sqdc)

        # Calculate rho 
        # sizes
        nf = len(fineIndices)
        nc = len(coarseIndices)
        #print("coarse ind len: ", len(coarseIndices)) debugging
        # print(Kc.shape) So Kc is the size of the fine matrix
        # build regularized matrices
        Kf_reg = Kf + lam * jnp.eye(nf) 
        Kc_reg = Kc + lam * jnp.eye(nc)

        # compute (Reg)^{-1} and then its square
        inv_f = jnp.linalg.inv(Kf_reg)
        inv_c = jnp.linalg.inv(Kc_reg)

        Kf_inv2 = inv_f @ inv_f
        Kc_inv2 = inv_c @ inv_c

        # quadratic forms
        N = Yc.T @ (Kc @ Kc_inv2) @ Yc # numerator
        D = Yf.T @ (Kf @ Kf_inv2) @ Yf # denominator

        # final rho
        ### Not sure if should multiply by nf/nc or not 
        rho = 1.0 - (N / D)
        rho_vec = rho_vec.at[i].set(rho)

    return np.mean(rho_vec) # returns avg of vector of rhos

## Gradient descent function for multiple rhos 
def grad_desc_multi(max_iter, w_init):

    rho_ts = np.zeros(max_iter + 1) # initialize array holding rho values at each step
    w_ts = np.zeros(max_iter + 1) # initialize array to hold parameter values at each step (1d) 

    rho_ts[0] = calc_rho_avg(w_init) # calculating initial rho value
    w_ts[0] = w_init # making first entry the initial parameters 

    for i in range(max_iter):
        calc_grad = jax.grad(calc_rho_avg) # compute gradient of rho 
        grad = calc_grad(w_ts[i]) # calculate gradient at current parameters

        step = w_ts[i] * 0.5 # start with stepsize equal to half of the parameter value 
        shrink = 0.5

        # line search for stepsize
        while True:
            w_trial = w_ts[i] - step * grad
            rho_trial = calc_rho_avg(w_trial)
            if w_trial < 0: # make sure parameter gamma can't be less than 0 
                w_trial = 0
                break
            if rho_trial < rho_ts[i]:
                break
            step *= shrink
            if step < 1e-8:
                w_trial = w_ts[i]
                rho_trial = rho_ts[i]
                break

        w_ts[i + 1] = w_trial
        rho_ts[i + 1] = rho_trial
        #print("rho", i+1, ": ", rho_ts[i+1])
    
    #print("rhos: ", rho_ts)
    #print("Final rho: ", rho_ts[max_iter]) # printing final rho value
    #print("Starting gamma: ", w_init)
    #print("Final gamma: ", w_ts[max_iter])
    # return data frame of parameters and rho values

    #df = pd.DataFrame({
    #    'iteration': np.arange(max_iter + 1),
    #    'gamma': w_ts,
    #    'rho^2': rho_ts
    #    })

    # testing if starting parameter changes final parameter and rho value much 

    row = pd.DataFrame({ # return single row data frame of starting gamma, final gamma, and final rho 
        'gamma_init': [w_init],
        'gamma_final': [w_ts[max_iter]],
        'rho_init': [rho_ts[0]],
        'rho_final': [rho_ts[max_iter]]
    })

    return row

df = grad_desc_multi(max_iter = 20, w_init = 10)

print(df)

## Function for handling gradient descent 
def grad_desc(max_iter, w_init):

    rho_ts = np.zeros(max_iter + 1) # initialize array holding rho values at each step
    w_ts = np.zeros(max_iter + 1) # initialize array to hold parameter values at each step (1d) 

    rho_ts[0] = calc_rho(w_init) # calculating initial rho value
    w_ts[0] = w_init # making first entry the initial parameters 

    for i in range(max_iter):
        calc_grad = jax.grad(calc_rho) # compute gradient of rho 
        grad = calc_grad(w_ts[i]) # calculate gradient at current parameters

        step = w_ts[i] * 0.5 # start with stepsize equal to half of the parameter value 
        shrink = 0.5

        # line search for stepsize
        while True:
            w_trial = w_ts[i] - step * grad
            rho_trial = calc_rho(w_trial)
            if w_trial < 0: # make sure parameter gamma can't be less than 0 
                w_trial = 0
                break
            if rho_trial < rho_ts[i]:
                break
            step *= shrink
            if step < 1e-8:
                w_trial = w_ts[i]
                rho_trial = rho_ts[i]
                break

        w_ts[i + 1] = w_trial
        rho_ts[i + 1] = rho_trial
        #print("rho", i+1, ": ", rho_ts[i+1])
    
    #print("rhos: ", rho_ts)
    #print("Final rho: ", rho_ts[max_iter]) # printing final rho value
    #print("Starting gamma: ", w_init)
    #print("Final gamma: ", w_ts[max_iter])
    # return data frame of parameters and rho values

    #df = pd.DataFrame({
    #    'iteration': np.arange(max_iter + 1),
    #    'gamma': w_ts,
    #    'rho^2': rho_ts
    #    })

    # testing if starting parameter changes final parameter and rho value much 

    row = pd.DataFrame({ # return single row data frame of starting gamma, final gamma, and final rho 
        'gamma_init': [w_init],
        'gamma_final': [w_ts[max_iter]],
        'rho_init': [rho_ts[0]],
        'rho_final': [rho_ts[max_iter]]
    })

    return row


# making plot of descent of rho^2
# plt.figure(figsize = (8,5))
# plt.plot(df['iteration'], df['rho^2'], marker = 'o', linestyle = '-')
# plt.xlabel('iteration')
# plt.ylabel('rho^2')
# plt.title('Gradient Descent of rho^2 Based on RBF Kernel Parameter gamma')
# plt.show()

# making plot of gamma 
# plt.figure(figsize = (8,5))
# plt.plot(df['iteration'], df['gamma'], marker = 'o', linestyle = '-')
# plt.xlabel('iteration')
# plt.ylabel('gamma')
# plt.title('Gradient Descent of gamma')
# plt.show()

## testing to see how starting gamma effects rho and ending gamma
# for i in range(200):
#     # getting random starting gamma
#     u = np.random.uniform(-3, 3) 
#     gamma_rand = 10**u

#     # get new row of starting gamma, ending gamma, ending rho
#     new_row = grad_desc(max_iter = 25, w_init = gamma_rand)

#     if i == 0:
#         df = new_row
#     else:
#         df = pd.concat([df, new_row], ignore_index=True)

# df.to_csv("gamma-vs-rho.csv",          # file name or full path
#           index=False,            # don’t write the row index column
#           header=True,            # keep column names (default)
#           sep=",",                # field delimiter
#           encoding="utf-8",       # text encoding
#           float_format="%.6g")    # optional numeric format

## Testing if gamma is being overfit on single sample 
# gamma_test = 39.1272
# rho_test_vec = np.zeros(1000)

# for i in range (1000):
#     rho_test_vec[i] = calc_rho(gamma_test)

# # plotting results
# plt.figure()
# plt.boxplot(rho_test_vec, vert=True)
# plt.ylabel('rho(gamma = 39.1272)')
# plt.title('Box Plot of Calculated rhos for 1000 Subsamples of Data')
# plt.show()

