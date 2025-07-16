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
from sklearn.model_selection import train_test_split
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import mean_squared_error

random.seed(51)

#######################################################################
## Parameters of Interest
lam = 200 # ridge penalty (0 = no penalty, +inf = OLS)
#######################################################################

# function for getting sample indices 
def _sample_indices(key_size, proportion):
    #Return a 1-D NumPy array of unique sorted indices
    n = round(key_size * proportion)
    return np.sort(
        np.random.choice(key_size, size=n, replace=False)
    ).astype(np.int32)

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

#a = 2.0
#b = -1.0
#c = 0.5

# Noise is N(0,1)
#epsilon = np.random.normal(loc = 0.0, scale = 1, size = X_1D.shape)

# calculate Y vector
#Y = a * X_1D**2 + b * X_1D + c + epsilon


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
#######################################################################


## Function for calculating rho
# Needs to only take kernel parameters w as an input
# matrix multiplication if w a vector and multiplication if w a scalar
# make data globally accessible so that it doesn't have to be passed in
# define kernel matrices as a function of w

# vectors for holding condition values of Kf and Kc matrices for testing
#Kf_conds = np.zeros(shape = 1000)
#Kc_conds = np.zeros(shape = 1000)

def calc_rho(w):
    # sampling done within for loop
    
    # getting fine sample
    fineProportion = 1.0 # proportion of pool to use for fine sample (default is 1)
    #fineIndices = jnp.asarray(np.random.choice(range(totalSampleSize), size = round(totalSampleSize * fineProportion), replace = False))
    fineIndices   = _sample_indices(totalSampleSize, fineProportion)
    fineSample = X_1D[fineIndices]

    #print("fine indices", fineIndices)
    #print("fine length", len(fineIndices))

    # coarse sample
    coarseProportion = 0.5 # testing if coarse and samples the same, if rho = 0
    #coarseIndices = fineIndices
    #coarseSample = fineSample
    #coarseIndices = np.random.choice(fineIndices, size = round(len(fineIndices) * coarseProportion), replace = False)
    coarseIndices = _sample_indices(len(fineIndices), coarseProportion)
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

    # Checking condition of matrix (>>1 means numerical instability likely)
    #print("Kf condition = ", np.linalg.cond(Kf_reg))
    #print("Kc condition = ", np.linalg.cond(Kc_reg))
    #Kf_cond = np.linalg.cond(Kf_reg)
    #Kc_cond = np.linalg.cond(Kc_reg)

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
    #print("rho=", rho, ", w=", w)

    return rho


# function for calculating mean squared error on validation data
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
    print("MSE = ", mse)

    return mse
     


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
        #fineIndices = np.random.choice(range(totalSampleSize), size = round(totalSampleSize * fineProportion), replace = False)
        fineIndices   = _sample_indices(totalSampleSize, fineProportion)
        fineSample = X_1D[fineIndices]

        #print("fine indices", fineIndices)
        #print("fine length", len(fineIndices))

        # getting coarse sample
        #coarseIndices = fineIndices
        #coarseSample = fineSample
        #coarseIndices = np.random.choice(fineIndices, size = round(len(fineIndices) * coarseProportion), replace = False)
        coarseIndices = _sample_indices(len(fineIndices), coarseProportion)
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

#df = grad_desc_multi(max_iter = 20, w_init = 10)

#print(df)

## Function for handling gradient descent with Line Search for step size 
def grad_desc_ls(max_iter, w_init):

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
        print("rho= ", rho_trial)
        w_ts[i + 1] = w_trial
        rho_ts[i + 1] = rho_trial
        #print("rho", i+1, ": ", rho_ts[i+1])
    
    #print("rhos: ", rho_ts)
    #print("Final rho: ", rho_ts[max_iter]) # printing final rho value
    #print("Starting gamma: ", w_init)
    #print("Final gamma: ", w_ts[max_iter])
    # return data frame of parameters and rho values

    df = pd.DataFrame({
        'iteration': np.arange(max_iter + 1),
        'gamma': w_ts,
        'rho': rho_ts
        })

    return df

# Gradient descent function with fixed step size
def grad_desc_fs(max_iter, w_init):

    rho_ts = np.zeros(max_iter + 1) # initialize array holding rho values at each step
    w_ts = np.zeros(max_iter + 1) # initialize array to hold parameter values at each step (1d) 

    rho_ts[0] = calc_rho(w_init) # calculating initial rho value
    w_ts[0] = w_init # making first entry the initial parameters 

    for i in range(max_iter):
        calc_grad = jax.grad(calc_rho) # compute gradient of rho 
        grad = calc_grad(w_ts[i]) # calculate gradient at current parameters

        step = 0.2 # fixed step size 

        # calculate new parameter value using fixed step size 
        new_w = w_ts[i] - step * grad
        new_rho = calc_rho(new_w)

        rho_ts[i + 1] = new_rho
        w_ts[i + 1] = new_w

        print(i, ": rho= ", new_rho)
        #print("rho", i+1, ": ", rho_ts[i+1])
    
    #print("rhos: ", rho_ts)
    #print("Final rho: ", rho_ts[max_iter]) # printing final rho value
    #print("Starting gamma: ", w_init)
    #print("Final gamma: ", w_ts[max_iter])
    # return data frame of parameters and rho values

    df = pd.DataFrame({
        'iteration': np.arange(max_iter + 1),
        'gamma': w_ts,
        'rho': rho_ts
        })

    return df

    # testing if starting parameter changes final parameter and rho value much 

    # use for outputting only initial and final values of grad desc
    #row = pd.DataFrame({ # return single row data frame of starting gamma, final gamma, and final rho 
    #    'gamma_init': [w_init],
    #    'gamma_final': [w_ts[max_iter]],
    #    'rho_init': [rho_ts[0]],
    #    'rho_final': [rho_ts[max_iter]]
    #})

    #return row


# Gradient descent function with mse penalty and fixed stepsize
def grad_desc_mse_fs(max_iter, w_init):

    rho_ts = np.zeros(max_iter + 1) # initialize array holding rho values at each step
    psi_ts = np.zeros(max_iter + 1) # initialize array holding rho values at each step
    w_ts = np.zeros(max_iter + 1) # initialize array to hold parameter values at each step (1d) 

    # training KRR on train2 data

    # testing KRR on validation data


    rho_ts[0] = calc_rho(w_init) # calculating initial rho value
    w_ts[0] = w_init # making first entry the initial parameters 

    for i in range(max_iter):
        calc_grad = jax.grad(calc_rho) # compute gradient of rho 
        grad = calc_grad(w_ts[i]) # calculate gradient at current parameters

        step = 0.2 # fixed step size 

        # calculate new parameter value using fixed step size 
        new_w = w_ts[i] - step * grad
        new_rho = calc_rho(new_w)

        rho_ts[i + 1] = new_rho
        w_ts[i + 1] = new_w

        print(i, ": rho= ", new_rho)
        #print("rho", i+1, ": ", rho_ts[i+1])
    
    #print("rhos: ", rho_ts)
    #print("Final rho: ", rho_ts[max_iter]) # printing final rho value
    #print("Starting gamma: ", w_init)
    #print("Final gamma: ", w_ts[max_iter])
    # return data frame of parameters and rho values

    df = pd.DataFrame({
        'iteration': np.arange(max_iter + 1),
        'gamma': w_ts,
        'rho': rho_ts
        })

    return df


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



###############################
# Experimenting

#print(calc_rho(30))

# graphing rho vs lambda for fixed gamma
#rhos = np.zeros(shape = 1000)
#lambdas = np.arange(1000) + 1

#for i in range(len(rhos)):
#    lam = lambdas[i] 
#    rhos[i], Kf_conds[i], Kc_conds[i] = calc_rho(10)

#plt.plot(lambdas, rhos)
#plt.xlabel('lambda')        
#plt.ylabel('rho')           
#plt.title('rho vs lambda')
#plt.grid(True)             # optional: adds a grid
#plt.show()

# getting data and exporting as csv 
#lambda_tuning_for_gram_stability = pd.DataFrame({ # return single row data frame of starting gamma, final gamma, and final rho 
#    'lambda': lambdas,
#    'rho': rhos,
#    'Kf_cond': Kf_conds,
#    'Kc_conds': Kc_conds,
#})

#lambda_tuning_for_gram_stability.to_csv("lambda_tuning_for_gram_stability.csv", index=False)
################


################
# testing with starting gamma of 10 and lambda of 200
# using fixed step size 
#df = grad_desc_fs(max_iter=1000, w_init=10)
#df.to_csv("Kf-result-rbf-w10-fs2-iter1000.csv", index=False)
################

################
# testing a few different starting gamma 
#test_gamma = np.array([0.2, 0.5, 0.75, 1, 2, 5, 8, 12, 15, 20, 50, 100, 500, 1000])
#rho = np.zeros(len(test_gamma))

#for i in range(len(test_gamma)):
#    new_rho = calc_rho(test_gamma[i])
#    print("gamma: ", test_gamma[i], "rho: ", new_rho)
#    rho[i] = new_rho

#init_w_test_smooth = pd.DataFrame({
#    'gamma': test_gamma,
#    'rho': rho,
#})
#init_w_test_smooth.to_csv("init-w-test-smooth.csv", index=False)
################

################
# testing many starting gamma
# multiple trials to account for random point selection 
# to be used in graph in latex later 
# generate gamma values from 0.2 to 50.0 inclusive, step 0.1

#gammas = np.arange(0.0, 50.0 + 1e-8, 0.1)
#gammas = np.arange(0.01, 5.0 + 1e-8, 0.01) # more fine array of test gammas for bump function 

# prepare a results DataFrame
#results = pd.DataFrame({'gamma': gammas})

## tests to run:
# init-w-test-smooth-noerror-lap.csv 
# init-w-test-bump-noerror-lap.csv 
# init-w-small-test-bump-noerror-lap.csv
# init-w-test-hfreq-noerror-lap.csv
# init-w-small-test-hfreq-noerror-lap.csv
# init-w-test-rough-noerror-lap.csv
# init-w-small-test-rough-noerror-lap.csv

## trials done:
# init-w-test-smooth-noerror-rbf.csv 
# init-w-test-bump-noerror-rbf.csv 
# init-w-small-test-bump-noerror-rbf.csv
# init-w-test-hfreq-noerror-rbf.csv
# init-w-small-test-hfreq-noerror-rbf.csv
# init-w-test-rough-noerror-rbf.csv
# init-w-small-test-rough-noerror-rbf.csv


# run 5 trials
#for trial in range(1, 6):
#    rho_vals = []
#    for g in gammas:
#        rho_vals.append(calc_rho(g))
#    results[f'rho_trial_{trial}'] = rho_vals
#    print(f"Completed trial {trial}")

# export to CSV
#results.to_csv("init-w-small-test-rough-noerror-rbf.csv", index=False)
################