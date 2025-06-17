### Python file for kernel flow implementation 
# Kade ALDRICH
# Internship 2025

import jax.numpy as jnp
import numpy as np
import jax
from jax import grad, make_jaxpr
import random

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
totalSampleSize = 200
X_1D = np.random.normal(loc = 0, scale = 1, size = totalSampleSize)

# creating output data 
# Y is quadratically related to X

a = 2.0
b = -1.0
c = 0.5

# Noise is N(0,1)
epsilon = np.random.normal(loc = 0.0, scale = 1, size = X_1D.shape)

# calculate Y vector
Y = a * X_1D**2 + b * X_1D + c + epsilon

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

    print(rho)
    return rho

calc_rho(0.1)