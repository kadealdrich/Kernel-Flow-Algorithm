# Kade Aldrich
# Internship 2025
# Kernel Flow Algorithm

### Script for checking the gradient calcualtion of the penalized rho criterion using Laplacian kernel 

# Kade Aldrich
# Internship 2025


### Gradient descent for tuning Laplacian kernel parameter given 1 dimensional data
# Criterion of interest is rho defined by Owhadi penalized via prediction MSE 
# Using kernel ridge regression estimator




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


###############################################################################################

#                               INITIAL HYPERPARAMETERS                                       #

###############################################################################################

max_iter = 1000
step_init = 0.1 # inital step size
decay_rate = 0.5 # step size decay factor
decay_threshold = round((max_iter - 1) / 4) # number of iterations before decay (defaults to decaying 3 times)
lam_fixed = 10
lam_init = 10 # in case of updating lambda
sigma_init = 0.25
desc_parameters_init = jnp.array([lam_init, sigma_init], dtype = jnp.float32)
MSE_WEIGHT = 0.5

###############################################################################################




###############################################################################################

#                                           DATA                                              #

###############################################################################################


# Options are:
# 'y_smooth' |  Smooth cosine function
# 'y_bump'   |  Function with abrupt bump in the middle 
# 'y_hfreq'  |  High frequency periodic function 
# 'y_rough'  |  Absolute value function 



df_experiment = pd.read_csv("test-functions.csv")
y = df_experiment['y_smooth'] # using smooth cosine y 
#y = df_experiment['y_hfreq'].to_numpy(dtype=np.float32) # high frequency, difficult y 
x = df_experiment['x'].to_numpy(dtype=np.float32)



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

###############################################################################################






###############################################################################################

#                                         ANALYSIS                                            #

###############################################################################################


# function for getting new coarse and fine samples for rho criterion
def get_coarse_indices(key, n_train, coarse_prop = 0.5):
    # ASSUMES FINE SAMPLE IS ALL TRAINING DATA 
    # gets a random coarse sample equal to a random selection of round(0.5 * len(fine_sample)) indices
    key, subkey = jax.random.split(key)
    n_coarse = jnp.maximum(1, jnp.round(coarse_prop*n_train).astype(jnp.int32))
    perm = jax.random.permutation(subkey, n_train) # gets random permutation 
    coarse_indices = perm[:n_coarse]
    return coarse_indices, key # anytime this is called have to update key value



## Kernel ridge regression using Laplacian kernel with L1 norm
def KRR_lapL1(sigma, lam, x_train, y_train, x_test):

    # converting to numpy arrays
    x_train = jnp.asarray(x_train)
    y_train = jnp.asarray(y_train)
    x_test  = jnp.asarray(x_test)

    # calculating kernel gram matrix
    train_diffs = x_train[:, None] - x_train[None, :]
    train_absdf = jnp.abs(train_diffs)  # square diff matrix (nf, nf)
    K_train = jnp.exp(-train_absdf * sigma ** -1) # kernel gram matrix of training data
    K_train_reg = K_train + lam * jnp.eye(len(x_train), dtype = K_train.dtype)  # regularized kernel gram matrix 

    # solving for the weights 
    weights = solve(K_train_reg, y_train, assume_a = 'pos', lower = True) # positive definite and symmetric
    # set argurments in solve for optimized solve: 
    #   assume_a = 'pos', lower=True
      
    # sym_pos: sym mean symmetric, pos means positive definite --> jax uses Cholesky decomp which is fast and should help with numerical issues

    # predicting unseen validation data
    train_test_diffs = x_test[:, None] - x_train[None, :] # difference matrix between test and train matrices
    abs_dists = jnp.abs(train_test_diffs)
    K_test_train = jnp.exp(-abs_dists * sigma ** -1) # kernel gram matrix of the train2 and validation x's 

    y_pred = K_test_train @ weights # KRR prediction of y validation

    return y_pred 



# function for calculating mean squared error of KRR prediction on validation data
# uses global variables for test y values and predicted y values so that jax can be used for gradient calculation in the future
# all the same as the KRR function except for it returns the prediction mean squared error
def calc_mse_lapL1(params, x_tr, x_val, y_tr, y_val):
    lam, sigma = params # two hyperparameters

    # convert to jax arrays
    x_tr, x_val, y_tr, y_val = map(jnp.asarray, (x_tr, x_val, y_tr, y_val))
    
    # calculating kernel gram matrix
    train_diffs = x_tr[:, None] - x_tr[None, :]
    train_absdf = jnp.abs(train_diffs)  # square diff matrix (nf, nf)
    K_train = jnp.exp(-train_absdf * sigma ** -1) # kernel gram matrix of training data
    K_train_reg = K_train + lam * jnp.eye(len(x_tr))  # regularized kernel gram matrix 

    # solving for the weights 
    weights = solve(K_train_reg, y_tr, assume_a = 'pos', lower=True) # positive definite and symmetric  
    # sym_pos: sym mean symmetric, pos means positive definite --> jax uses Cholesky decomp which is fast and should help with numerical issues

    # predicting unseen validation data
    train_test_diffs = x_val[:, None] - x_tr[None, :] # difference matrix between test and train matrices
    abs_dists = train_test_diffs**2
    K_test_train = jnp.exp(-abs_dists * sigma ** -1) # kernel gram matrix of the train2 and validation x's 

    y_pred = K_test_train @ weights # KRR prediction of y validation

    mse = jnp.mean((y_val - y_pred)**2) # calculate mean squared error of kernel ridge regression prediction
    
    return mse



# function for calculating Normlized and bounded mse 
def calc_nmse_lapL1(params, x_tr, x_val, y_tr, y_val):
    lam, sigma = params # two hyperparameters

    # convert to jax arrays
    x_tr, x_val, y_tr, y_val = map(jnp.asarray, (x_tr, x_val, y_tr, y_val))
    
    # calculating kernel gram matrix
    train_diffs = x_tr[:, None] - x_tr[None, :]
    train_absdf = jnp.abs(train_diffs) # square diff matrix (nf, nf)
    K_train = jnp.exp(-train_absdf * sigma ** -1) # kernel gram matrix of training data
    K_train_reg = K_train + lam * jnp.eye(len(x_tr))  # regularized kernel gram matrix 

    # solving for the weights 
    weights = solve(K_train_reg, y_tr, assume_a = 'pos', lower=True) # positive definite and symmetric  
    # sym_pos: sym mean symmetric, pos means positive definite --> jax uses Cholesky decomp which is fast and should help with numerical issues

    # predicting unseen validation data
    train_test_diffs = x_val[:, None] - x_tr[None, :] # difference matrix between test and train matrices
    abs_dists = jnp.abs(train_test_diffs)
    K_test_train = jnp.exp(-abs_dists * sigma ** -1) # kernel gram matrix of the train2 and validation x's 

    y_pred = K_test_train @ weights # KRR prediction of y validation

    mse = jnp.mean((y_val - y_pred)**2) # calculate mean squared error of kernel ridge regression prediction
    
    var = jnp.var(y_tr) + 1e-8 # adds epsilon because this goes in denominator
    nmse = mse / var
    bounded_nmse = 1.0 - jnp.exp(-nmse)

    return bounded_nmse



# function for calculating rho criterion 
def calc_rho_lapL1(params, x_fine, x_coarse, y_fine, y_coarse):
    lam, sigma = params # two hyperparameters

    # converting to jax numpy arrays 
    x_f, x_c, y_f, y_c = map(jnp.asarray, (x_fine, x_coarse, y_fine, y_coarse))

    ## constructing kernel Gram matrices

    # calculating difference matrices
    diffs_f = x_f[:, None] - x_f[None:,]
    diffs_c = x_c[:, None] - x_c[None:,]

    # using L1-norm
    absdf = jnp.abs(diffs_f)
    absdc = jnp.abs(diffs_c)

    # constructing Laplacian kernel Gram matrices
    Kf = jnp.exp(-absdf * sigma ** -1.0) # fine sample kernel Gram matrix
    Kc = jnp.exp(-absdc * sigma ** -1.0) # coarse sample kernel Gram matrix

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



# function for getting rho penalized by mse using Laplacian kernel
def calc_pen_crit_lapL1(params, x_fine, x_coarse, y_fine, y_coarse, x_tr, x_val, y_tr, y_val, mse_weight = 0.5):
    rho = calc_rho_lapL1(params, x_fine, x_coarse, y_fine, y_coarse)
    nmse = calc_nmse_lapL1(params, x_tr, x_val, y_tr, y_val)
    return rho + mse_weight*nmse
    


# calculates rho criterion with mse penalty given sigma from the Laplacian kernel
def pen_crit_sigma_lapL1(sigma, x_fine, x_coarse, y_fine, y_coarse, x_tr, x_val, y_tr, y_val, mse_weight = 0.5):
    params = (lam_fixed, sigma)
    return calc_pen_crit_lapL1(params, x_fine, x_coarse, y_fine, y_coarse, x_tr, x_val, y_tr, y_val, mse_weight)



# jit wrappers for functions 
pen_crit_sigma_jit = jit(pen_crit_sigma_lapL1)
value_and_grad = jit(jax.grad(pen_crit_sigma_jit, argnums=0))
KRR_lapL1_jit = jit(KRR_lapL1)



## USING JAX TO GET CRITERRION VALUES AND GRADIENTS AT GIVEN PARAMETER VALUES
# function for getting criterion value and gradient at given parameter values
@jax.jit 
def grad_rho_lapL1(params, x_fine, x_coarse, y_fine, y_coarse):
    rho_fun = lambda p: calc_rho_lapL1(p, x_fine, x_coarse, y_fine, y_coarse)
    rho_val, grads = jax.value_and_grad(rho_fun)(params)
    dlam, dsig = grads  # same structure as params=(lambda, sigma)
    return rho_val, dlam, dsig

@jax.jit
def grad_nmse_lapL1(params, x_tr, x_val, y_tr, y_val):
    nmse_fun = lambda p: calc_nmse_lapL1(p, x_tr, x_val, y_tr, y_val)
    nmse_val, grads = jax.value_and_grad(nmse_fun)(params)
    dlam, dsig = grads  # same structure as params=(lambda, sigma)
    return nmse_val, dlam, dsig   



## Calculating gradient manually 
def manual_grad_rho_lapL1(params, x_fine, x_coarse, y_fine, y_coarse):
    """
    Compute rho(sigma,λ) = 1 - F_c / F_f and its manual gradients for the L1 Laplacian kernel.

    F(X,y) := yᵀ K(X) (K(X)+λI)^{-2} y
    ∂F/∂σ  = yᵀ[ dK A^{-2} - K A^{-1} dK A^{-2} - K A^{-2} dK A^{-1} ] y
    ∂F/∂λ  = -2 yᵀ K A^{-3} y
    ⇒ ∂ρ/∂θ = -[(∂θ F_c) F_f - F_c (∂θ F_f)] / F_f²,  θ∈{sigma,λ}.

    Args
    ----
    params   : tuple (lam, sigma)
    x_fine   : (n_f,) array of inputs for the fine set X_f
    x_coarse : (n_c,) array of inputs for the coarse set X_c ⊂ X_f
    y_fine   : (n_f,) array of targets for X_f
    y_coarse : (n_c,) array of targets for X_c

    Returns
    -------
    rho        : scalar rho(sigma,λ)
    d_rho_dlam : scalar ∂ρ/∂λ
    d_rho_dsig : scalar ∂ρ/∂σ
    """
    lam, sigma = params

    # ---------- helpers ----------
    def K_and_dK_sigma_l1(x, sigma):
        """
        Laplacian (L1) kernel and its sigma-derivative.
        K_ij = exp(-|x_i - x_j| / sigma),
        dK/dsigma = (|x_i - x_j| / sigma^2) * K_ij.
        """
        x = jnp.asarray(x).ravel()
        D = jnp.abs(x[:, None] - x[None, :])
        K = jnp.exp(-D / sigma)
        dK = (D / (sigma**2)) * K
        return K, dK

    def F_and_grads(K, dK, y, lam):
        """
        Given K, dK=dK/dsigma and y, compute:
          F = yᵀ K A^{-2} y,
          dF/dsigma and dF/δλ, with A = K + λ I.
        All A^{-k} are done via repeated linear solves for stability.
        """
        y = jnp.asarray(y).ravel()
        n = K.shape[0]
        I = jnp.eye(n, dtype=K.dtype)
        A = K + lam * I

        # A^{-1} y and A^{-2} y
        v = jnp.linalg.solve(A, y)   # A^{-1} y
        w = jnp.linalg.solve(A, v)   # A^{-2} y
        Ky = K @ y

        # F = yᵀ K A^{-2} y
        F = y @ (K @ w)

        # dF/dσ = yᵀ[dK A^{-2}]y - yᵀ[K A^{-1} dK A^{-2}]y - yᵀ[K A^{-2} dK A^{-1}]y
        term1 = y @ (dK @ w)
        s = jnp.linalg.solve(A, dK @ w)   # A^{-1} dK A^{-2} y
        term2 = Ky @ s
        t = jnp.linalg.solve(A, dK @ v)   # A^{-1} dK A^{-1} y
        u = jnp.linalg.solve(A, t)        # A^{-2} dK A^{-1} y
        term3 = Ky @ u
        dF_dsigma = term1 - term2 - term3

        # dF/dλ = -2 yᵀ K A^{-3} y
        u3 = jnp.linalg.solve(A, w)       # A^{-3} y
        dF_dlam = -2.0 * (Ky @ u3)

        return F, dF_dsigma, dF_dlam

    # ---------- fine set ----------
    Kf, dKf = K_and_dK_sigma_l1(x_fine, sigma)
    Ff, dFf_dsig, dFf_dlam = F_and_grads(Kf, dKf, y_fine, lam)

    # ---------- coarse set ----------
    Kc, dKc = K_and_dK_sigma_l1(x_coarse, sigma)
    Fc, dFc_dsig, dFc_dlam = F_and_grads(Kc, dKc, y_coarse, lam)

    # ---------- rho and its gradients ----------
    rho = 1.0 - Fc / Ff
    denom = Ff ** 2
    d_rho_dsig = - (dFc_dsig * Ff - Fc * dFf_dsig) / denom
    d_rho_dlam = - (dFc_dlam * Ff - Fc * dFf_dlam) / denom

    return rho, d_rho_dlam, d_rho_dsig



# function for manually calculating gradient of NMSE 
def manual_grad_nmse_lapL1(params, x_tr, x_val, y_tr, y_val, eps=1e-8):
    """
    Compute nmse(σ,λ) = mse(σ,λ) / (Var(y_tr) + eps) and its manual gradients.

    Kernel (ℓ1 Laplacian):
      K_ij = exp( -|x_i - z_j| / σ ),  dK/dσ = (|x_i - z_j| / σ^2) * K_ij.

    Validation prediction with KRR:
      A = K_tt + λ I,  α = A^{-1} y_tr,  ŷ_val = K_vt α,
      r = y_val - ŷ_val,  mse = (1/n_val) * rᵀ r,
      nmse = mse / var,  var = Var(y_tr) + eps (constant wrt σ,λ).

    Manual gradients (Fréchet/matrix calculus):
      d(mse)/dσ = (-2/n_val) [ rᵀ (dK_vt α) - rᵀ K_vt A^{-1} (dK_tt α) ]
      d(mse)/dλ = ( 2/n_val) [ rᵀ K_vt A^{-1} α ]
      d(nmse)/dθ = (1/var) * d(mse)/dθ,  θ∈{σ,λ}.

    Args
    ----
    params : (lam, sigma)
    x_tr   : (n_tr,) training inputs
    x_val  : (n_val,) validation inputs
    y_tr   : (n_tr,) training targets
    y_val  : (n_val,) validation targets
    eps    : small positive constant for numerical stability in var

    Returns
    -------
    nmse        : scalar
    d_nmse_dlam : scalar ∂nmse/∂λ
    d_nmse_dsig : scalar ∂nmse/∂σ
    """
    lam, sigma = params

    # ---- helpers: Laplacian kernel and its σ-derivative (1D ℓ1) ----
    def K_and_dK_sigma_l1(x, z, sigma):
        x = jnp.asarray(x).ravel()
        z = jnp.asarray(z).ravel()
        D = jnp.abs(x[:, None] - z[None, :])       # pairwise ℓ1 distances
        K = jnp.exp(-D / sigma)                     # kernel matrix
        dK = (D / (sigma**2)) * K                   # ∂K/∂σ
        return K, dK

    # ---- build Gram matrices ----
    Ktt, dKtt = K_and_dK_sigma_l1(x_tr,  x_tr,  sigma)  # train–train
    Kvt, dKvt = K_and_dK_sigma_l1(x_val, x_tr,  sigma)  # val–train

    # ---- ridge solve on training set ----
    n_tr = Ktt.shape[0]
    A = Ktt + lam * jnp.eye(n_tr, dtype=Ktt.dtype)
    alpha = jnp.linalg.solve(A, y_tr)                    # A^{-1} y_tr

    # ---- validation residual and MSE ----
    yhat_val = Kvt @ alpha
    r = y_val - yhat_val
    n_val = r.shape[0]
    mse = (r @ r) / n_val

    var = jnp.var(y_tr) + eps          # constant wrt σ and λ
    nmse = mse / var

    # ---- manual gradients of mse ----
    # term1: rᵀ (dK_vt α)
    term1 = r @ (dKvt @ alpha)

    # term2: rᵀ K_vt A^{-1} (dK_tt α)
    s = jnp.linalg.solve(A, dKtt @ alpha)               # A^{-1}(dK_tt α)
    term2 = r @ (Kvt @ s)

    dmse_dsigma = (-2.0 / n_val) * (term1 - term2)
    dmse_dlam   = ( 2.0 / n_val) * (r @ (Kvt @ jnp.linalg.solve(A, alpha)))  # A^{-1}α

    # ---- chain rule to nmse ----
    d_nmse_dsig = dmse_dsigma / var
    d_nmse_dlam = dmse_dlam   / var

    return nmse, d_nmse_dlam, d_nmse_dsig

# function for getting bounded NMSE 
def manual_grad_phi_from_nmse(params, x_tr, x_val, y_tr, y_val, eps=1e-8):
    """
    Compute φ = 1 - exp(-nmse) and its grads from the manual nmse function.
    Returns: (phi, dphi/dlambda, dphi/dsigma).
    """
    nmse, d_nmse_dlam, d_nmse_dsig = manual_grad_nmse_lapL1(
        params, x_tr, x_val, y_tr, y_val, eps=eps
    )
    fac = jnp.exp(-nmse)           # chain factor
    phi = 1.0 - fac
    dphi_dlam = fac * d_nmse_dlam
    dphi_dsig = fac * d_nmse_dsig
    return phi, dphi_dlam, dphi_dsig



## Gaussian kernel

# KRR
def KRR_rbf(w, lam, x_train, y_train, x_test):

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

# NMSE
# function for calculating Normlized and bounded mse 
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

# rho
def calc_rho_rbf(params, x_fine, x_coarse, y_fine, y_coarse):
    lam, gamma = params # two hyperparameters

    # converting to jax numpy arrays 
    x_f, x_c, y_f, y_c = map(jnp.asarray, (x_fine, x_coarse, y_fine, y_coarse))

    ## constructing kernel Gram matrices

    # calculating difference matrices
    diffs_f = x_f[:, None] - x_f[None:,]
    diffs_c = x_c[:, None] - x_c[None:,]

    # using L2-norm
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


## USING JAX TO GET CRITERRION VALUES AND GRADIENTS AT GIVEN PARAMETER VALUES
# function for getting criterion value and gradient at given parameter values
@jax.jit 
def grad_rho_rbf(params, x_fine, x_coarse, y_fine, y_coarse):
    rho_fun = lambda p: calc_rho_rbf(p, x_fine, x_coarse, y_fine, y_coarse)
    rho_val, grads = jax.value_and_grad(rho_fun)(params)
    dlam, dsig = grads  # same structure as params=(lambda, sigma)
    return rho_val, dlam, dsig

@jax.jit
def grad_nmse_rbf(params, x_tr, x_val, y_tr, y_val):
    nmse_fun = lambda p: calc_nmse_rbf(p, x_tr, x_val, y_tr, y_val)
    nmse_val, grads = jax.value_and_grad(nmse_fun)(params)
    dlam, dsig = grads  # same structure as params=(lambda, sigma)
    return nmse_val, dlam, dsig   


# manual closed form 
def manual_grad_rho_rbf(params, x_fine, x_coarse, y_fine, y_coarse):
    """
    Manual gradient of ρ(γ,λ) for the RBF kernel k_γ(x,z)=exp(-γ ||x-z||^2).

    Definitions:
      F(X,y) := yᵀ K(X) (K(X)+λI)^{-2} y,   A(X)=K(X)+λI
      ρ      := 1 - F_c / F_f
      ∂F/∂γ  = yᵀ[ dK A^{-2} - K A^{-1} dK A^{-2} - K A^{-2} dK A^{-1} ] y
      ∂F/∂λ  = -2 yᵀ K A^{-3} y
      ⇒ ∂ρ/∂θ = -[(∂θF_c) F_f - F_c (∂θF_f)] / F_f²,  θ∈{γ,λ}.

    Args
    ----
    params   : (lam, gamma)
    x_fine   : (n_f,) inputs for X_f
    x_coarse : (n_c,) inputs for X_c (subset of X_f)
    y_fine   : (n_f,) targets for X_f
    y_coarse : (n_c,) targets for X_c

    Returns
    -------
    rho        : scalar ρ(γ,λ)
    d_rho_dlam : scalar ∂ρ/∂λ
    d_rho_dgam : scalar ∂ρ/∂γ
    """
    lam, gamma = params

    # --- helpers -------------------------------------------------------------
    def rbf_K_and_dgamma(x):
        """
        Build RBF Gram K and its derivative wrt γ.
        For 1D inputs: S_ij = (x_i - x_j)^2, K = exp(-γ S), dK/dγ = -S ⊙ K.
        """
        x = jnp.asarray(x).ravel()
        diffs = x[:, None] - x[None, :]
        S = diffs * diffs              # squared L2 distances
        K = jnp.exp(-gamma * S)
        dK = -S * K                    # ∂K/∂γ
        return K, dK

    def F_and_grads(K, dK, y, lam):
        """
        Compute F, ∂F/∂γ, ∂F/∂λ with solves (no explicit inverses).
        """
        y = jnp.asarray(y).ravel()
        n = K.shape[0]
        A = K + lam * jnp.eye(n, dtype=K.dtype)

        # A^{-1}y, A^{-2}y
        v = jnp.linalg.solve(A, y)          # A^{-1} y
        w = jnp.linalg.solve(A, v)          # A^{-2} y
        Ky = K @ y

        # F = yᵀ K A^{-2} y
        F = y @ (K @ w)

        # ∂F/∂γ
        term1 = y @ (dK @ w)
        s = jnp.linalg.solve(A, dK @ w)     # A^{-1} dK A^{-2} y
        term2 = Ky @ s
        t = jnp.linalg.solve(A, dK @ v)     # A^{-1} dK A^{-1} y
        u = jnp.linalg.solve(A, t)          # A^{-2} dK A^{-1} y
        term3 = Ky @ u
        dF_dgamma = term1 - term2 - term3

        # ∂F/∂λ = -2 yᵀ K A^{-3} y
        u3 = jnp.linalg.solve(A, w)         # A^{-3} y
        dF_dlam = -2.0 * (Ky @ u3)

        return F, dF_dgamma, dF_dlam

    # --- fine set ------------------------------------------------------------
    Kf, dKf = rbf_K_and_dgamma(x_fine)
    Ff, dFf_dgam, dFf_dlam = F_and_grads(Kf, dKf, y_fine, lam)

    # --- coarse set ----------------------------------------------------------
    Kc, dKc = rbf_K_and_dgamma(x_coarse)
    Fc, dFc_dgam, dFc_dlam = F_and_grads(Kc, dKc, y_coarse, lam)

    # --- rho and its gradients ----------------------------------------------
    rho = 1.0 - Fc / Ff
    denom = Ff ** 2
    d_rho_dgam = - (dFc_dgam * Ff - Fc * dFf_dgam) / denom
    d_rho_dlam = - (dFc_dlam * Ff - Fc * dFf_dlam) / denom

    return rho, d_rho_dlam, d_rho_dgam


# manually getting bounded nmse
def manual_grad_bounded_nmse_rbf(params, x_tr, x_val, y_tr, y_val, eps=1e-8):
    """
    Bounded-NMSE φ(λ,γ) = 1 - exp(-nmse),  with
      nmse = mse / (Var(y_tr) + eps),
      mse  = (1/n_val) * || y_val - K_vt * α ||^2,
      α    = (K_tt + λI)^{-1} y_tr,
      RBF: K_ij = exp( -γ * (x_i - z_j)^2 ),   ∂K/∂γ = -(x_i - z_j)^2 ⊙ K.

    Returns:
      phi, dphi/dlambda, dphi/dgamma
    """
    lam, gamma = params

    # ---------- 1) Build Gram matrices and their γ-derivatives ----------
    x_tr = jnp.asarray(x_tr).ravel()
    x_val = jnp.asarray(x_val).ravel()
    y_tr = jnp.asarray(y_tr).ravel()
    y_val = jnp.asarray(y_val).ravel()

    # Train–train
    dif_tt = x_tr[:, None] - x_tr[None, :]
    S_tt = dif_tt * dif_tt                      # squared L2 distances
    K_tt = jnp.exp(-gamma * S_tt)
    dK_tt = -S_tt * K_tt                        # ∂K_tt/∂γ

    # Val–train
    dif_vt = x_val[:, None] - x_tr[None, :]
    S_vt = dif_vt * dif_vt
    K_vt = jnp.exp(-gamma * S_vt)
    dK_vt = -S_vt * K_vt                        # ∂K_vt/∂γ

    # ---------- 2) KRR solve on training set ----------
    n_tr = K_tt.shape[0]
    A = K_tt + lam * jnp.eye(n_tr, dtype=K_tt.dtype)
    alpha = jnp.linalg.solve(A, y_tr)           # A^{-1} y_tr

    # ---------- 3) Validation prediction and (bounded) NMSE ----------
    y_hat = K_vt @ alpha
    r = y_val - y_hat
    n_val = r.shape[0]
    mse = (r @ r) / n_val

    var = jnp.var(y_tr) + eps                   # constant w.r.t. λ, γ
    nmse = mse / var
    phi = 1.0 - jnp.exp(-nmse)                  # bounded in [0,1)

    # ---------- 4) Manual grads of MSE ----------
    # d(mse)/dγ = (-2/n) [ rᵀ (dK_vt α) - rᵀ K_vt A^{-1} (dK_tt α) ]
    term1 = r @ (dK_vt @ alpha)
    s = jnp.linalg.solve(A, dK_tt @ alpha)      # A^{-1}(dK_tt α)
    term2 = r @ (K_vt @ s)
    dmse_dgamma = (-2.0 / n_val) * (term1 - term2)

    # d(mse)/dλ = (2/n) rᵀ K_vt A^{-1} α
    dmse_dlambda = (2.0 / n_val) * (r @ (K_vt @ jnp.linalg.solve(A, alpha)))

    # ---------- 5) Chain rule to bounded NMSE φ ----------
    # nmse = mse / var,  φ = 1 - exp(-nmse)
    # ⇒ dφ/dθ = exp(-nmse) * d(nmse)/dθ = exp(-nmse)/var * d(mse)/dθ
    chain = jnp.exp(-nmse) / var
    dphi_dgamma = chain * dmse_dgamma
    dphi_dlambda = chain * dmse_dlambda

    return phi, dphi_dlambda, dphi_dgamma

###### TESTING ######

x_tr, x_val, y_tr, y_val = get_validation_split()

x_tr = jnp.asarray(x_tr, dtype = jnp.float32)
y_tr = jnp.asarray(y_tr, dtype = jnp.float32)
x_val = jnp.asarray(x_val, dtype = jnp.float32)
y_val = jnp.asarray(y_val, dtype = jnp.float32) 
    
# x_fine is the same as x_tr
# it used to be different so this is a quick fix to make sure everything else still works
x_fine = x_tr
y_fine = y_tr

# selecting coarse subset from reduced training subset
coarse_indices, key = get_coarse_indices(key=key_init, n_train = len(x_fine))

# get the coarse subset for calculating rho 
x_coarse = x_tr[coarse_indices]
y_coarse = y_tr[coarse_indices]


lam_test = 10
sigma_test = 2
params = jnp.asarray([lam_test, sigma_test], dtype=jnp.float32)


print("LAPLACIAN KERNEL")
print("")
print("JAX RESULTS:")
print("")
print(f"lambda = {lam_test}, sigma = {sigma_test}")

rho, rho_dlam, rho_dsig = grad_rho_lapL1(params, x_fine, x_coarse, y_fine, y_coarse)
print(f"rho = {rho}, drho/dlambda = {rho_dlam}, drho/dsigma = {rho_dsig}")

phi, phi_dlam, phi_dsig = grad_nmse_lapL1(params, x_tr = x_tr, x_val = x_val, y_tr = y_tr, y_val = y_val)
print(f"phi = {phi}, drho/dlambda = {phi_dlam}, drho/dsigma = {phi_dsig}")

print("")
print("")
print("")

print("MANUAL RESULTS:")

rho, d_rho_dlam, d_rho_dsig = manual_grad_rho_lapL1(params=params, x_fine = x_fine, x_coarse = x_coarse, y_fine = y_fine, y_coarse = y_coarse)

print(f"rho = {rho}, drho/dlambda = {d_rho_dlam}, drho/dsigma = {d_rho_dsig}")

#manual_grad_nmse_lapL1(params, x_tr, x_val, y_tr, y_val, eps=1e-8)
phi, d_phi_dlam, d_phi_dsig = manual_grad_phi_from_nmse(params=params, x_tr = x_tr, x_val = x_val, y_tr = y_tr, y_val = y_val)
print(f"phi = {phi}, drho/dlambda = {d_phi_dlam}, drho/dsigma = {d_phi_dsig}")


print("")
print("")
print("")

print("GAUSSIAN KERNEL")

print("")

print("JAX RESULTS:")
print("")
print(f"lambda = {lam_test}, sigma = {sigma_test}")

rho_rbf_jax, rho_dlam_rbf_jax, rho_dsig_rbf_jax = grad_rho_rbf(params, x_fine, x_coarse, y_fine, y_coarse)
print(f"rho = {rho_rbf_jax}, drho/dlambda = {rho_dlam_rbf_jax}, drho/dsigma = {rho_dsig_rbf_jax}")

phi_rbf_jax, phi_dlam_rbf_jax, phi_dsig_rbf_jax = grad_nmse_rbf(params, x_tr = x_tr, x_val = x_val, y_tr = y_tr, y_val = y_val)
print(f"phi = {phi_rbf_jax}, drho/dlambda = {phi_dlam_rbf_jax}, drho/dsigma = {phi_dsig_rbf_jax}")

print("")
print("")
print("")

print("MANUAL RESULTS:")
print("")

rho_rbf_manual, d_rho_dlam_rbf_manual, d_rho_dsig_rbf_manual = manual_grad_rho_rbf(params=params, x_fine = x_fine, x_coarse = x_coarse, y_fine = y_fine, y_coarse = y_coarse)
print(f"rho = {rho_rbf_manual}, drho/dlambda = {d_rho_dlam_rbf_manual}, drho/dsigma = {d_rho_dsig_rbf_manual}")

phi_rbf_manual, d_phi_dlam_rbf_manual, d_phi_dsig_rbf_manual = manual_grad_bounded_nmse_rbf(params=params, x_tr = x_tr, x_val = x_val, y_tr = y_tr, y_val = y_val)
print(f"phi = {phi_rbf_manual}, drho/dlambda = {d_phi_dlam_rbf_manual}, drho/dsigma = {d_phi_dsig_rbf_manual}")
