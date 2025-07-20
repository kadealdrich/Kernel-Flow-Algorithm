### File for getting stats for ANOVA for algorithm runtimes 
# Kade Aldrich
# Internship 2025


import pandas as pd
import numpy as np 
from scipy.stats import f, ttest_rel, kstest
import matplotlib.pyplot as plt


# load csv's of runtimes
manual  = pd.read_csv("manual_krr_runtimes_noJIT_genmatrix.csv", header=None).iloc[:,0].values
sklearn = pd.read_csv("sklearn_krr_runtimes_noJIT_genmatrix.csv", header=None).iloc[:,0].values

# calculate basic stats:
n1, n2 = len(manual), len(sklearn)
mean1, mean2 = manual.mean(), sklearn.mean()
overall = np.concatenate([manual, sklearn]).mean()

# calculate sums of squares:
SS_between = n1*(mean1-overall)**2 + n2*(mean2-overall)**2
SS_within  = ((manual-mean1)**2).sum() + ((sklearn-mean2)**2).sum()

df_between = 1
df_within  = n1+n2-2

MS_between = SS_between/df_between
MS_within  = SS_within/df_within

F_val = MS_between/MS_within
p_val = f.sf(F_val, df_between, df_within)

# print summary:
print("Source         SS          DF      MS          F        Pr(>F)")
print(f"Between  {SS_between:.4e}   {df_between}   {MS_between:.4e}   {F_val:.4f}   {p_val:.3e}")
print(f"Within   {SS_within:.4e}   {df_within}   {MS_within:.4e}")
print(f"Total    {(SS_between+SS_within):.4e}   {df_between+df_within}")


## loading in the different pairs of run times

# with jit
manual_jit  = pd.read_csv("manual_krr_runtimes.csv", header=None).iloc[:,0].values
sklearn_jit = pd.read_csv("sklearn_krr_runtimes.csv", header=None).iloc[:,0].values

# without jit
manual_nojit  = pd.read_csv("manual_krr_runtimes_noJIT.csv", header=None).iloc[:,0].values
sklearn_nojit = pd.read_csv("sklearn_krr_runtimes_noJIT.csv", header=None).iloc[:,0].values

# without jit or matrix assumptions
manual_nojit_genmatrix  = pd.read_csv("manual_krr_runtimes_noJIT_genmatrix.csv", header=None).iloc[:,0].values
sklearn_nojit_genmatrix = pd.read_csv("sklearn_krr_runtimes_noJIT_genmatrix.csv", header=None).iloc[:,0].values


## calculating the statistics

# checking if differences are normally distributed 
# necessary for running paired ttest
diffs_jit = manual_jit - sklearn_jit
diffs_nojit = manual_nojit - sklearn_nojit
diffs_nojit_genmatrix = manual_nojit_genmatrix - sklearn_nojit_genmatrix

mu_jit = np.mean(diffs_jit)
mu_nojit = np.mean(diffs_nojit)
mu_nojit_genmatrix = np.mean(diffs_nojit_genmatrix)

sig_jit = np.std(diffs_jit, ddof = 1) # sample standard deviation
sig_nojit = np.std(diffs_nojit, ddof = 1)
sig_nojit_genmatrix = np.std(diffs_nojit_genmatrix, ddof = 1)

# calculating ks statistics for each of the differences 
ks_stat_jit, p_value_ks_jit = kstest(diffs_jit, 'norm', args=(mu_jit, sig_jit))
ks_stat_nojit, p_value_ks_nojit = kstest(diffs_nojit, 'norm', args=(mu_nojit, sig_nojit))
ks_stat_nojit_genmatrix, p_value_ks_nojit_genmatrix = kstest(diffs_nojit_genmatrix, 'norm', args=(mu_nojit_genmatrix, sig_nojit_genmatrix))

print(f"KS statistic results (JIT):  |  KS = {ks_stat_jit:.3f}  |  p = {p_value_ks_jit:.3e}") 
print(f"KS statistic results (no JIT):  |  KS = {ks_stat_nojit:.3f}  |  p = {p_value_ks_nojit:.3e}") 
print(f"KS statistic results (no JIT genmatrix):  |  KS = {ks_stat_nojit_genmatrix:.3f}  |  p = {p_value_ks_nojit_genmatrix:.3e}") 

# diffs histograms

plt.figure() #jit
plt.hist(diffs_jit, bins='auto')
plt.xlabel('diffs_jit')
plt.ylabel('Frequency')
plt.title('Histogram of diffs')
plt.show()

plt.figure() #nojit
plt.hist(diffs_nojit, bins='auto')
plt.xlabel('diffs_nojit')
plt.ylabel('Frequency')
plt.title('Histogram of diffs')
plt.show()

plt.figure() #nojit genmatrix
plt.hist(diffs_nojit_genmatrix, bins='auto')
plt.xlabel('diffs_nojit_genmatrix')
plt.ylabel('Frequency')
plt.title('Histogram of diffs')
plt.show()


# scatterplot of diffs
# Generate an array of indices 0, 1, 2, …, len(diffs)-1
indices = np.arange(len(diffs_jit))

# jit
plt.figure()
plt.scatter(indices, diffs_jit, s=20, alpha=0.7)  # s=size of points, alpha=transparency
plt.xlabel('Iteration')
plt.ylabel('Milliseconds')
plt.title('Scatter Plot of diffs_jit vs. Iteration')
plt.grid(True)
plt.show()

# no jit
plt.figure()
plt.scatter(indices, diffs_nojit, s=20, alpha=0.7)  # s=size of points, alpha=transparency
plt.xlabel('Iteration')
plt.ylabel('Milliseconds')
plt.title('Scatter Plot of diffs_nojit vs. Iteration')
plt.grid(True)
plt.show()

# no jit general matrix
plt.figure()
plt.scatter(indices, diffs_nojit_genmatrix, s=20, alpha=0.7)  # s=size of points, alpha=transparency
plt.xlabel('Iteration')
plt.ylabel('Milliseconds')
plt.title('Scatter Plot of diffs_nojit_genmatrix vs. Iteration')
plt.grid(True)
plt.show()


# alternate hypothesis is that the distribution of manual KRR run times is smaller 
# with JIT and nice matrix assumptions 
paired_ttest_jit = ttest_rel(a = manual_jit, b = sklearn_jit,
                             nan_policy='raise',
                             alternative='less'
                             )

# without JIT but WITH nice matrix assumptions
paired_ttest_nojit = ttest_rel(a = manual_nojit, b = sklearn_nojit,
                             nan_policy='raise',
                             alternative='less'
                             )

# without JIT or nice matrix assumptions
paired_ttest_nojit_genmatrix = ttest_rel(a = manual_nojit_genmatrix, b = sklearn_nojit_genmatrix,
                             nan_policy='raise',
                             alternative='less'
                             )

# printing results
print(f"Paired t-test results (JIT):  |  t = {paired_ttest_jit.statistic:.3f}  |  p = {paired_ttest_jit.pvalue:.3e}") # jit and matrix assumptions
print(f"Paired t-test results (no JIT):  |  t = {paired_ttest_nojit.statistic:.3f}  |  p = {paired_ttest_nojit.pvalue:.3e}") # matrix assumptions but no jit  
print(f"Paired t-test results (no JIT genmatrix):  |  t = {paired_ttest_nojit_genmatrix.statistic:.3f}  |  p = {paired_ttest_nojit_genmatrix.pvalue:.3e}") # no jit or nice matrix assumptions