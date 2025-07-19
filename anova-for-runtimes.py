### File for getting stats for ANOVA for algorithm runtimes 
# Kade Aldrich
# Internship 2025


import pandas as pd
import numpy as np 
from scipy.stats import f


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