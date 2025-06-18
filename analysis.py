import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np 

### File for analyizing data related to experiments with kernel flow algorithm

gvr = pd.read_csv("gamma-vs-rho.csv")

# variables: gamma_init, gamma_final, rho_init, rho_final

# removing crazy outliers
gvr_filt = gvr[gvr['rho_final'] >= -2]



## PLOTS

# gamma init vs rho final
plt.figure(figsize = (8,5))
plt.scatter(gvr_filt['gamma_init'], gvr_filt['rho_final'], marker = 'o')
plt.xlabel('Initial gamma')
plt.ylabel('Final rho')
plt.title('Initial gamma vs Final rho\n(Ignoring rho < -2)')
#plt.show()
# seems like weird stuff really only happens when starting gamma near zero

# gamma final vs rho final
plt.figure(figsize = (8,5))
plt.scatter(gvr_filt['gamma_final'], gvr_filt['rho_final'], marker = 'o')
plt.xlabel('Final gamma')
plt.ylabel('Final rho')
plt.title('Final gamma vs Final rho\n(Ignoring rho < -2)')
#plt.show()
# looks pretty identical to plot with initial gamma
#   going to look at plot of differences of gamma vs differences of rho 

# gamma init vs rho init 
plt.figure(figsize = (8,5))
plt.scatter(gvr['gamma_init'], gvr['rho_init'], marker = 'o')
plt.xlabel('Initial gamma')
plt.ylabel('Initial rho')
plt.title('Initial gamma vs Inital rho')
plt.show()
