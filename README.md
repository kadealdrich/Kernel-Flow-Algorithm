# Kernel Flow Algorithm

This repository holds work related to the internship/master's thesis of Kade Aldrich in the summer of 2025. The internship is joint through the SAMM Laboratory at Universite Paris 1 and Universite Paris Dauphine-PSL.

## Kernel Flow Algorithm (Parametric Case)

### Experiments:

#### Setup of experiment:

1. Solve regression problem: $y_i = f^*(x_i) + \epsilon_i$  

> Solve via reproducing kernel $k(\cdot, \cdot)$ by minimizing empirical risk (quadratic loss)  

2. Generate experimental data

> Draw $n = 10^3$ realizations $(x_i, Y_i)$ of the above model choosing:
>
>> - Candidate function $f^*$ for generating observations
>> - Variance $\sigma^2$ of the error terms $\epsilon_i$
>> - The distribution of the $x_i$'s

3. Select a kernel 

> $k(\cdot, \cdot)$ is the Gaussian kernel
> $k(\cdot, \cdot)$ is the Laplace kernel

4. For each kernel, optimize the rho criterion 

5. Study convergence by drawing pictures

> Check the prediction mean squared error of hte regression using the kernel parameters output at each iteration to check the change in performance. Compare with a naively chosen kernel and one picked through cross-validation. 

6. Check if degeneracy phenomena occur 
