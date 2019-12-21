# pTax3D.mod
#
# An NLP to solve a tax example with 2-dimensional types of tax payers.
#
# 29 Mar 2005: Original AMPL coding by K. Judd and C.-L. Su.
# 20 Sep 2016: Revised by D. Ma and M. A. Saunders.
# 08 Nov 2016: 3D version created by D. Ma and M. A. Saunders.
# 03 Jan 2017: Switch to piece-wise utility
# 12 Apr 2019: real wage distribution adopted.
# Number of variables:
# Number of constraints:

# Define parameters for agents (taxpayers)
param    na;              # number of first types
param    nb;              # number of second types
param    nc;              # number of third types
set A := 1..na;           # set of first types
set B := 1..nb;           # set of second types
set C := 1..nc;           # set of third types
set T = {A, B, C};        # set of agents

# Define wages for agents (taxpayers)
param wmin;               # minimum wage level
param wmax;               # maximum wage level
param w {A};              # wage vector
param mu{B};              # mu = 1/eta# mu vector
param mu1{B};             # mu1[j] = mu[j] + 1
param alpha{C};           # ak vector for utility

param lambda0{A};         # wage distribution
param lambda {A, B, C};

param l_star {A, B, C};   # zero tax wage solution
param c_star {A, B, C};
param epsilon;

var y {i in A, j in B, k in C} >= 0.01;   # income for tax payer (i,j)
var c {i in A, j in B, k in C} >= 0.01; # 0.01; #1e-5; # consumption for tax payer (i,j)

minimize f:
   -sum {i in A, j in B, k in C}
      if c[i,j,k] - alpha[k] >= epsilon then
         lambda[i,j,k] * (log(c[i,j,k] - alpha[k]) - (y[i,j,k]/w[i])^mu1[j] / mu1[j])
   else
         lambda[i,j,k] * (-1/(2*epsilon^2)*(c[i,j,k] - alpha[k])^2
                               + 2/epsilon*(c[i,j,k] - alpha[k]) + log(epsilon) - 1.5
                               - (y[i,j,k]/w[i])^mu1[j] / mu1[j]);

subject to

Incentive {(i,j,k) in T, (p,q,r) in T: if i=p and j=q then k!=r}:
    (if c[i,j,k] - alpha[k] >= epsilon then
       (log(c[i,j,k]-alpha[k]) - (y[i,j,k]/w[i])^mu1[j] / mu1[j])
     else
        - 1/(2*epsilon^2)*(c[i,j,k] - alpha[k])^2
                               + 2/epsilon*(c[i,j,k] - alpha[k]) + log(epsilon) - 1.5
                               - (y[i,j,k]/w[i])^mu1[j] / mu1[j]
    )
  - (if c[p,q,r] - alpha[k] >= epsilon then
       (log(c[p,q,r]-alpha[k]) - (y[p,q,r]/w[i])^mu1[j] / mu1[j])
     else
        - 1/(2*epsilon^2)*(c[p,q,r] - alpha[k])^2
                               + 2/epsilon*(c[p,q,r] - alpha[k]) + log(epsilon) - 1.5
                               - (y[p,q,r]/w[i])^mu1[j] / mu1[j]
    ) >= 0;

Technology:
   sum {i in A, j in B, k in C} lambda[i,j,k]*(y[i,j,k] - c[i,j,k]) >= 0;
