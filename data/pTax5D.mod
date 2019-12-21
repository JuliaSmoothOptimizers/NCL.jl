# pTax5D.mod	
#
# 15 Apr 2019: Created for real tax distribution.

# Number of variables:
# Number of constraints:

# Define parameters for agents (taxpayers)
param na;                # number of types in wage
param nb;                # number of types in eta
param nc;                # number of types in alpha
param nd;                # number of types in psi
param ne;                # number of types in gamma
set A := 1..na;          # set of wages
set B := 1..nb;          # set of eta
set C := 1..nc;          # set of alpha
set D := 1..nd;          # set of psi
set E := 1..ne;          # set of gamma
set T = {A,B,C,D,E};     # set of agents

# Define wages for agents (taxpayers)
param wmin;              # minimum wage level
param wmax;              # maximum wage level
param w {A};             # wage vector
param mu{B};             # mu = 1/eta# mu vector
param mu1{B};            # mu1[j] = mu[j] + 1
param alpha{C};          # ak vector for utility
param psi{D};            # g
param gamma{E};          # h

param lambda0{A};        # wage distribution
param lambda{A,B,C,D,E}; # distribution density
param epsilon;
param primreg     default 1e-8;  # Small primal regularization

param c_star{(i,j,k,g,h) in T};
param l_star{(i,j,k,g,h) in T};
var c{(i,j,k,g,h) in T} >= 0.1;  # consumption for tax payer (i,j,k,g,h)
var y{(i,j,k,g,h) in T} >= 0.1;  # income      for tax payer (i,j,k,g,h)

minimize f:
   sum{(i,j,k,g,h) in T}
   (
	(
	  if c[i,j,k,g,h] - alpha[k] >= epsilon then
          - lambda[i,j,k,g,h] * (
               (c[i,j,k,g,h] - alpha[k])^(1-1/gamma[h]) / (1-1/gamma[h])
               - psi[g]*(y[i,j,k,g,h]/w[i])^mu1[j] / mu1[j]
	    )
       else
          - lambda[i,j,k,g,h] * (
            - 0.5/gamma[h] * epsilon^(-1/gamma[h]-1) * (c[i,j,k,g,h] - alpha[k])^2
            + (1+1/gamma[h])* epsilon^(-1/gamma[h] ) * (c[i,j,k,g,h] - alpha[k])
            + (1/(1-1/gamma[h]) - 1 - 0.5/gamma[h]) * epsilon^(1-1/gamma[h])
            - psi[g]*(y[i,j,k,g,h]/w[i])^mu1[j] / mu1[j]
	    )
        )
   + 0.5 * primreg * (c[i,j,k,g,h]^2 + y[i,j,k,g,h]^2)
   );


subject to

Incentive{(i,j,k,g,h) in T, (p,q,r,s,t) in T:
          !(i=p and j=q and k=r and g=s and h=t)}:
   (
     if c[i,j,k,g,h] - alpha[k] >= epsilon then
      (c[i,j,k,g,h] - alpha[k])^(1-1/gamma[h]) / (1-1/gamma[h])
       - psi[g]*(y[i,j,k,g,h]/w[i])^mu1[j] / mu1[j]
    else
       -  0.5/gamma[h] *epsilon^(-1/gamma[h]-1)*(c[i,j,k,g,h] - alpha[k])^2
       + (1+1/gamma[h])*epsilon^(-1/gamma[h]  )*(c[i,j,k,g,h] - alpha[k])
       + (1/(1-1/gamma[h]) - 1 - 0.5/gamma[h])*epsilon^(1-1/gamma[h])
       - psi[g]*(y[i,j,k,g,h]/w[i])^mu1[j] / mu1[j]
   )
 - (
     if c[p,q,r,s,t] - alpha[k] >= epsilon then
      (c[p,q,r,s,t] - alpha[k])^(1-1/gamma[h]) / (1-1/gamma[h])
       - psi[g]*(y[p,q,r,s,t]/w[i])^mu1[j] / mu1[j]
     else
       -  0.5/gamma[h] *epsilon^(-1/gamma[h]-1)*(c[p,q,r,s,t] - alpha[k])^2
       + (1+1/gamma[h])*epsilon^(-1/gamma[h]  )*(c[p,q,r,s,t] - alpha[k])
       + (1/(1-1/gamma[h]) - 1 - 0.5/gamma[h])*epsilon^(1-1/gamma[h])
       - psi[g]*(y[p,q,r,s,t]/w[i])^mu1[j] / mu1[j]
   ) >= 0;

Technology:
   sum{(i,j,k,g,h) in T} lambda[i,j,k,g,h]*(y[i,j,k,g,h] - c[i,j,k,g,h]) >= 0;
