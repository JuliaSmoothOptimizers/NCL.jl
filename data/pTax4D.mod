# pTax4D.mod	
#
# 08 Dec 2016: 4D version created by D. Ma and M. A. Saunders.
# 04 Jan 2017: Switch to piece-wise utility 
# 12 Apr 2019: real wage distribution adopted. 	       

 
# Define parameters for agents (taxpayers)
param na;                # number of types in wage
param nb;                # number of types in eta
param nc;                # number of types in alpha
param nd;                # number of types in psi
set A := 1..na;          # set of wages
set B := 1..nb;          # set of eta
set C := 1..nc;          # set of alpha
set D := 1..nd;          # set of psi
set T = {A,B,C,D};     # set of agents

# Define wages for agents (taxpayers)
param wmin;               # minimum wage level
param wmax;               # maximum wage level
param w {A};              # wage vector
param mu{B};              # mu = 1/eta# mu vector
param mu1{B};             # mu1[j] = mu[j] + 1
param alpha{C};           # ak vector for utility 
param psi{D};             # g

param lambda0{A};         # wage distribution
param lambda {A, B, C, D};

param l_star {A, B, C, D};   # zero tax wage solution 
param c_star {A, B, C, D};
param epsilon; 
param primreg     default 1e-8;  # Small primal regularization

var c{(i,j,k,g) in T} >= 0.1;  # consumption for tax payer (i,j,k,g)
var y{(i,j,k,g) in T} >= 0.1;  # income      for tax payer (i,j,k,g) 

minimize f:
   sum{(i,j,k,g) in T}
   (
      if c[i,j,k,g] - alpha[k] >= epsilon then
          - lambda[i,j,k,g] * (
               log(c[i,j,k,g] - alpha[k])
               - psi[g] * (y[i,j,k,g]/w[i])^mu1[j] / mu1[j]
	    )
       else
          - lambda[i,j,k,g] * (
		 -1/(2*epsilon^2)*(c[i,j,k,g] - alpha[k])^2
                      + 2/epsilon*(c[i,j,k,g] - alpha[k]) + log(epsilon) - 1.5
                 - psi[g] * (y[i,j,k,g]/w[i])^mu1[j] / mu1[j]
            )
   + 0.5 * primreg * (c[i,j,k,g]^2 + y[i,j,k,g]^2)
   );
 
subject to

Incentive {(i,j,k,g) in T, (p,q,r,s) in T:
          !(i=p and j=q and k=r and g=s)}:
   (
       if c[i,j,k,g] - alpha[k] >= epsilon then
        log(c[i,j,k,g] - alpha[k])
        - psi[g] * (y[i,j,k,g]/w[i])^mu1[j] / mu1[j]
       else
        -1/(2*epsilon^2)*(c[i,j,k,g] - alpha[k])^2
                + 2/epsilon*(c[i,j,k,g] - alpha[k]) + log(epsilon) - 1.5
        - psi[g] * (y[i,j,k,g]/w[i])^mu1[j] / mu1[j]
   )
 - (
	if c[p,q,r,s] - alpha[k] >= epsilon then
         log(c[p,q,r,s] - alpha[k])
         - psi[g]*(y[p,q,r,s]/w[i])^mu1[j] / mu1[j]
        else
         -1/(2*epsilon^2)*(c[p,q,r,s] - alpha[k])^2
                + 2/epsilon*(c[p,q,r,s] - alpha[k]) + log(epsilon) - 1.5
         - psi[g] * (y[p,q,r,s]/w[i])^mu1[j] / mu1[j]
   )
   >= 0;

Technology:
   sum{(i,j,k,g) in T} lambda[i,j,k,g]*(y[i,j,k,g] - c[i,j,k,g]) >= 0;


