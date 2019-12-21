# Tax2D.mod
#
# An NLP to solve a tax example with 2-dimensional types of tax payers.
#
# 29 Mar 2005: Original AMPL coding by K. Judd and C.-L. Su.
# 20 Sep 2016: Revised by D. Ma and M. A. Saunders.
# 01 Aug 2017: Put sensible lower bounds on c, y.
# 11 Aug 2017: Define mu1[j] = mu[j] + 1.
# 11 Aug 2017: Add small primal regularization term 0.5e-7*etak (x - xk)^2
#              to match NCL at final k with etak = 1e-7.
#              Luckily this is small enough not to cause more superbasics.
# 14 Aug 2017: Make it 0.5 * gamma * (x - xk)^2.
# 25 Aug 2017: Make it 0.5 * gamma * ||x||^2, gamma = 1e-7.
# 29 Aug 2017: Make it 0.5 * gamma * ||x||^2, gamma = 1e-8.
# 08 Apr 2019: Real wage distribution adopted.

# Define parameters for agents (taxpayers)
param na;                 # number of first types
param nb;                 # number of second types
set A := 1..na;           # set of first types
set B := 1..nb;           # set of second types
set T = {A, B};           # set of agents

# Define wages for agents (taxpayers)
param wmin;               # minimum wage level
param wmax;               # maximum wage level
param w {A};              # wage vector
param mu{B};              # mu = 1/eta # mu vector
param mu1{B};             # mu1[j] = mu[j] + 1

param lambda0{A};         # wage distribution
param lambda{A,B};
param l_star{A,B};        # zero tax wage solution
param c_star{A,B};

param clb{A,B} > 0, default 0.1;    # lower bounds on c
param ylb{A,B} > 0, default 0.1;    # lower bounds on y

param gamma     default 1e-6;       # Small primal regularization

var c{i in A, j in B} >= clb[i,j];  # consumption of tax payer (i,j)
var y{i in A, j in B} >= ylb[i,j];  # income      of tax payer (i,j)

minimize f:
   -sum{i in A, j in B}
      (lambda[i,j] * (log(c[i,j]) - (y[i,j]/w[i])^mu1[j] / mu1[j])
       - 0.5 * gamma * (c[i,j]^2 + y[i,j]^2));

subject to
   Incentive{(i,j) in T, (p,q) in T: if i=p then j!=q}:
     (log(c[i,j]) - (y[i,j]/w[i])^mu1[j] / mu1[j])
   - (log(c[p,q]) - (y[p,q]/w[i])^mu1[j] / mu1[j]) >= 0;

Technology:
   sum{i in A, j in B} lambda[i,j]*(y[i,j] - c[i,j]) >= 0;

# Consumption:
# sum{i in A, j in B} c[i,j] >= 0;
