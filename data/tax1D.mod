# Tax1D.mod
#
# An NLP to solve a tax example with 2-dimensional types of tax payers.
#
# 29 Mar 2005: Original AMPL coding by K. Judd and C.-L. Su.
# 20 Sep 2016: Revised by D. Ma and M. A. Saunders.
# 09 Jul 2017: Updated to match later models.
# 26 Mar 2019: Real wage distribution adopted.

# Define parameters for agents (taxpayers)
param       na;           # number of first types
set A := 1..na;           # set of first types
set T = {A};              # set of agents

# Define wages for agents (taxpayers)
param  wmin;         # minimum wage level
param  wmax;         # maximum wage level
param  w {A};        # wage vector
param  mu {A};       # mu = 1/eta # mu vector

param lambda {A};

param l_star {A};    # zero tax wage solution
param c_star {A};
param y_star {A};

var y {i in A} >= 1e-5;     # income for tax payer (i)
var c {i in A} >= 1e-5;     # consumption for tax payer (i)

minimize f: -sum {i in A}
   lambda[i] * (log(c[i]) - (y[i]/w[i])^(mu[i]+1) / (mu[i]+1));

subject to

   Incentive {i in T, p in T: i != p}:
      (log(c[i]) - (y[i]/w[i])^(mu[i]+1) / (mu[i]+1))
    - (log(c[p]) - (y[p]/w[i])^(mu[i]+1) / (mu[i]+1)) >= 0;

   Technology : sum {i in A} lambda[i]* (y[i] - c[i]) >= 0;

 # Consumption: sum {i in A} c[i] >= 0;
