model;

var x {1..2} >= 0;

minimize obj:
  x[1]^2 + x[2]^2;

subject to compl:
  x[1] * x[2] <= 0;
