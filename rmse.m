function rmseout=rmse(a1,a2)
  diff=a1-a2;
  rmseout=sqrt(mean(diff.^2));