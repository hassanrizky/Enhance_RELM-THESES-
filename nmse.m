function nmseout=nmse(a1,act2)
  diff=a1-act2;
  nmseout=(sum(diff.^2))/(sum(act2.^2));