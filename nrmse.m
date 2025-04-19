function nrmseout=nrmse(a1,act2)
  diff=a1-act2;
  nrmseout=sqrt(mean(diff.^2))/(max(act2)-min(act2));