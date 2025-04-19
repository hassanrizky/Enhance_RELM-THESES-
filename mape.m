function mapeout=mape(a1,act2)
  diff=abs(a1-act2);
  mapeout=mean(diff./act2);