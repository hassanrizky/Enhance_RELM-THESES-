function maeout=mae(a1,a2)
  diff=abs(a1-a2);
  maeout=mean(diff);