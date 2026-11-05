import numpy as np, pandas as pd
exec(open('tuning/trend_line4.py').read().split("print(\"Schmitt")[0])
years=n/24/365
def schmitt2(h,minbars):
    st=0; pend=0; flips=0; runs=[]
    for i in range(n):
        if C[i]<line[i]-h*ATR[i] and st!=-1: nd=-1
        elif C[i]>line[i]+h*ATR[i] and st!=1: nd=1
        else: nd=st
        if nd!=st:
            if nd==(-1 if pend<0 else 1) and pend!=0: pend+= -1 if nd==-1 else 1
            else: pend = -1 if nd==-1 else 1
            if abs(pend)>=minbars: flips+=1; st=nd; pend=0
        else: pend=0
    return flips
for h in [0.8,1.0,1.2]:
    for mb in [2,3]:
        print(f"h={h} minbars={mb}: {schmitt2(h,mb)/years:.1f} смен/год")
