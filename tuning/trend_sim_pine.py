import numpy as np, pandas as pd
exec(open('tuning/trend_line4.py').read().split("print(\"Schmitt")[0])
years=n/24/365

# точная симуляция логики Pine v6: chanMult*ATR вокруг линии KAMA, pend-счётчик, confBars
def pine_sim(chanMult,confBars,kS=30):
    ln=kama(C,10,2,kS)
    up=ln+chanMult*ATR; dn=ln-chanMult*ATR
    trend=0; pend=0; flips=0; whips=0; lastflip=-999; runs=[]; run=0
    touches=0
    for i in range(n):
        if C[i]<dn[i] and trend!=-1:
            pend = pend-1 if pend<0 else -1
        elif C[i]>up[i] and trend!=1:
            pend = pend+1 if pend>0 else 1
        else:
            pend=0
        if pend>=confBars: trend=1
        if pend<=-confBars: trend=-1
        # touch (bounce) count
        if trend==1 and L[i]<ln[i]+0.25*ATR[i] and C[i]>ln[i]: touches+=1
        if trend==-1 and H[i]>ln[i]-0.25*ATR[i] and C[i]<ln[i]: touches+=1
    # second pass to count flips & short runs
    trend=0; pend=0; flips=0; runs=[]
    for i in range(n):
        prev=trend
        if C[i]<dn[i] and trend!=-1:
            pend = pend-1 if pend<0 else -1
        elif C[i]>up[i] and trend!=1:
            pend = pend+1 if pend>0 else 1
        else:
            pend=0
        if pend>=confBars: trend=1
        if pend<=-confBars: trend=-1
        if trend!=prev and prev!=0: runs.append(i)
    fl=len(runs)
    gaps=np.diff(np.array(runs+[n]))
    short=(gaps<12).sum()  # flip within <12h = "ёрш"
    return fl/years, touches/years/12, short/years

for cm in [0.8,1.0,1.2,1.5]:
    for cb in [2,3,4]:
        f,t,s=pine_sim(cm,cb)
        print(f"chan={cm} conf={cb}: смен/год={f:6.1f} отбоев/мес={t:5.1f} быстрых(ерш)/год={s:5.1f}")
