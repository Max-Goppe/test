exec(open('backtest/scalp_sim2.py').read().split('print("=== BASE')[0])

import itertools, math

def vdo_series2(length,sig_len,band_len,band_mult):
    sv=[0.0]*n
    for i in range(1,n):
        sv[i]=V[i] if C[i]>C[i-1] else (-V[i] if C[i]<C[i-1] else 0.0)
    tv=ema(V,length); vp=ema(sv,length)
    vzo=[100.0*vp[i]/tv[i] if tv[i]!=0 else 0.0 for i in range(n)]
    vdo=[max(0.0,min(100.0,50.0+z*0.5)) for z in vzo]
    sig=ema(vdo,sig_len)
    dev=[math.sqrt(max(0.0,sum((vdo[j]-sig[j])**2 for j in range(max(0,i-band_len+1),i+1))/min(i+1,band_len))) for i in range(n)]
    up=[min(80.0,sig[i]+dev[i]*band_mult) for i in range(n)]
    dn=[max(20.0,sig[i]-dev[i]*band_mult) for i in range(n)]
    return vdo,sig,up,dn

def simulate2(regLen=34, smooth=8, volMult=1.6, bodyMin=0.75, zoneMax=16, cooldown=6,
             requireAll=True, combo='touch', vdoP=(14,8,50,1.35), spread=0.0, hold=None,
             timeStopBars=None, trendStep=12, confStep=3):
    A=ATR
    regE=ema(vwma(HL,V,regLen),smooth)
    above=[C[i]>=regE[i] for i in range(n)]
    vb=sma(V,50)
    conf=bull_on_big(confStep,regLen,smooth)
    trend=bull_on_big(trendStep,regLen,smooth)
    vdo,vsig,vup,vdn=vdo_series2(*vdoP)
    res=[]; zones_d=[]; zones_s=[]; last_sig=-999; pend=None
    for i in range(60,n):
        if pend is not None:
            side,en,sl,tp,eb,atr0=pend
            closed=False
            if side=='buy':
                if L[i]<=sl+spread: res.append(-1.0-(spread/ (en-sl))); closed=True
                elif H[i]>=tp-spread: res.append(+1.0-(spread/(en-sl))); closed=True
            else:
                if H[i]>=sl-spread: res.append(-1.0-(spread/(sl-en))); closed=True
                elif L[i]<=tp+spread: res.append(+1.0-(spread/(sl-en))); closed=True
            if not closed and timeStopBars is not None and i-eb>=timeStopBars:
                pnl = ((C[i]-en)/(en-sl)) if side=='buy' else ((en-C[i])/(sl-en))
                pnl -= spread/(en-sl)
                res.append(pnl); closed=True
            if closed: pend=None
        imp=vb[i]>0 and V[i]>vb[i]*volMult
        body=abs(C[i]-O[i]); rng=H[i]-L[i]
        ok=body>=A[i]*bodyMin and rng>=A[i]*0.9
        if imp and ok and C[i]>O[i] and above[i]:
            zones_d.append([min(O[i],C[i]),L[i],i,False])
            if len(zones_d)>zoneMax: zones_d.pop(0)
        if imp and ok and C[i]<O[i] and not above[i]:
            zones_s.append([H[i],max(O[i],C[i]),i,False])
            if len(zones_s)>zoneMax: zones_s.pop(0)
        td=False; ts=False; zd=None; zs=None
        for z in zones_d:
            if not z[3] and i>z[2] and L[i]<=z[0] and H[i]>=z[1]: z[3]=True; td=True; zd=z
        for z in zones_s:
            if not z[3] and i>z[2] and H[i]>=z[1] and L[i]<=z[0]: z[3]=True; ts=True; zs=z
        mb=(1 if above[i] else 0)+(1 if conf[i] else 0)+(1 if trend[i] else 0)
        bull_ok = mb==3 if requireAll else mb>=2
        bear_ok = (3-mb)==3 if requireAll else (3-mb)>=2
        cv=True; sv_=True
        if combo=='touch+vdo':
            brkU = i>0 and vdo[i]>vup[i] and vdo[i-1]<=vup[i-1]
            brkD = i>0 and vdo[i]<vdn[i] and vdo[i-1]>=vdn[i-1]
            cv = brkU or vdo[i]<30
            sv_= brkD or vdo[i]>70
        buy = bull_ok and above[i] and td and cv and pend is None and i-last_sig>cooldown
        sell= bear_ok and (not above[i]) and ts and sv_ and pend is None and i-last_sig>cooldown
        if buy or sell:
            last_sig=i
            if buy:
                en=C[i]; sl=zd[1]-0.15*A[i]; risk=en-sl
                if risk>0: pend=('buy',en,sl,en+(hold or 2.0)*risk,i,A[i])
            else:
                en=C[i]; sl=zs[0]+0.15*A[i]; risk=sl-en
                if risk>0: pend=('sell',en,sl,en-(hold or 2.0)*risk,i,A[i])
    tot=len(res)
    if not tot: return (0,0,0,0,0)
    w=sum(1 for r in res if r>0); wr=w/tot
    gp=sum(r for r in res if r>0); gl=-sum(r for r in res if r<=0)
    pf=gp/gl if gl>0 else 99
    edge=sum(res)/tot
    return tot,w,round(wr,3),round(pf,2),round(edge,3)

print("combo strict, hold sweep, spread 0.25:")
for h in [1.5,2.0,3.0]:
    print("h",h, simulate2(combo='touch+vdo',hold=h,spread=0.25))
print("combo strict + timestop 24 bars, no TP:")
print(simulate2(combo='touch+vdo',hold=99,timeStopBars=24,spread=0.25))
print("touch-only, vm2.0 bm0.5, hold2, spread0.25:")
print(simulate2(combo='touch',volMult=2.0,bodyMin=0.5,hold=2.0,spread=0.25))
print("combo strict, vm2.0 bm0.5, hold2, spread0.25:")
print(simulate2(combo='touch+vdo',volMult=2.0,bodyMin=0.5,hold=2.0,spread=0.25))
