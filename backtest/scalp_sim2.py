import csv, glob, math, collections, itertools

def load():
    rows=[]
    for f in sorted(glob.glob('/workspace/2022-2026/GOLD_M5_*.csv')):
        with open(f, encoding="utf-16") as fh:
            for r in csv.reader(fh):
                if len(r)<6 or not r[0][:4].isdigit(): continue
                try: o,h,l,c,v=float(r[1]),float(r[2]),float(r[3]),float(r[4]),float(r[5])
                except: continue
                rows.append((o,h,l,c,v))
    return rows

R=load(); n=len(R)
O=[r[0] for r in R]; H=[r[1] for r in R]; L=[r[2] for r in R]; C=[r[3] for r in R]; V=[r[4] for r in R]

def ema(x,p):
    k=2/(p+1); out=[0.0]*len(x); e=x[0]
    for i,v in enumerate(x): e=v*k+e*(1-k); out[i]=e
    return out

def sma(x,p):
    q=collections.deque(); s=0.0; out=[0.0]*len(x)
    for i,v in enumerate(x):
        q.append(v); s+=v
        if len(q)>p: s-=q.popleft()
        out[i]=s/len(q)
    return out

TR=[H[0]-L[0]]+[max(H[i]-L[i],abs(H[i]-C[i-1]),abs(L[i]-C[i-1])) for i in range(1,n)]
ATR=ema(TR,14)
HL=(H+L+C)/3.0 if False else [(H[i]+L[i]+C[i])/3 for i in range(n)]

def vwma(x,vols,p):
    qn=collections.deque(); qd=collections.deque(); sn=0.0; sd=0.0; out=[0.0]*len(x)
    for i in range(len(x)):
        a=x[i]*vols[i]; b=vols[i]
        qn.append(a); qd.append(b); sn+=a; sd+=b
        if len(qn)>p: sn-=qn.popleft(); sd-=qd.popleft()
        out[i]=sn/sd if sd!=0 else x[i]
    return out

def resample(step):
    idx=list(range(0,n,step)); m=len(idx)
    ro=[0.0]*m; rh=[0.0]*m; rl=[0.0]*m; rc=[0.0]*m; rv=[0.0]*m
    for j,i0 in enumerate(idx):
        i1=min(i0+step,n)
        ro[j]=O[i0]; rh[j]=max(H[i0:i1]); rl[j]=min(L[i0:i1]); rc[j]=C[i1-1]; rv[j]=sum(V[i0:i1])
    return idx,ro,rh,rl,rc,rv

def bull_on_big(step,regLen,smooth):
    idx,ro,rh,rl,rc,rv=resample(step)
    hlc=[(rh[k]+rl[k]+rc[k])/3 for k in range(len(rc))]
    reg=ema(vwma(hlc,rv,regLen),smooth)
    big_bull=[rc[k]>=reg[k] for k in range(len(rc))]
    # map to main bars using LAST COMPLETED big bar only (no lookahead)
    out=[False]*n; bj=-1
    for i in range(n):
        while bj+1<len(idx) and i>=idx[bj+1]:  # big bar bj completed when we reach start of bj+1
            bj+=1
        out[i]=big_bull[bj] if bj>=0 else big_bull[0]
    return out

def vdo_series(length,sig_len,band_len,band_mult):
    sv=[0.0]*n
    for i in range(1,n):
        sv[i]=V[i] if C[i]>C[i-1] else (-V[i] if C[i]<C[i-1] else 0.0)
    tv=ema(V,length); vp=ema(sv,length)
    vzo=[100.0*vp[i]/tv[i] if tv[i]!=0 else 0.0 for i in range(n)]
    vdo=[max(0.0,min(100.0,50.0+z*0.5)) for z in vzo]
    sig=ema(vdo,sig_len)
    dev=sma([(vdo[i]-sig[i])**2 for i in range(n)],band_len)
    dev=[math.sqrt(d) for d in dev]
    up=[min(80.0,sig[i]+dev[i]*band_mult) for i in range(n)]
    dn=[max(20.0,sig[i]-dev[i]*band_mult) for i in range(n)]
    return vdo,sig,up,dn

def simulate(regLen=34, smooth=8, volMult=1.6, bodyMin=0.75, zoneMax=16, cooldown=6,
             requireAll=True, tpR=1.5, combo='touch', vdoP=(14,8,50,1.35), spread=0.0):
    A=ATR
    regE=ema(vwma(HL,V,regLen),smooth)
    above=[C[i]>=regE[i] for i in range(n)]
    vb=sma(V,50)
    conf=bull_on_big(3,regLen,smooth)
    trend=bull_on_big(12,regLen,smooth)
    vdo,vsig,vup,vdn=vdo_series(*vdoP)
    trades=[]
    zones_d=[]; zones_s=[]
    last_sig=-999; pend=None
    for i in range(60,n):
        # manage open position
        if pend is not None:
            side,en,sl,tp,eb=pend
            if side=='buy':
                if L[i]<=sl+spread: trades.append(-1.0); pend=None
                elif H[i]>=tp-spread: trades.append(+1.0); pend=None
            else:
                if H[i]>=sl-spread: trades.append(-1.0); pend=None
                elif L[i]<=tp+spread: trades.append(+1.0); pend=None
        # zone creation
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
            if not z[3] and i>z[2] and L[i]<=z[0] and H[i]>=z[1]:
                z[3]=True; td=True; zd=z
        for z in zones_s:
            if not z[3] and i>z[2] and H[i]>=z[1] and L[i]<=z[0]:
                z[3]=True; ts=True; zs=z
        mb=(1 if above[i] else 0)+(1 if conf[i] else 0)+(1 if trend[i] else 0)
        ms=3-mb
        bull_ok = mb==3 if requireAll else mb>=2
        bear_ok = ms==3 if requireAll else ms>=2
        cv=True; sv_=True
        if combo=='touch+vdo':
            cv = (vdo[i]>vsig[i]) or (vdo[i]<30) or (i>0 and vdo[i-1]<=vup[i-1] and vdo[i]>vup[i])
            sv_= (vdo[i]<vsig[i]) or (vdo[i]>70) or (i>0 and vdo[i-1]>=vdn[i-1] and vdo[i]<vdn[i])
        buy = bull_ok and above[i] and td and cv and pend is None and i-last_sig>cooldown
        sell= bear_ok and (not above[i]) and ts and sv_ and pend is None and i-last_sig>cooldown
        if buy or sell:
            last_sig=i
            if buy:
                en=C[i]; sl=zd[1]-0.15*A[i]; risk=en-sl
                if risk>0: pend=('buy',en,sl,en+tpR*risk,i)
            else:
                en=C[i]; sl=zs[0]+0.15*A[i]; risk=sl-en
                if risk>0: pend=('sell',en,sl,en-tpR*risk,i)
    if not trades: return (0,0,0,0,0)
    tot=len(trades); w=sum(1 for t in trades if t>0)
    wr=w/tot
    pf=(w*tpR)/max(1,(tot-w))
    edge=(wr*tpR-(1-wr))
    return tot,w,round(wr,3),round(pf,2),round(edge,3)

print("=== BASE M5 ladder (chart M5 -> confirm M15 -> trend H1), touch-only ===")
for tpR in [1.0,1.5,2.0]:
    for cd in [4,6,10]:
        print("tpR",tpR,"cd",cd, simulate(tpR=tpR,cooldown=cd,combo='touch'))

print("=== combo (touch + VDO confirm) ===")
for tpR in [1.0,1.5,2.0]:
    print("tpR",tpR, simulate(tpR=tpR,cooldown=6,combo='touch+vdo'))

print("=== vol multiplier / body filter sweep (tpR=1.5, touch-only) ===")
for vm in [1.3,1.6,2.0,2.4]:
    for bm in [0.5,0.75,1.0]:
        print("vm",vm,"bm",bm, simulate(volMult=vm,bodyMin=bm,tpR=1.5,combo='touch'))

print("=== requireAll off (2 of 3) ===")
print(simulate(requireAll=False,tpR=1.5,combo='touch'))
print(simulate(requireAll=False,tpR=1.5,combo='touch+vdo'))

print("=== with spread 0.35 USD (~ typical XAU) tpR=1.5 ===")
print(simulate(tpR=1.5,combo='touch',spread=0.35))
print(simulate(tpR=1.5,combo='touch+vdo',spread=0.35))
print(simulate(tpR=2.0,combo='touch',spread=0.35))
