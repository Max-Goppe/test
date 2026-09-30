import pandas as pd, numpy as np, sys
from sim import load, regme, impulses, ema, sma, atr, stdev

def zone_stats(d, volLen,volMult,minB,minR, maxZones=24, useWicks=True, rmode="VWMA+EMA", rL=55,rS=8, hz=12,
               minZoneBars=0, wickConfirm=False, retest=False):
    imp,vi,a=impulses(d,volLen,volMult,14,minB,minR)
    r=regme(d,rmode,rL,rS)
    o=d["open"].values;h=d["high"].values;l=d["low"].values;c=d["close"].values
    n=len(d)
    bull=imp&(c>o)&(c>r); bear=imp&(c<o)&(c<r)
    openD=[];openS=[];touches=[]
    for i in range(n):
        if bull[i]: openD.append([min(o[i],c[i]) if useWicks else max(o[i],c[i]), l[i], i, 0]); 
        if len(openD)>maxZones: openD.pop(0)
        if bear[i]: openS.append([h[i], max(o[i],c[i]) if useWicks else min(o[i],c[i]), i, 0])
        if len(openS)>maxZones: openS.pop(0)
        for arr in (openD,openS):
            for z in list(arr):
                tp,bt,st,cnt=z
                dd = 1 if arr is openD else -1
                if i<=st+minZoneBars: continue
                touched=(l[i]<=tp and h[i]>=bt)
                broken=(c[i]<bt) if dd==1 else (c[i]>tp)
                if touched:
                    cnt+=1; z[3]=cnt
                    if wickConfirm and not (dd==1 and c[i]>tp) and not (dd==-1 and c[i]<bt):
                        continue  # only count touches that closed back inside zone direction
                    if retest and cnt<2: continue
                    j=min(i+hz,n-1)
                    move=(c[j]-c[i])/a[i] if dd==1 else (c[i]-c[j])/a[i]
                    touches.append(move); arr.remove(z)
                elif broken: arr.remove(z)
    t=np.array(touches)
    if len(t)<50: return None
    return dict(cnt=len(t),per_bar=len(t)/n,avg=t.mean(),win05=(t>0.5).mean(),med=np.median(t))

def run(tf):
    d=load(tf)
    res=[]
    grid=[(vl,vm,mb,mr) for vl in (34,50) for vm in (1.5,1.65,1.8,2.0,2.2,2.5) for mb in (0.35,0.5,0.7) for mr in (0.75,1.0)]
    for g in grid:
        s=zone_stats(d,*g)
        if s: res.append((g,s,False))
        s2=zone_stats(d,*g,wickConfirm=True)
        if s2: res.append((g,s2,True))
    res.sort(key=lambda x:-x[1]["avg"])
    print(f"=== TF={tf} bars={len(d)} ===")
    for g,s,wc in res[:10]: print("wickConf" if wc else "plain   ", g, {k:(round(v,3) if isinstance(v,float) else v) for k,v in s.items()})
    best=res[0]
    # test more horizons & maxZones for best config
    for hz in (6,12,24,48):
        s=zone_stats(d,*best[0],wickConfirm=best[2],hz=hz)
        print("  hz",hz,{k:round(v,3) for k,v in s.items() if isinstance(v,float)},s["cnt"])
    for mz in (12,24,40):
        s=zone_stats(d,*best[0],wickConfirm=best[2],maxZones=mz)
        print("  maxZones",mz,{k:round(v,3) for k,v in s.items() },s["cnt"],round(s["avg"],3),round(s["win05"],3))

for tf in [int(x) for x in sys.argv[1:]] or [5,15,60]:
    run(tf)
