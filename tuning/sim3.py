import pandas as pd, numpy as np, sys
from sim import load, regme, impulses, ema, stdev

def zone_stats(d, volLen,volMult,minB,minR, rmode="VWMA+EMA", rL=55, maxZones=24, rS=8, hz=12, wickConfirm=True):
    imp,vi,a=impulses(d,volLen,volMult,14,minB,minR)
    r=regme(d,rmode,rL,rS)
    o=d["open"].values;h=d["high"].values;l=d["low"].values;c=d["close"].values
    n=len(d)
    bull=imp&(c>o)&(c>r); bear=imp&(c<o)&(c<r)
    openD=[];openS=[];touches=[]
    for i in range(n):
        if bull[i]: openD.append([min(o[i],c[i]), l[i], i])
        if len(openD)>maxZones: openD.pop(0)
        if bear[i]: openS.append([h[i], max(o[i],c[i]), i])
        if len(openS)>maxZones: openS.pop(0)
        for arr in (openD,openS):
            for z in list(arr):
                tp,bt,st=z
                dd = 1 if arr is openD else -1
                if i<=st: continue
                touched=(l[i]<=tp and h[i]>=bt)
                broken=(c[i]<bt) if dd==1 else (c[i]>tp)
                if touched:
                    okc = (dd==1 and c[i]>tp) or (dd==-1 and c[i]<bt)
                    if wickConfirm and not okc:
                        if broken: arr.remove(z)
                        continue
                    j=min(i+hz,n-1)
                    move=(c[j]-c[i])/a[i] if dd==1 else (c[i]-c[j])/a[i]
                    touches.append(move); arr.remove(z)
                elif broken: arr.remove(z)
    t=np.array(touches)
    if len(t)<30: return None
    return dict(cnt=len(t),per_100bars=100*len(t)/n,avg=t.mean(),win05=(t>0.5).mean())

def vdo_stats(d,vLen,sigLen,bandLen,bandMult,reqI,volLen,volMult,useR,hz=12,rmode="VWMA+EMA",rL=55,rS=8):
    n=len(d)
    v=d["tick_volume"].values.astype(float)[:n]
    c=d["close"].values[:n]
    signed=np.where(c>np.roll(c,1),v,np.where(c<np.roll(c,1),-v,0.0));signed[0]=0
    tot=ema(v,vLen);pos=ema(signed,vLen)
    vzo=100*np.divide(pos,tot,out=np.zeros_like(pos),where=tot!=0)
    vdo=np.clip(50+vzo*0.5,0,100)
    sig=ema(vdo,sigLen); dev=stdev(vdo,bandLen)
    ub=np.minimum(80,sig+dev*bandMult); lb=np.maximum(20,sig-dev*bandMult)
    _,vi,a=impulses(d,volLen,volMult,14,0.0,0.0)
    vi=np.asarray(vi)[:n]; a=np.asarray(a)[:n]
    r=regme(d,rmode,rL,rS)[:n]
    bt=lambda x: np.concatenate([[False]*2,x[:len(x)-2]])
    bullturn=bt((vdo[2:]<lb[2:])&(vdo[2:]>vdo[1:-1])&(vdo[1:-1]<=vdo[:-2]))
    bearturn=bt((vdo[2:]>ub[2:])&(vdo[2:]<vdo[1:-1])&(vdo[1:-1]>=vdo[:-2]))
    cok=vi if reqI else np.ones(n,dtype=bool)
    rcond=(c>r) if useR else np.ones(n,dtype=bool)
    bs=bullturn&cok[:len(bullturn)]&rcond[:len(bullturn)]
    ss=bearturn&cok[:len(bearturn)]&rcond[:len(bearturn)]
    out=[]
    for arr,sg in ((bs,1),(ss,-1)):
        idx=np.where(arr)[0]
        if len(idx)==0: continue
        mv=np.array([sg*(c[min(i+hz,n-1)]-c[i])/a[i] for i in idx])
        out.append((len(idx),mv.mean(),(mv>0.5).mean()))
    if not out: return None
    cnt=sum(x[0] for x in out)
    av=np.average([x[1] for x in out],weights=[x[0] for x in out])
    w=np.average([x[2] for x in out],weights=[x[0] for x in out])
    return dict(cnt=cnt,per_100bars=100*cnt/n,avg=round(av,3),win05=round(w,3))

tf=int(sys.argv[1]); d=load(tf)[:int(sys.argv[2])] if len(sys.argv)>2 else load(tf)
print(f"### TF={tf} bars={len(d)}")
print("--- zones sweep (maxZones fixed 24) ---")
res=[]
for vl in (20,34,50):
 for vm in (1.5,1.65,1.8,2.0,2.2):
  for mb in (0.35,0.5,0.7):
   for mr in (0.75,1.0):
    for rm,rl in (("VWMA+EMA",55),("Session VWAP",55),("LinRegVWMA",55),("VWMA+EMA",100)):
     s=zone_stats(d,vl,vm,mb,mr,rmode=rm,rL=rl)
     if s: res.append(((vl,vm,mb,mr,rm,rl),s))
res.sort(key=lambda x:-x[1]["avg"])
for k,s in res[:8]: print(k,{kk:(round(vv,3) if isinstance(vv,(float,np.floating)) else vv) for kk,vv in s.items()})
best=res[0][0]
print("--- best zone cfg across horizons:",best)
for hz in (6,12,24,48):
    s=zone_stats(d,*best,hz=hz); print(" hz",hz,{kk:(round(vv,3) if isinstance(vv,(float,np.floating)) else vv) for kk,vv in s.items()})
print("--- maxZones sensitivity for best cfg:",best)
for mz in (12,24,40,60):
    s=zone_stats(d,*best,maxZones=mz); print(" mz",mz,{kk:(round(vv,3) if isinstance(vv,(float,np.floating)) else vv) for kk,vv in s.items()})
print("--- VDO sweep ---")
vr=[]
for vL in (10,14,20):
 for bm in (1.0,1.35,1.7):
  for reqI in (False,True):
   for useR in (False,True):
    s=vdo_stats(d,vL,8,50,bm,reqI,34,1.65,useR)
    if s: vr.append(((vL,bm,reqI,useR),s))
vr.sort(key=lambda x:-x[1]["avg"])
for k,s in vr[:8]: print(k,s)
