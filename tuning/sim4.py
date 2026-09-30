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
    return dict(cnt=len(t),per_100bars=round(100*len(t)/n,3),avg=round(float(t.mean()),3),win05=round(float((t>0.5).mean()),3))

# VDO CONFLUENCE: signal fires when vdo crosses its dynamic band back toward center
# direction = mean-reversion side; then require price at/near an impulse-zone level OR deep extreme
def vdo_conf(d,vLen,sigLen,bandLen,bandMult,useRegme,hz=12,rmode="VWMA+EMA",rL=55,rS=8,extremeLo=25,extremeHi=75,mode="rev",minVol=0.0):
    v=d["tick_volume"].values.astype(float)
    c=d["close"].values
    a=impulses(d,20,1.65,14,0.0,0.0)[2]
    n=len(d)
    v=v[:n]; c=c[:n]
    vb=ema(v,20)[:n]
    signed=np.where(c>np.roll(c,1),v,np.where(c<np.roll(c,1),-v,0.0));signed[0]=0
    tot=ema(v,vLen)[:n];pos=ema(signed,vLen)[:n]
    vzo=100*np.divide(pos,tot,out=np.zeros_like(pos),where=tot!=0)
    vdo=np.clip(50+vzo*0.5,0,100)
    sig=ema(vdo,sigLen); dev=stdev(vdo,bandLen)
    ub=np.minimum(80,sig+dev*bandMult); lb=np.maximum(20,sig-dev*bandMult)
    r=regme(d,rmode,rL,rS)[:n]
    prev=np.concatenate([[np.nan],vdo[:-1]])
    pprev=np.concatenate([[np.nan]*2,vdo[:-2]])
    if mode=="cross":
        # price-side (trend continuation): VDO crosses band in direction of trend
        bullCross=np.concatenate([[False]*2,(vdo[2:]>ub[2:])&(prev[1:-1]<=ub[1:-1])])
        bearCross=np.concatenate([[False]*2,(vdo[2:]<lb[2:])&(prev[1:-1]>=lb[1:-1])])
        cond_s=bullCross; cond_b=bearCross
    elif mode=="breakout":
        # volume breakout: strong impulse bar + VDO breaks its band in bar direction
        vi=v>vb*(1.0+minVol)
        up=np.concatenate([[False],c[1:]>c[:-1]])
        dn=np.concatenate([[False],c[1:]<c[:-1]])
        cond_s=vi&(vdo>ub)&up
        cond_b=vi&(vdo<lb)&dn
    else:
        # momentum turning down while above upper band -> sell-side reaction
        bearCross=np.concatenate([[False]*2,(vdo[2:]<ub[2:])&(prev[1:-1]>=ub[1:-1])])
        bullCross=np.concatenate([[False]*2,(vdo[2:]>lb[2:])&(prev[1:-1]<=lb[1:-1])])
        # reversal (peak/trough) inside extreme region
        bearRev=np.concatenate([[False]*2,(vdo[1:-1]>=pprev[1:-1])&(vdo[2:]<vdo[1:-1])&(vdo[1:-1]>extremeHi)])
        bullRev=np.concatenate([[False]*2,(vdo[1:-1]<=pprev[1:-1])&(vdo[2:]>vdo[1:-1])&(vdo[1:-1]<extremeLo)])
        cond_s=bullCross|bullRev; cond_b=bearCross|bearRev
    if useRegme:
        cond_b=cond_b&(c<r); cond_s=cond_s&(c>r)
    a=np.asarray(a)[:n]
    out=[]
    for arr,sg in ((cond_s,1),(cond_b,-1)):
        idx=np.where(arr)[0]
        if len(idx)==0: continue
        mv=np.array([sg*(c[min(i+hz,n-1)]-c[i])/a[i] for i in idx])
        out.append((len(idx),mv,(idx)))
    cnt=sum(x[0] for x in out)
    if cnt==0: return None
    allmv=np.concatenate([x[1] for x in out])
    return dict(cnt=cnt,per_100bars=round(100*cnt/n,3),avg=round(float(allmv.mean()),3),win05=round(float((allmv>0.5).mean()),3))

tf=int(sys.argv[1]); d=load(tf)[:int(sys.argv[2])] if len(sys.argv)>2 else load(tf)
print(f"### TF={tf} bars={len(d)}")
print("--- zones sweep ---")
res=[]
for vl in (20,34,50):
 for vm in (1.65,1.8,2.0,2.2):
  for mb in (0.5,0.7):
   for mr in (0.75,):
    for rm,rl in (("VWMA+EMA",55),("VWMA+EMA",100),("LinRegVWMA",55)):
     s=zone_stats(d,vl,vm,mb,mr,rmode=rm,rL=rl)
     if s: res.append(((vl,vm,mb,mr,rm,rl),s))
res.sort(key=lambda x:-x[1]["avg"])
for k,s in res[:6]: print(k,s)
best=res[0][0]
print("--- horizons for best:",best)
for hz in (6,12,24,48):
    print(" hz",hz,zone_stats(d,*best,hz=hz))
print("--- VDO confluence sweep ---")
vr=[]
for mode in ("cross","breakout"):
 for vL in (10,14,20):
  for bm in (1.0,1.35):
   for useR in (False,True):
    for mv in ((0.0,0.5,1.0) if mode=="breakout" else (0.0,)):
     s=vdo_conf(d,vL,8,50,bm,useR,mode=mode,minVol=mv)
     if s: vr.append(((mode,vL,bm,useR,mv),s))
vr.sort(key=lambda x:-x[1]["avg"])
for k,s in vr[:14]: print(k,s)
print("--- horizons for best VDO cfg ---")
if vr:
    bk=vr[0][0]
    for hz in (6,12,24,48):
        print(" hz",hz,vdo_conf(d,bk[1],8,50,bk[2],bk[3],hz=hz,mode=bk[0],minVol=bk[4]))
