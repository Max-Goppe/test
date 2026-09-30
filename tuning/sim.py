import pandas as pd, numpy as np, itertools, json, sys

def load(tf):
    parts=[]
    for y in range(2022,2027):
        df=pd.read_csv(f"/workspace/2022-2026/GOLD_M5_{y}.csv",header=None,encoding="utf-16",
            names=["dt","open","high","low","close","tick_volume","real_volume"])
        parts.append(df)
    d=pd.concat(parts,ignore_index=True)
    d["dt"]=pd.to_datetime(d["dt"],format="%Y.%m.%d %H:%M")
    d=d.drop_duplicates("dt").sort_values("dt").reset_index(drop=True)
    if tf==5: return d
    agg={"open":"first","high":"max","low":"min","close":"last","tick_volume":"sum","real_volume":"sum"}
    d=d.set_index("dt").resample(f"{tf}min").agg(agg).dropna().reset_index()
    return d

def ema(x,n): return pd.Series(x).ewm(span=n,adjust=False).mean().values
def sma(x,n): return pd.Series(x).rolling(n).mean().values
def atr(h,l,c,n):
    pc=c[1:]
    tr=np.concatenate([[h[0]-l[0]],np.maximum(h[1:]-l[1:],np.maximum(abs(h[1:]-pc),abs(l[1:]-pc)))])
    return ema(tr,n)
def vwma(p,v,n):
    pv=p*v
    return pd.Series(pv).rolling(n).mean().values/pd.Series(v).rolling(n).mean().values
def stdev(x,n): return pd.Series(x).rolling(n).std(ddof=0).values
def linreg(x,n,off=0):
    x=np.asarray(x,dtype=float)
    out=np.full(len(x),np.nan)
    idx=np.arange(n); im=idx.mean()
    denom=(idx**2).sum()-n*im*im
    rollmean=pd.Series(x).rolling(n).mean().values
    wx=np.convolve(x,idx[::-1],mode='valid')  # len = N-n+1
    slope=(wx-n*im*rollmean[n-1:])/denom
    out[n-1:]=rollmean[n-1:]+slope*((n-1)-im)
    return out

def regme(d,mode,length,smooth):
    v=d["tick_volume"].values.astype(float); v=np.where(v<=0,1e-9,v)
    p=d["close"].values; h=d["high"].values; l=d["low"].values
    vw=vwma((h+l+p)/3.0,v,length)
    if mode=="VWMA+EMA": r=ema(vw,smooth)
    elif mode=="LinRegVWMA": r=linreg(vw,smooth)
    else:
        # session VWAP reset daily
        c=(h+l+p)/3.0; day=d["dt"].dt.floor("D")
        tpv=pd.Series(c*v).groupby(day).cumsum().values
        vv=pd.Series(v).groupby(day).cumsum().values
        r=tpv/vv
    r=np.where(np.isnan(r),ema(p,smooth),r)
    return r

def impulses(d,volLen,volMult,atrLen,minBodyATR,minRangeATR):
    v=d["tick_volume"].values.astype(float)
    vb=sma(v,volLen)
    vi=v>vb*volMult
    a=atr(d["high"].values,d["low"].values,d["close"].values,atrLen)
    body=np.abs(d["close"].values-d["open"].values)
    rng=d["high"].values-d["low"].values
    ri=(body>=a*minBodyATR)&(rng>=a*minRangeATR)
    return vi&ri,vi,a

def zone_touch_stats(d,volLen,volMult,atrLen,minBodyATR,minRangeATR,maxZones,useWicks,rmode,rL,rS,horizon_bars):
    """For each impulse-created zone, find first touch by later price within horizon.
       Return reaction quality: avg subsequent move in zone direction over N bars after touch."""
    imp,vi,a=impulses(d,volLen,volMult,atrLen,minBodyATR,minRangeATR)
    r=regme(d,rmode,rL,rS)
    o=d["open"].values;h=d["high"].values;l=d["low"].values;c=d["close"].values
    n=len(d)
    bull=imp&(c>o)&(c>r)
    bear=imp&(c<o)&(c<r)
    zones=[]  # (dir,top,bottom,start)
    openD=[];openS=[]
    touches=[]
    for i in range(n):
        if bull[i]:
            top=min(o[i],c[i]) if useWicks else max(o[i],c[i]); bot=l[i]
            z=(1,top,bot,i); openD.append(z)
            if len(openD)>maxZones: openD.pop(0)
        if bear[i]:
            top=h[i]; bot=max(o[i],c[i]) if useWicks else min(o[i],c[i])
            z=(-1,top,bot,i); openS.append(z)
            if len(openS)>maxZones: openS.pop(0)
        # check touches on this bar for zones older than i
        for arr in (openD,openS):
            for z in list(arr):
                dd,tp,bt,st=z
                if i<=st: continue
                if bt==tp: continue
                touched = (l[i]<=tp and h[i]>=bt)
                broken = (c[i]<bt) if dd==1 else (c[i]>tp)
                if touched:
                    j=min(i+horizon_bars,n-1)
                    move=(c[j]-c[i])/a[i] if dd==1 else (c[i]-c[j])/a[i]
                    touches.append((dd,i,move,1 if broken else 0))
                    arr.remove(z)
                elif broken:
                    arr.remove(z)
    t=pd.DataFrame(touches,columns=["dir","i","move","broken"])
    if len(t)==0: return None
    good=(t.move>0.5).mean()
    return dict(touches=len(t),per_bar=len(t)/n,avg_move=t.move.mean(),
                win_rate_05atr=good,std=t.move.std())

def vdo_signals(d,vLen,sigLen,bandLen,bandMult,reqImpulse,volLen,volMult,useRegme,rmode,rL,rS,earlyCross,hz):
    v=d["tick_volume"].values.astype(float)
    c=d["close"].values;o=d["open"].values
    signed=np.where(c>np.roll(c,1),v,np.where(c<np.roll(c,1),-v,0.0));signed[0]=0
    tot=ema(v,vLen);pos=ema(signed,vLen)
    vzo=100*np.divide(pos,tot,out=np.zeros_like(pos),where=tot!=0)
    vdo=np.clip(50+vzo*0.5,0,100)
    sig=ema(vdo,sigLen)
    dev=stdev(vdo,bandLen)
    ub=np.minimum(80,sig+dev*bandMult); lb=np.maximum(20,sig-dev*bandMult)
    vi,_ ,a=impulses(d,volLen,volMult,14,0.0,0.0)
    r=regme(d,rmode,rL,rS)
    bullturn=(vdo[1:]<lb[:-1])&(vdo[1:]>vdo[:-1])&(vdo[1:-1]<=vdo[:-2])
    bearturn=(vdo[1:]>ub[:-1])&(vdo[1:]<vdo[:-1])&(vdo[1:-1]>=vdo[:-2])
    pad=lambda x: np.concatenate([[False],x])
    bullturn,bearturn=pad(bullturn),pad(bearturn)
    cok=(~reqImpulse)|vi
    bok=cok&(c>r) if useRegme else cok
    sok=cok&(c<r) if useRegme else cok
    bs=bullturn&bok; ss=bearturn&sok
    res=[]
    n=len(d)
    for arr,sgn,name in ((bs,1,"bull"),(ss,-1,"bear")):
        idx=np.where(arr)[0]
        mv=[]
        for i in idx:
            j=min(i+hz,n-1)
            mv.append((sgn*(c[j]-c[i]))/a[i])
        mv=np.array(mv)
        if len(mv):
            res.append(dict(name=name,count=len(idx),per_bar=len(idx)/n,avg=mv.mean(),win=(mv>0).mean(),win05=(mv>0.5).mean()))
    return res

if __name__=="__main__":
    cmd=sys.argv[1]
    if cmd=="zonesweep":
        tf=int(sys.argv[2]); d=load(tf)
        best=[]
        for volMult in [1.3,1.5,1.65,1.8,2.0]:
            for minB in [0.2,0.35,0.5]:
                for minR in [0.6,0.75,1.0]:
                    for volLen in [20,34,50]:
                        s=zone_touch_stats(d,volLen,volMult,14,minB,minR,24,True,"VWMA+EMA",55,8,12)
                        if s: best.append(((volLen,volMult,minB,minR),s))
        best.sort(key=lambda x:-x[1]["avg_move"])
        print(f"TF={tf}")
        for k,s in best[:12]: print(k,{kk:round(vv,3) for kk,vv in s.items()})
    elif cmd=="vdosweep":
        tf=int(sys.argv[2]); d=load(tf)
        best=[]
        for vLen in [10,14,20]:
          for bandMult in [1.0,1.35,1.7]:
            for reqI in [False,True]:
              for useR in [False,True]:
                r=vdo_signals(d,vLen,8,50,bandMult,reqI,34,1.65,useR,"VWMA+EMA",55,8,12)
                tot=sum(x["count"] for x in r)
                av=np.average([x["avg"] for x in r],weights=[x["count"] for x in r]) if tot else 0
                w=np.average([x["win05"] for x in r],weights=[x["count"] for x in r]) if tot else 0
                best.append(((vLen,bandMult,reqI,useR),tot,len(r) and av, w,[x["count"] for x in r]))
        best.sort(key=lambda x:-x[2] if x[2]==x[2] else -9)
        print(f"TF={tf}")
        for k,tot,av,w,cnt in best[:12]: print(k,"sig=",tot,"avgATR=",round(av,3),"win0.5=",round(w,3),cnt)
