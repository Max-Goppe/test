import csv, glob, math, itertools, statistics

def load():
    rows = []
    for f in sorted(glob.glob('/workspace/2022-2026/GOLD_M5_*.csv')):
        raw=open(f,'rb').read()
        txt=raw.decode('utf-16',errors='replace') if raw[:2] in (b'\xff\xfe',b'\xfe\xff') else raw.decode('utf-8',errors='replace')
        for line in txt.splitlines():
            p = line.strip().split(',')
            if len(p) < 7: continue
            try:
                dt, o, h, l, c, tick, vol = p[0], float(p[1]), float(p[2]), float(p[3]), float(p[4]), int(p[5]), int(p[6])
            except ValueError:
                continue
            rows.append((dt, o, h, l, c, tick, vol))
    return rows

M5 = load()
print("M5 bars:", len(M5))

def resample(rows, n):
    out = []
    for i in range(0, len(rows) - n + 1, n):
        chunk = rows[i:i+n]
        out.append((chunk[0][0], chunk[0][1], max(r[2] for r in chunk), min(r[3] for r in chunk),
                    chunk[-1][4], sum(r[5] for r in chunk), sum(r[6] for r in chunk)))
    return out

TF = {'M15': resample(M5, 3), 'H1': resample(M5, 12)}
for k, v in TF.items(): print(k, "bars:", len(v))

def ema(vals, n):
    k = 2.0/(n+1); out=[]; e=None
    for v in vals:
        e = v if e is None else v*k + e*(1-k)
        out.append(e)
    return out

def sma_c(vals, n):
    out=[]; s=0.0
    for i,v in enumerate(vals):
        s += v
        if i >= n: s -= vals[i-n]
        out.append(s/min(i+1,n))
    return out

def stdev_c(vals, n):
    out=[]
    for i in range(len(vals)):
        w = vals[max(0,i-n+1):i+1]
        m = sum(w)/len(w)
        out.append(math.sqrt(sum((x-m)**2 for x in w)/len(w)))
    return out

def atr_c(h,l,c,n=14):
    trs=[]
    for i in range(len(c)):
        tr = h[i]-l[i] if i==0 else max(h[i]-l[i], abs(h[i]-c[i-1]), abs(l[i]-c[i-1]))
        trs.append(tr)
    return ema(trs, n)

def vwma(h,l,c,vol,n):
    hlc=[(h[i]+l[i]+c[i])/3.0 for i in range(len(c))]
    num=[hlc[i]*max(vol[i],0) for i in range(len(c))]
    sn=sma_c(num,n); sd=sma_c([float(max(v,0)) for v in vol],n)
    return [sn[i]/sd[i] if sd[i]>0 else hlc[i] for i in range(len(c))]

def run_tf(rows, tfname, cfg):
    # cfg: regme_len, smooth, volbase, volmult, bodyAtr, rangeAtr, vdoLen, vdoSig, bandLen, bandMult
    o=[r[1] for r in rows]; h=[r[2] for r in rows]; l=[r[3] for r in rows]; c=[r[4] for r in rows]
    vol=[r[6] for r in rows]
    atr=atr_c(h,l,c,14)
    volBase=sma_c([float(v) for v in vol], cfg['volbase'])
    vw=vwma(h,l,c,vol,cfg['regmeLen'])
    regme=ema(vw, cfg['smooth'])
    N=len(rows)
    impulses=[]  # (idx, side, top, bottom, touched)
    lastBuyTouch=-999; lastSellTouch=-999
    zonesD=[]; zonesS=[]
    lastDz=-9999; lastSz=-9999
    signed=[0.0]*N; tot=[0.0]*N; posv=[0.0]*N
    for i in range(N):
        if i>0:
            signed[i]= vol[i] if c[i]>c[i-1] else (-vol[i] if c[i]<c[i-1] else 0.0)
    totE=ema([float(v) for v in vol], cfg['vdoLen'])
    posE=ema(signed, cfg['vdoLen'])
    vdo=[max(0.0,min(100.0, 50.0+0.5*(posE[i]/totE[i]*100.0))) if totE[i]>0 else 50.0 for i in range(N)]
    vdoSig=ema(vdo, cfg['vdoSig'])
    vdoDev=stdev_c(vdo, cfg['bandLen'])
    upper=[min(80.0, vdoSig[i]+vdoDev[i]*cfg['bandMult']) for i in range(N)]
    lower=[max(20.0, vdoSig[i]-vdoDev[i]*cfg['bandMult']) for i in range(N)]
    bullBrk=[False]*N
    bearBrk=[False]*N
    for i in range(1,N):
        if vdo[i]>upper[i] and vdo[i-1]<=upper[i-1]: bullBrk[i]=True
        if vdo[i]<lower[i] and vdo[i-1]>=lower[i-1]: bearBrk[i]=True
    bullBrkAge=[9999]*N; bearBrkAge=[9999]*N
    lb=-9999
    for i in range(N):
        if bullBrk[i]: lb=i
        bullBrkAge[i]=i-lb
    lb=-9999
    for i in range(N):
        if bearBrk[i]: lb=i
        bearBrkAge[i]=i-lb
    # Precompute first-touch bar for every zone (two-pass, linear-ish).
    demandZ=[]; supplyZ=[]
    def mkz(i,top,bot): return {'birth':i,'top':top,'bot':bot,'touch':None}
    # We must replay zone creation identically: do it here in a separate pass.
    zD=[]; zS=[]
    for i in range(20,N):
        ib=(volBase[i]>0 and vol[i]>volBase[i]*cfg['volmult'] and atr[i]>0
            and abs(c[i]-o[i])>=atr[i]*cfg['bodyAtr'] and (h[i]-l[i])>=atr[i]*cfg['rangeAtr'])
        if ib and c[i]>o[i] and c[i]>regme[i]:
            zD.append(mkz(i,min(o[i],c[i]),l[i]))
            if len(zD)>24: zD.pop(0)
        if ib and c[i]<o[i] and c[i]<regme[i]:
            zS.append(mkz(i,h[i],max(o[i],c[i])))
            if len(zS)>24: zS.pop(0)
    # first touches: iterate bars, check zones not yet touched/broken — but now record into arrays of touch events
    touchDbar=[-9999]*N; touchSbar=[-9999]*N
    liveD=list(range(len(zD))); liveS=list(range(len(zS)))
    # need zone births to be processed as they happen; do chronological sweep
    liveD=[]; liveS=[]
    zdi=0; zsi=0
    zdq=[]; zsq=[]  # lists of active zone dicts
    for i in range(20,N):
        ib=(volBase[i]>0 and vol[i]>volBase[i]*cfg['volmult'] and atr[i]>0
            and abs(c[i]-o[i])>=atr[i]*cfg['bodyAtr'] and (h[i]-l[i])>=atr[i]*cfg['rangeAtr'])
        if ib and c[i]>o[i] and c[i]>regme[i]:
            zdq.append({'birth':i,'top':min(o[i],c[i]),'bot':l[i],'touched':False})
            if len(zdq)>24: zdq.pop(0)
        if ib and c[i]<o[i] and c[i]<regme[i]:
            zsq.append({'birth':i,'top':h[i],'bot':max(o[i],c[i]),'touched':False})
            if len(zsq)>24: zsq.pop(0)
        for z in zdq:
            if z['birth']<i and not z['touched']:
                if c[i]<z['bot']: z['touched']=True
                elif l[i]<=z['top'] and h[i]>=z['bot']:
                    z['touched']=True; touchDbar[i]=i
        for z in zsq:
            if z['birth']<i and not z['touched']:
                if c[i]>z['top']: z['touched']=True
                elif h[i]>=z['bot'] and l[i]<=z['top']:
                    z['touched']=True; touchSbar[i]=i
    # running "last touch" array
    ltD=-9999; ltS=-9999
    lastTouchD=[-9999]*N; lastTouchS=[-9999]*N
    for i in range(N):
        if touchDbar[i]==i: ltD=i
        if touchSbar[i]==i: ltS=i
        lastTouchD[i]=ltD; lastTouchS[i]=ltS
    signals=[]
    lastSig=-999
    for i in range(20, N):
        above = c[i] >= regme[i]
        impulseBull = (volBase[i]>0 and vol[i]>volBase[i]*cfg['volmult'] and atr[i]>0
                       and abs(c[i]-o[i])>=atr[i]*cfg['bodyAtr'] and (h[i]-l[i])>=atr[i]*cfg['rangeAtr']
                       and c[i]>o[i] and c[i]>regme[i])
        impulseBear = (volBase[i]>0 and vol[i]>volBase[i]*cfg['volmult'] and atr[i]>0
                       and abs(c[i]-o[i])>=atr[i]*cfg['bodyAtr'] and (h[i]-l[i])>=atr[i]*cfg['rangeAtr']
                       and c[i]<o[i] and c[i]<regme[i])
        if impulseBull: lastDz=i
        if impulseBear: lastSz=i
        touchD = touchDbar[i]==i
        touchS = touchSbar[i]==i
        lastBuyTouch=lastTouchD[i]; lastSellTouch=lastTouchS[i]
        impAgeB = 0
        recentImpB = (i-lastDz)<=cfg['impulseAge'] and above
        recentImpS = (i-lastSz)<=cfg['impulseAge'] and not above
        freshTouchB = (i-lastBuyTouch)<=cfg['touchLook'] and above
        freshTouchS = (i-lastSellTouch)<=cfg['touchLook'] and not above
        vdoB = bullBrkAge[i]<=cfg['brkLook'] and vdo[i]>vdoSig[i]
        vdoS = bearBrkAge[i]<=cfg['brkLook'] and vdo[i]<vdoSig[i]
        scoreB = (1 if recentImpB else 0)+(1 if freshTouchB else 0)+(1 if vdoB else 0)
        scoreS = (1 if recentImpS else 0)+(1 if freshTouchS else 0)+(1 if vdoS else 0)
        if i-lastSig <= cfg['cooldown']: continue
        if above and scoreB>=cfg['minConf']:
            signals.append((i,'BUY')); lastSig=i
        elif (not above) and scoreS>=cfg['minConf']:
            signals.append((i,'SELL')); lastSig=i
    return rows, signals, atr, c, regme

def evaluate(rows, signals, atr, horizon):
    c=[r[4] for r in rows]
    trades=[]
    for idx,(i,side) in enumerate(signals):
        j=min(i+horizon, len(c)-1)
        if j<=i: continue
        entry=c[i]; exit_=c[j]
        a=atr[i] if atr[i]>0 else 1e-9
        pnl=(exit_-entry)/a if side=='BUY' else (entry-exit_)/a
        trades.append(pnl)
    if not trades: return None
    wins=sum(1 for t in trades if t>0)
    return dict(n=len(trades), winrate=wins/len(trades)*100, avg=statistics.mean(trades),
               med=statistics.median(trades), tot=sum(trades))



def base(**kw):
    cfg=dict(regmeLen=55,smooth=8,volbase=50,volmult=2.2,bodyAtr=0.70,rangeAtr=1.10,
             vdoLen=14,vdoSig=8,bandLen=50,bandMult=1.35,
             minConf=2,touchLook=3,brkLook=3,impulseAge=15,cooldown=10)
    cfg.update(kw); return cfg

HORIZONS=(6,12,24)

def score(cfg, rows):
    _,sigs,atr,c,_=run_tf(rows,'X',cfg)
    res={}
    for hor in HORIZONS:
        res[hor]=evaluate(rows,sigs,atr,hor)
    if not all(res[h] and res[h]['n']>=30 for h in HORIZONS): return None
    avgs=[res[h]['avg'] for h in HORIZONS]
    return (statistics.mean(avgs), res[24]['winrate'], min(res[h]['n'] for h in HORIZONS), res)

def detail(cfg,label,rows):
    r=score(cfg,rows)
    if not r: print(label,"-> too few signals"); return None
    s,wr,n,res=r
    line=f"{label}: obj={s:+.3f} | "+" | ".join(f"h{h}: avg={res[h]['avg']:+.3f} win={res[h]['winrate']:.0f}% n={res[h]['n']}" for h in HORIZONS)
    print(line); return s

import sys
rows_h1=TF['H1']; rows_m15=TF['M15']
print("== sanity ==")
detail(base(),"DEFAULT @H1",rows_h1)

# Stage 1: signal rules on H1
best=[]
for mc in (2,3):
    for tl in (1,2,3,5):
        for bl in (1,2,3,5):
            for ia in (5,8,15,25,40):
                for cd in (5,10,20,40):
                    v=dict(minConf=mc,touchLook=tl,brkLook=bl,impulseAge=ia,cooldown=cd)
                    r=score(base(**v),rows_h1)
                    if r: best.append((r[0],r[1],r[2],v))
best.sort(key=lambda x:-x[0])
print("\n=== STAGE 1 top rules ===")
for s,wr,n,v in best[:8]:
    print(f"obj={s:+.3f} win@24={wr:.0f}% n={n} | conf={v['minConf']} touch={v['touchLook']} brk={v['brkLook']} age={v['impulseAge']} cd={v['cooldown']}")

rules=best[0][3]

# Stage 2: core params
best2=[]
for vm in (1.5,1.8,2.2,2.8,3.5):
    for ba in (0.5,0.7,0.9):
        for ra in (0.8,1.1,1.4):
            for rl in (34,55,89):
                cfg=base(**rules); cfg['volmult']=vm; cfg['bodyAtr']=ba; cfg['rangeAtr']=ra; cfg['regmeLen']=rl
                r=score(cfg,rows_h1)
                if r: best2.append((r[0],r[1],r[2],dict(volmult=vm,bodyAtr=ba,rangeAtr=ra,regmeLen=rl)))
best2.sort(key=lambda x:-x[0])
print("\n=== STAGE 2 top core params ===")
for s,wr,n,v in best2[:8]:
    print(f"obj={s:+.3f} win@24={wr:.0f}% n={n} | volm={v['volmult']} body={v['bodyAtr']} rng={v['rangeAtr']} regmeL={v['regmeLen']}")

finalC=dict(rules); finalC.update(best2[0][3])
print("\nFINAL:",finalC)
detail(base(**finalC),"FINAL @H1",rows_h1)
detail(base(**finalC),"FINAL @M15",rows_m15)

# yearly stability of FINAL on H1
_,sigs,atr,c,_=run_tf(rows_h1,'X',base(**finalC))
years={}
for i,side in sigs:
    y=rows_h1[i][0][:4]
    j=min(i+24,len(c)-1)
    a=atr[i] if atr[i]>0 else 1e-9
    pnl=(c[j]-c[i])/a if side=='BUY' else (c[i]-c[j])/a
    years.setdefault(y,[]).append(pnl)
print("\nyearly avg ATR (h24):")
for y in sorted(years):
    v=years[y]; print(f"  {y}: n={len(v)} avg={statistics.mean(v):+.3f} win={sum(1 for x in v if x>0)/len(v)*100:.0f}%")

# buy&hold reference per year for scale
print("\nB&H yearly move (ATR units of H1):")
by={}
for i,r_ in enumerate(rows_h1):
    by.setdefault(r_[0][:4],[]).append(r_)
for y in sorted(by):
    rr=by[y]; tot=sum(rr[k][2]-rr[k][3] for k in range(len(rr)))
    pass
