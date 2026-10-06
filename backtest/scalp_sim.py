import csv, glob, math

def load():
    rows=[]
    for f in sorted(glob.glob('/workspace/2022-2026/GOLD_M5_*.csv')):
        with open(f) as fh:
            for r in csv.reader(fh):
                if len(r)<6 or not r[0][:4].isdigit(): continue
                try:
                    o,h,l,c,v=float(r[1]),float(r[2]),float(r[3]),float(r[4]),float(r[5])
                except: continue
                rows.append((r[0],o,h,l,c,v))
    return rows

rows=load()
n=len(rows)
t=[i for i in range(n)]
O=[rows[i][1] for i in t]; H=[rows[i][2] for i in t]; L=[rows[i][3] for i in t]
C=[rows[i][4] for i in t]; V=[rows[i][5] for i in t]

def ema(x,p):
    k=2/(p+1); out=[]; e=x[0]
    for v in x: e=v*k+e*(1-k); out.append(e)
    return out

def sma(x,p):
    out=[]; s=0; buf=[]
    for i,v in enumerate(x):
        buf.append(v); s+=v
        if i>=p: s-=buf.pop(0)
        out.append(s/min(i+1,p))
    return out

def atr(p=14):
    tr=[abs(H[0]-L[0])]+[max(H[i]-L[i],abs(H[i]-C[i-1]),abs(L[i]-C[i-1])) for i in range(1,n)]
    return ema(tr,p)

# VWMA hlc3
hlc3=[(H[i]+L[i]+C[i])/3 for i in range(n)]
def vwma(x,vols,p):
    out=[]; num=0; den=0
    # simple O(n*p) is too slow; use running sums
    import collections
    qn=collections.deque(); qd=collections.deque(); sn=0.0; sd=0.0
    for i in range(len(x)):
        a=x[i]*vols[i]; b=vols[i]
        qn.append(a); qd.append(b); sn+=a; sd+=b
        if len(qn)>p: sn-=qn.popleft(); sd-=qd.popleft()
        out.append(sn/sd if sd!=0 else x[i])
    return out

def resample(step):
    # aggregate M5 bars into step-M5 blocks
    idx=list(range(0,n,step))
    ro=[O[i] for i in idx]
    rh=[max(H[i:i+step]) if i+step<=n else max(H[i:]) for i in idx]
    rl=[min(L[i:i+step]) if i+step<=n else min(L[i:]) for i in idx]
    rc=[C[min(i+step-1,n-1)] for i in idx]
    rv=[sum(V[i:i+step]) for i in idx]
    return idx,ro,rh,rl,rc,rv

def side_series(idx,ro,rh,rl,rc,rv,regLen,smooth):
    vw=vwma([(rh[k]+rl[k]+rc[k])/3 for k in range(len(rc))],rv,regLen)
    reg=ema(vw,smooth)
    bull=[rc[k]>=reg[k] for k in range(len(rc))]
    return bull,idx,reg

def map_to_main(bull_big,idx):
    # for each main bar i, state of last completed big bar strictly before i
    out=[False]*n
    j=0
    for i in range(n):
        while j<len(idx) and idx[j]<i:
            out[i]=bull_big[j]; j+=1
        # note: idx[j] is start of big bar; use last big bar whose START < i (may be forming -> lookahead risk)
    return out

def map_completed(bull_big,idx):
    out=[False]*n
    bj=-1
    for i in range(n):
        # big bar k covers [idx[k], idx[k]+step-1]; completed when i > idx[k]+step-1
        step = idx[1]-idx[0] if len(idx)>1 else 9
        while bj+1<len(idx) and i > idx[bj+1]+step-1:
            bj+=1
        out[i]= bull_big[bj] if bj>=0 else False
    return out

def run(entry_step, cfgs, signal_mode='combo', require_all=True, tp_r=1.5, sl_atr=None, cooldown=6, touch_lb=3, vol_mult_entry=1.6, body_min=0.75, zone_max=16, impulse_body=True):
    regLen, smooth = cfgs
    A=atr(14)
    vwE=vwma(hlc3,V,regLen); regE=ema(vwE,smooth)
    above=[C[i]>=regE[i] for i in range(n)]
    vb=sma(V,50)
    # confirm/trend series
    idxc,roc,rhc,rlc,rcc,rvc=resample(3)
    bull_c_raw,idxc,_=side_series(idxc,roc,rhc,rlc,rcc,rvc,regLen,smooth)
    conf_bull=map_completed(bull_c_raw,idxc)
    idxt,rot,rht,rlt,rt_,rvt=resample(12)
    bull_t_raw,idxt,_=side_series(idxt,rot,rht,rlt,rt_,rvt,regLen,smooth)
    trend_bull=map_completed(bull_t_raw,idxt)

    zones_d=[]; zones_s=[]  # list of dicts top,bottom,birth,touched
    def new_bar_zone(i):
        imp = vb[i]>0 and V[i]>vb[i]*vol_mult_entry
        body=abs(C[i]-O[i])
        rng=H[i]-L[i]
        ok = (body>=A[i]*body_min and rng>=A[i]*0.9) if impulse_body else True
        if imp and ok and C[i]>O[i] and above[i]:
            zones_d.append({'top':min(O[i],C[i]),'bot':L[i],'birth':i,'t':False})
            if len(zones_d)>zone_max: zones_d.pop(0)
        if imp and ok and C[i]<O[i] and not above[i]:
            zones_s.append({'top':H[i],'bot':max(O[i],C[i]),'birth':i,'t':False})
            if len(zones_s)>zone_max: zones_s.pop(0)

    trades=[]
    last_sig=-999
    pending=None
    for i in range(60,n):
        new_bar_zone(i)
        # exits first
        if pending is not None:
            side,entry,sl,tp,eb=pending
            hit=None
            if side=='buy':
                if L[i]<=sl: hit=-tp_r*0-1.0*(entry-sl)/ (tp-entry+1e-9) *0 
            # simpler: check SL then TP by price distance
            if side=='buy':
                if L[i]<=sl: trades.append(('buy',-1.0,i-eb)); pending=None
                elif H[i]>=tp: trades.append(('buy',1.0,i-eb)); pending=None
            else:
                if H[i]>=sl: trades.append(('sell',-1.0,i-eb)); pending=None
                elif L[i]<=tp: trades.append(('sell',1.0,i-eb)); pending=None
        mtfB=(1 if above[i] else 0)+(1 if conf_bull[i] else 0)+(1 if trend_bull[i] else 0)
        mtfS=(0 if above[i] else 1)+(0 if conf_bull[i] else 1)+(0 if trend_bull[i] else 1)
        bull_ok = mtfB==3 if require_all else mtfB>=2
        bear_ok = mtfS==3 if require_all else mtfS>=2
        # fresh touches this bar
        td=False; ts=False; zD=None; zS=None
        for z in zones_d:
            if not z['t'] and i>z['birth'] and L[i]<=z['top'] and H[i]>=z['bot']:
                z['t']=True; td=True; zD=z
        for z in zones_s:
            if not z['t'] and i>z['birth'] and H[i]>=z['bot'] and L[i]<=z['top']:
                z['t']=True; ts=True; zS=z
        # VDO
        sv=[0.0]*n
        for k in range(1,n):
            sv[k]=V[k] if C[k]>C[k-1] else (-V[k] if C[k]<C[k-1] else 0.0)
        # compute vdo lazily only here would be slow; precompute outside instead
        buySig=False; sellSig=False
        if bull_ok and above[i] and td and pending is None and i-last_sig>cooldown:
            buySig=True
        if bear_ok and (not above[i]) and ts and pending is None and i-last_sig>cooldown:
            sellSig=True
        if buySig or sellSig:
            last_sig=i
            if buySig:
                entry=C[i]; slp=zD['bot']-0.15*A[i]; risk=entry-slp
                tp=entry+tp_r*risk; pending=('buy',entry,slp,tp,i)
            else:
                entry=C[i]; slp=zS['top']+0.15*A[i]; risk=slp-entry
                tp=entry-tp_r*risk; pending=('sell',entry,slp,tp,i)
    wins=sum(1 for s,r,a in trades if r>0)
    tot=len(trades)
    avg=sum(r for s,r,a in trades)/tot if tot else 0
    pf_num=sum(1 for s,r,a in trades if r>0); pf_den=max(1,(tot-pf_num))
    return tot,wins,tot and wins/tot or 0,avg,[ (i,a) for s,r,a in trades if r>0][:0]

# NOTE: the loop above recomputes sv inside loop (slow). Rewrite properly below.
