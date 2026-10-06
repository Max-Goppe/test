#!/usr/bin/env python3
"""Compare scalp entry modes on GOLD data: Zone-touch-only vs VDO-only vs Combo, per TF (M1/M5/M15)."""
import sys; sys.path.insert(0, "/workspace")
import numpy as np
from backtest_scalp import load, resample, Engine, _MTF_CACHE

def build_signals(d5, tf_rule, cfg, confirm_rules, trend_rule, cooldown, touch_lookback=3, breakout_lookback=3):
    de = d5.copy() if tf_rule=="5min" else resample(d5, tf_rule)
    if "vol" not in de.columns: de["vol"]=de["tick_vol"]
    eng = Engine(de.open.values.astype(float),de.high.values.astype(float),de.low.values.astype(float),
                 de.close.values.astype(float),de["vol"].values.astype(float), **cfg)
    bull_side=(de.close.values>=eng.regme.values)
    impulse=eng.impulse.values; rngimp=eng.rangeimp.values
    hi=de.high.values; lo=de.low.values; cl=de.close.values; op=de.open.values; atrv=eng.atr.values
    vdo=eng.vdo.values; upb=eng.upb.values; lwb=eng.lwb.values; vsig=eng.vsig.values
    n=len(de)
    bull_imp=impulse&rngimp&(cl>op)&bull_side
    bear_imp=impulse&rngimp&(cl<op)&(~bull_side)
    d_top=np.where(bull_imp,np.minimum(op,cl),np.nan); d_bot=np.where(bull_imp,lo,np.nan)
    s_top=np.where(bear_imp,hi,np.nan); s_bot=np.where(bear_imp,np.maximum(op,cl),np.nan)
    def first_touch(top_a,bot_a,demand):
        idxs=np.where(~np.isnan(top_a))[0]; touched=np.zeros(n,int)-1
        for k in range(len(idxs)-1,-1,-1):
            i0=idxs[k]; top=top_a[i0]; bot=bot_a[i0]
            end=min((idxs[k-1] if k>0 else n)+20, n)
            seg_lo=lo[i0+1:end]; seg_hi=hi[i0+1:end]
            hit=np.nonzero((seg_lo<=top)&(seg_hi>=bot))[0] if demand else np.nonzero((seg_hi>=bot)&(seg_lo<=top))[0]
            if len(hit)>0: touched[i0+1+hit[0]]=i0+1+hit[0]
        return touched
    td=first_touch(d_top,d_bot,True); ts=first_touch(s_top,s_bot,False)
    for arr in (td,ts):
        last=-1
        for i in range(n):
            if arr[i]>0: last=arr[i]
            elif last>=0: arr[i]=last
    fd=np.array([(td[i]>=0 and i-td[i]<=touch_lookback and bull_side[i]) for i in range(n)])
    fs=np.array([(ts[i]>=0 and i-ts[i]<=touch_lookback and not bull_side[i]) for i in range(n)])
    cross_up=(vdo>upb)&(np.roll(vdo,1)<=np.roll(upb,1)); cross_dn=(vdo<lwb)&(np.roll(vdo,1)>=np.roll(lwb,1))
    cross_up[0]=False; cross_dn[0]=False
    bu=np.zeros(n,bool); bd=np.zeros(n,bool); lb=-999
    for i in range(n):
        if cross_up[i]: lb=i
        bu[i]=(i-lb)<=breakout_lookback
    lb=-999
    for i in range(n):
        if cross_dn[i]: lb=i
        bd[i]=(i-lb)<=breakout_lookback
    # VDO extreme os mode too
    os_bull=np.zeros(n,bool); os_bear=np.zeros(n,bool)
    lb=-999
    for i in range(n):
        if vdo[i]<30: lb=i
        os_bull[i]=(i-lb)<=breakout_lookback
    lb=-999
    for i in range(n):
        if vdo[i]>70: lb=i
        os_bear[i]=(i-lb)<=breakout_lookback
    vdo_conf = bu|os_bull  # bullish VDO confirmation
    vdo_conf_s = bd|os_bear
    def mtf_sides(rule,mult):
        key=(rule,round(mult,2),n)
        if key in _MTF_CACHE: return _MTF_CACHE[key]
        dd=resample(d5,rule)
        e=Engine(dd.open.values.astype(float),dd.high.values.astype(float),dd.low.values.astype(float),
                 dd.close.values.astype(float),dd["vol"].values.astype(float),
                 regme_len=55,smooth=8,atr_n=14,volbase=50,volmult=mult,minbody=0.55)
        sides=(dd.close.values>=e.regme.values)
        idx=np.searchsorted(dd["time"].values,de["time"].values,side="right")-1
        out=np.where(idx>=0,sides[np.clip(idx,0,len(sides)-1)],False)
        _MTF_CACHE[key]=out; return out
    conf=[mtf_sides(r,2.0) for r in confirm_rules]; tr=mtf_sides(trend_rule,2.0)
    acb=np.ones(n,bool); abe=np.ones(n,bool)
    for s in conf: acb&=s; abe&=~s
    bull_ok=bull_side&acb&tr; bear_ok=(~bull_side)&abe&(~tr)
    return dict(n=n,bull_ok=bull_ok,bear_ok=bear_ok,fd=fd,fs=fs,vdo_conf=vdo_conf,vdo_conf_s=vdo_conf_s,
                cl=cl,hi=hi,lo=lo,atrv=atrv,d_bot=d_bot,s_top=s_top,de=de)

def simulate(sig, mode, tp_r=2.0, sl_atr_mult=1.5, cooldown=6):
    bull_ok=sig["bull_ok"]; bear_ok=sig["bear_ok"]; fd=sig["fd"]; fs=sig["fs"]
    vc=sig["vdo_conf"]; vcs=sig["vdo_conf_s"]; cl=sig["cl"]; hi=sig["hi"]; lo=sig["lo"]
    atrv=sig["atrv"]; d_bot=sig["d_bot"]; s_top=sig["s_top"]; n=sig["n"]
    if mode=="zone": b_sig=fd; s_sig=fs
    elif mode=="vdo": b_sig=vc; s_sig=vcs
    elif mode=="combo": b_sig=fd&vc; s_sig=fs&vcs
    trades=[]; last=-999
    for i in range(20,n-30):
        if np.isnan(atrv[i]): continue
        st=0
        if bull_ok[i] and b_sig[i] and (i-last)>=cooldown: st=1
        elif bear_ok[i] and s_sig[i] and (i-last)>=cooldown: st=-1
        if st==0: continue
        last=i; entry=cl[i]; a=atrv[i]
        if st>0:
            stop=entry-sl_atr_mult*a
            cand=np.nonzero((~np.isnan(d_bot[max(0,i-40):i])) & (d_bot[max(0,i-40):i]<entry))[0]
            if len(cand)>0: stop=d_bot[i-1-cand[-1]]-0.15*a
            risk=entry-stop
            if risk<=0 or np.isnan(risk): continue
            tp=entry+tp_r*risk
            for j in range(i+1,min(i+240,n)):
                if lo[j]<=stop: trades.append((-risk,a)); break
                if hi[j]>=tp: trades.append((tp_r*risk,a)); break
        else:
            stop=entry+sl_atr_mult*a
            cand=np.nonzero((~np.isnan(s_top[max(0,i-40):i])) & (s_top[max(0,i-40):i]>entry))[0]
            if len(cand)>0: stop=s_top[i-1-cand[-1]]+0.15*a
            risk=stop-entry
            if risk<=0 or np.isnan(risk): continue
            tp=entry-tp_r*risk
            for j in range(i+1,min(i+240,n)):
                if hi[j]>=stop: trades.append((-risk,a)); break
                if lo[j]<=tp: trades.append((tp_r*risk,a)); break
    if len(trades)<5: return None
    t=np.array(trades); pnl=t[:,0]/t[:,1]
    wins=(pnl>0).sum(); gp=pnl[pnl>0].sum(); gl=-pnl[pnl<0].sum()
    return dict(n=len(t),wr=100*wins/len(t),mean=pnl.mean(),pf=(gp/gl if gl>0 else 99))

if __name__=="__main__":
    d5=load()
    # resample M1 from... we only have M5 data; approximate M1 entry with M5 grid shifted? Use M5 as fastest real TF.
    grids={
      "M5":("5min",["15min"],"60min",dict(regme_len=34,smooth=8,atr_n=14,volbase=50,volmult=2.0,minbody=0.5,vdo_bandlen=50)),
      "M15":("15min",["60min"],"240min",dict(regme_len=55,smooth=8,atr_n=14,volbase=50,volmult=2.0,minbody=0.5,vdo_bandlen=24)),
    }
    print("mode comparison (TP=2R, SL zone-edge/1.5ATR fallback, all-TF aligned):")
    for tf,(rule,confs,trend,cfg) in grids.items():
        sig=build_signals(d5,rule,cfg,confs,trend,cooldown=6)
        for mode in ("zone","vdo","combo"):
            r=simulate(sig,mode)
            if r: print(f"  {tf:>4} {mode:<6} n={r['n']:>4} wr={r['wr']:.1f}% mean={r['mean']:+.3f} ATR pf={r['pf']:.2f}")
            else: print(f"  {tf:>4} {mode:<6} too few trades")
    # combo vs zone also at TP=1.5R and TP=3R
    print("\nsensitivity (M5):")
    rule,confs,trend,cfg=grids["M5"]
    sig=build_signals(d5,rule,cfg,confs,trend,cooldown=6)
    for mode in ("zone","combo"):
        for tp in (1.5,2.0,3.0):
            r=simulate(sig,mode,tp_r=tp)
            if r: print(f"  M5 {mode:<6} tp={tp} n={r['n']:>4} wr={r['wr']:.1f}% mean={r['mean']:+.3f} pf={r['pf']:.2f}")
