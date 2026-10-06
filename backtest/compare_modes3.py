#!/usr/bin/env python3
"""Relaxed-touch comparison: Zone touch (relaxed) vs VDO-only vs Combo, matching Pine logic."""
import sys; sys.path.insert(0,"/workspace")
import numpy as np
from backtest_scalp import load, resample, Engine, _MTF_CACHE

def build(d5, tf_rule, cfg, confirm_rules, trend_rule, touch_lookback=6, zone_age_max=40, breakout_lookback=3):
    de = d5.copy() if tf_rule=="5min" else resample(d5, tf_rule)
    if "vol" not in de.columns: de["vol"]=de["tick_vol"]
    eng = Engine(de.open.values.astype(float),de.high.values.astype(float),de.low.values.astype(float),
                 de.close.values.astype(float),de["vol"].values.astype(float), **cfg)
    bull_side=(de.close.values>=eng.regme.values)
    impulse=eng.impulse.values; rngimp=eng.rangeimp.values
    hi=de.high.values; lo=de.low.values; cl=de.close.values; op=de.open.values; atrv=eng.atr.values
    vdo=eng.vdo.values; upb=eng.upb.values; lwb=eng.lwb.values; vsig=eng.vsig.values
    n=len(de)
    # --- live zone simulation identical to Pine arrays (keep last 16 per side) ---
    K=16
    d_top=np.full(n,np.nan); d_bot=np.full(n,np.nan)   # most-recent TOUCHED demand info at bar i
    s_top=np.full(n,np.nan); s_bot=np.full(n,np.nan)
    last_d_touch=-999; last_s_touch=-999
    D=[]; S=[]  # list of [top,bottom,birth,touched]
    for i in range(n):
        if impulse[i] and rngimp[i] and cl[i]>op[i] and bull_side[i]:
            D.append([min(op[i],cl[i]), lo[i], i, False])
            if len(D)>K: D.pop(0)
        if impulse[i] and rngimp[i] and cl[i]<op[i] and not bull_side[i]:
            S.append([hi[i], max(op[i],cl[i]), i, False])
            if len(S)>K: S.pop(0)
        for z in D:
            top,bot,birth,touched=z
            if touched or i<=birth: continue
            if i-birth>zone_age_max: continue
            if lo[i]<=top and hi[i]>=bot:
                z[3]=True; last_d_touch=i
        for z in S:
            top,bot,birth,touched=z
            if touched or i<=birth: continue
            if i-birth>zone_age_max: continue
            if hi[i]>=bot and lo[i]<=top:
                z[3]=True; last_s_touch=i
        d_top[i]=last_d_touch
    fd=np.array([(i-d_top[i]<=touch_lookback and i-d_top[i]>=0 and d_top[i]>=0) for i in range(n)])
    fs=np.array([(i-last_s_touch<=touch_lookback and i-last_s_touch>=0 and last_s_touch>=0) for _ in [0] for i in [0]])  # placeholder
    # recompute fs properly
    ts_arr=np.zeros(n,int)-1
    last=-999
    # redo supply pass quickly using stored events
    # simpler: rerun loop storing both
    fd2=np.zeros(n,bool); fs2=np.zeros(n,bool)
    D=[]; S=[]; ldt=-999; lst=-999
    for i in range(n):
        if impulse[i] and rngimp[i] and cl[i]>op[i] and bull_side[i]:
            D.append([min(op[i],cl[i]), lo[i], i, False])
            if len(D)>K: D.pop(0)
        if impulse[i] and rngimp[i] and cl[i]<op[i] and not bull_side[i]:
            S.append([hi[i], max(op[i],cl[i]), i, False])
            if len(S)>K: S.pop(0)
        for z in D:
            if z[3] or i<=z[2] or i-z[2]>zone_age_max: continue
            if lo[i]<=z[0] and hi[i]>=z[1]: z[3]=True; ldt=i
        for z in S:
            if z[3] or i<=z[2] or i-z[2]>zone_age_max: continue
            if hi[i]>=z[1] and lo[i]<=z[0]: z[3]=True; lst=i
        fd2[i]= ldt>=0 and (i-ldt)<=touch_lookback
        fs2[i]= lst>=0 and (i-lst)<=touch_lookback
    fd=fd2; fs=fs2
    # VDO confirmation states (fresh breakout OR extreme OR above/below signal line - like Pine comboOk)
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
    os_bull=np.zeros(n,bool); os_bear=np.zeros(n,bool); lb=-999
    for i in range(n):
        if vdo[i]<30: lb=i
        os_bull[i]=(i-lb)<=breakout_lookback
    lb=-999
    for i in range(n):
        if vdo[i]>70: lb=i
        os_bear[i]=(i-lb)<=breakout_lookback
    vdo_b_ok = bu|os_bull|(vdo>vsig)      # Pine buyComboOk
    vdo_s_ok = bd|os_bear|(vdo<vsig)
    vdo_strong_b = bu|os_bull             # VDO-only trigger (no side-of-signal-line)
    vdo_strong_s = bd|os_bear
    def mtf(rule,mult):
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
    conf=[mtf(r,2.0) for r in confirm_rules]; tr=mtf(trend_rule,2.0)
    acb=np.ones(n,bool); abe=np.ones(n,bool)
    for s in conf: acb&=s; abe&=~s
    bull_ok=bull_side&acb&tr; bear_ok=(~bull_side)&abe&(~tr)
    # nearest zone edges for SL (demand bottom born recently & below close)
    return dict(n=n,bull_ok=bull_ok,bear_ok=bear_ok,fd=fd,fs=fs,vdo_b_ok=vdo_b_ok,vdo_s_ok=vdo_s_ok,
                vdo_strong_b=vdo_strong_b,vdo_strong_s=vdo_strong_s,cl=cl,hi=hi,lo=lo,atrv=atrv,de=de,
                D=None,S=None)

def sim_zones(d5, tf_rule, cfg, confirm_rules, trend_rule, mode, cooldown=6, tp_r=2.0, sl_atr_mult=1.5, horizon=24, touch_lookback=6, zone_age_max=40):
    """Simulate with live zone list so we can get SL from the touched zone edge."""
    de = d5.copy() if tf_rule=="5min" else resample(d5, tf_rule)
    if "vol" not in de.columns: de["vol"]=de["tick_vol"]
    eng = Engine(de.open.values.astype(float),de.high.values.astype(float),de.low.values.astype(float),
                 de.close.values.astype(float),de["vol"].values.astype(float), **cfg)
    bull_side=(de.close.values>=eng.regme.values)
    impulse=eng.impulse.values; rngimp=eng.rangeimp.values
    hi=de.high.values; lo=de.low.values; cl=de.close.values; op=de.open.values; atrv=eng.atr.values
    vdo=eng.vdo.values; upb=eng.upb.values; lwb=eng.lwb.values; vsig=eng.vsig.values
    n=len(de); K=16
    sig=build(d5,tf_rule,cfg,confirm_rules,trend_rule,touch_lookback,zone_age_max)
    bull_ok=sig["bull_ok"]; bear_ok=sig["bear_ok"]; fd=sig["fd"]; fs=sig["fs"]
    D=[]; S=[]; ldt=-999; lst=-999; last_d_edge=np.nan; last_s_edge=np.nan
    trades=[]; last_sig=-999
    for i in range(n):
        if impulse[i] and rngimp[i] and cl[i]>op[i] and bull_side[i]:
            D.append([min(op[i],cl[i]), lo[i], i, False]); 
            if len(D)>K: D.pop(0)
        if impulse[i] and rngimp[i] and cl[i]<op[i] and not bull_side[i]:
            S.append([hi[i], max(op[i],cl[i]), i, False])
            if len(S)>K: S.pop(0)
        for z in D:
            if z[3] or i<=z[2] or i-z[2]>zone_age_max: continue
            if lo[i]<=z[0] and hi[i]>=z[1]: z[3]=True; ldt=i; last_d_edge=z[1]
        for z in S:
            if z[3] or i<=z[2] or i-z[2]>zone_age_max: continue
            if hi[i]>=z[1] and lo[i]<=z[0]: z[3]=True; lst=i; last_s_edge=z[0]
        if i<20 or i>n-30 or np.isnan(atrv[i]): continue
        f_d = ldt>=0 and (i-ldt)<=touch_lookback
        f_s = lst>=0 and (i-lst)<=touch_lookback
        if mode=="zone": b_trig=f_d; s_trig=f_s
        elif mode=="combo": b_trig=f_d and sig["vdo_b_ok"][i]; s_trig=f_s and sig["vdo_s_ok"][i]
        elif mode=="vdo": b_trig=sig["vdo_strong_b"][i]; s_trig=sig["vdo_strong_s"][i]
        st=0
        if bull_ok[i] and b_trig and (i-last_sig)>=cooldown: st=1
        elif bear_ok[i] and s_trig and (i-last_sig)>=cooldown: st=-1
        if st==0: continue
        last_sig=i; entry=cl[i]; a=atrv[i]
        if st>0:
            stop=entry-sl_atr_mult*a
            if mode in ("zone","combo") and not np.isnan(last_d_edge) and last_d_edge<entry and entry-last_d_edge<3*a:
                stop=last_d_edge-0.15*a
            risk=entry-stop
            if risk<=0 or np.isnan(risk): continue
            tp=entry+tp_r*risk; done=False
            for j in range(i+1,min(i+horizon,n)):
                if lo[j]<=stop: trades.append((-risk,a)); done=True; break
                if hi[j]>=tp: trades.append((tp_r*risk,a)); done=True; break
            if not done: trades.append((cl[min(i+horizon,n-1)]-entry,a))
        else:
            stop=entry+sl_atr_mult*a
            if mode in ("zone","combo") and not np.isnan(last_s_edge) and last_s_edge>entry and last_s_edge-entry<3*a:
                stop=last_s_edge+0.15*a
            risk=stop-entry
            if risk<=0 or np.isnan(risk): continue
            tp=entry-tp_r*risk; done=False
            for j in range(i+1,min(i+horizon,n)):
                if hi[j]>=stop: trades.append((-risk,a)); done=True; break
                if lo[j]<=tp: trades.append((tp_r*risk,a)); done=True; break
            if not done: trades.append((entry-cl[min(i+horizon,n-1)],a))
    if len(trades)<5: return None
    t=np.array(trades); pnl=t[:,0]/t[:,1]
    wins=(pnl>0).sum(); gp=pnl[pnl>0].sum(); gl=-pnl[pnl<0].sum()
    return dict(n=len(t),wr=100*wins/len(t),mean=pnl.mean(),pf=(gp/gl if gl>0 else 99))

if __name__=="__main__":
    d5=load()
    grids={
      "M5":("5min",["15min"],"60min",dict(regme_len=34,smooth=8,atr_n=14,volbase=50,volmult=2.0,minbody=0.5,vdo_bandlen=50)),
      "M15":("15min",["60min"],"240min",dict(regme_len=55,smooth=8,atr_n=14,volbase=50,volmult=2.0,minbody=0.5,vdo_bandlen=24)),
    }
    print("relaxed touch (lookback 6, age 40), TP=2R, hold<=24 bars:")
    for tf,(rule,confs,trend,cfg) in grids.items():
        for mode in ("zone","combo","vdo"):
            r=sim_zones(d5,rule,cfg,confs,trend,mode)
            print(f"  {tf:>4} {mode:<6}", "n/a" if r is None else f"n={r['n']:>4} wr={r['wr']:.1f}% mean={r['mean']:+.3f} pf={r['pf']:.2f}")
    print("\nTP sensitivity, zone mode:")
    for tf,(rule,confs,trend,cfg) in grids.items():
        for tp in (1.0,1.5,2.0,3.0):
            r=sim_zones(d5,rule,cfg,confs,trend,"zone",tp_r=tp)
            print(f"  {tf:>4} zone tp={tp}", "n/a" if r is None else f"n={r['n']:>4} wr={r['wr']:.1f}% mean={r['mean']:+.3f} pf={r['pf']:.2f}")
