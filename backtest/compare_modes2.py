#!/usr/bin/env python3
"""Zone-touch vs VDO vs combo, realistic exit horizon for scalping (max 12 bars), M5/M15."""
import sys; sys.path.insert(0,"/workspace")
import numpy as np
from backtest_scalp import load, resample, Engine, _MTF_CACHE
exec(open("/workspace/backtest/compare_modes.py").read().split('if __name__')[0].split('"""')[2])

def simulate(sig, mode, tp_r=2.0, sl_atr_mult=1.5, cooldown=6, horizon=12):
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
            done=False
            for j in range(i+1,min(i+horizon,n)):
                if lo[j]<=stop: trades.append((-risk,a)); done=True; break
                if hi[j]>=tp: trades.append((tp_r*risk,a)); done=True; break
            if not done: trades.append((cl[min(i+horizon,n-1)]-entry,a))
        else:
            stop=entry+sl_atr_mult*a
            cand=np.nonzero((~np.isnan(s_top[max(0,i-40):i])) & (s_top[max(0,i-40):i]>entry))[0]
            if len(cand)>0: stop=s_top[i-1-cand[-1]]+0.15*a
            risk=stop-entry
            if risk<=0 or np.isnan(risk): continue
            tp=entry-tp_r*risk
            done=False
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
    print("realistic scalp exits (TP=2R, SL zone-edge, max 12 bars hold):")
    for tf,(rule,confs,trend,cfg) in grids.items():
        sig=build_signals(d5,rule,cfg,confs,trend,cooldown=6)
        for mode in ("zone","vdo","combo"):
            r=simulate(sig,mode)
            if r: print(f"  {tf:>4} {mode:<6} n={r['n']:>4} wr={r['wr']:.1f}% mean={r['mean']:+.3f} ATR pf={r['pf']:.2f}")
            else: print(f"  {tf:>4} {mode:<6} too few trades")
    print("\nno-TP variant (pure touch reaction, exit after 6 bars):")
    for tf,(rule,confs,trend,cfg) in grids.items():
        sig=build_signals(d5,rule,cfg,confs,trend,cooldown=6)
        for mode in ("zone","combo"):
            # emulate: TP huge so only time-exit hits
            r=simulate(sig,mode,tp_r=99,horizon=6)
            if r: print(f"  {tf:>4} {mode:<6} n={r['n']:>4} wr={r['wr']:.1f}% mean={r['mean']:+.3f} pf={r['pf']:.2f}")
