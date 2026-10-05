import time, math, statistics, itertools

src=open('/workspace/backtest/rvp_fast.py').read().split("PRESETS =")[0]
g={}
exec(src,g)
TF=g['TF']; run_tf=g['run_tf']; evaluate=g['evaluate']

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
        ev=evaluate(rows,sigs,atr,hor)
        res[hor]=ev
    # combined objective: mean of avg pnl across horizons, require enough trades
    if not all(res[h] and res[h]['n']>=30 for h in HORIZONS): return None
    avgs=[res[h]['avg'] for h in HORIZONS]
    wr=res[24]['winrate'] if res[24] else 0
    n=min(res[h]['n'] for h in HORIZONS)
    return (statistics.mean(avgs), wr, n, res)

# Stage 1: signal-rule grid on H1
rows_h1=TF['H1']
best=[]
grid=[]
for mc in (2,3):
    for tl in (1,2,3,5):
        for bl in (1,2,3,5):
            for ia in (5,8,15,25,40):
                for cd in (5,10,20,40):
                    grid.append(dict(minConf=mc,touchLook=tl,brkLook=bl,impulseAge=ia,cooldown=cd))
t0=time.time()
for v in grid:
    r=score(base(**v),rows_h1)
    if r: best.append((r[0],r[1],r[2],v))
print("stage1 done %.0fs, configs kept: %d"%(time.time()-t0,len(best)))
best.sort(reverse=True)
print("\n=== STAGE 1: top signal rules (H1, obj=avg ATR over hor 6/12/24) ===")
for s,wr,n,v in best[:10]:
    print(f"obj={s:+.3f} win@24={wr:.0f}% n={n} | conf={v['minConf']} touch={v['touchLook']} brk={v['brkLook']} age={v['impulseAge']} cd={v['cooldown']}")

# baseline default vs best rule set detail
def detail(cfg,label,rows):
    r=score(cfg,rows)
    if not r: print(label,"-> too few signals"); return
    s,wr,n,res=r
    print(f"{label}: obj={s:+.3f} | "+" | ".join(f"h{h}: avg={res[h]['avg']:+.3f} win={res[h]['winrate']:.0f}% n={res[h]['n']}" for h in HORIZONS))

detail(base(),"DEFAULT(conf2,touch3,brk3,age15,cd10)",rows_h1)
if best: detail(base(**best[0][3]),"BEST-STAGE1",rows_h1)

# Stage 2: tune core params around the best rule set
rules=best[0][3] if best else dict(minConf=2,touchLook=3,brkLook=3,impulseAge=15,cooldown=10)
best2=[]
for vm in (1.5,1.8,2.2,2.8,3.5):
    for ba in (0.5,0.7,0.9):
        for ra in (0.8,1.1,1.4):
            for rl in (34,55,89):
                for bl_ in (34,50):
                    cfg=base(volumeMult=vm,**rules); cfg['volmult']=vm; cfg['bodyAtr']=ba; cfg['rangeAtr']=ra; cfg['regmeLen']=rl; cfg['bandLen']=bl_
                    r=score(cfg,rows_h1)
                    if r: best2.append((r[0],r[1],r[2],dict(volmult=vm,bodyAtr=ba,rangeAtr=ra,regmeLen=rl,bandLen=bl_)))
best2.sort(reverse=True)
print("\n=== STAGE 2: top core params (H1) ===")
for s,wr,n,v in best2[:10]:
    print(f"obj={s:+.3f} win@24={wr:.0f}% n={n} | volm={v['volmult']} body={v['bodyAtr']} rng={v['rangeAtr']} regmeL={v['regmeLen']} bandL={v['bandLen']}")

final=dict(rules); final.update(best2[0][3] if best2 else {})
cfgF=base(**final)
cfgF.update(best2[0][3] if best2 else {})
print("\nFINAL CFG:", {k:cfgF[k] for k in ('regmeLen','volmult','bodyAtr','rangeAtr','bandLen','minConf','touchLook','brkLook','impulseAge','cooldown')})
detail(cfgF,"FINAL @H1",rows_h1)
detail(cfgF,"FINAL @M15",TF['M15'])
