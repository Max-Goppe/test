exec(open('backtest/scalp_sim2.py').read().split('print("=== BASE')[0])

import itertools
best=[]
for vm,bm,tpR,combo,ra in itertools.product([1.6,2.0],[0.5,0.75],[2.0,3.0],['touch','touch+vdo'],[True,False]):
    r=simulate(volMult=vm,bodyMin=bm,tpR=tpR,combo=combo,requireAll=ra)
    best.append((r[4],r,r[0],vm,bm,tpR,combo,ra))
best.sort(reverse=True)
for b in best[:15]: print(b)
