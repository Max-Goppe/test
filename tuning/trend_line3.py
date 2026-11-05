import numpy as np, pandas as pd

def load():
    dfs=[]
    for y in [2022,2023,2024,2025,2026]:
        d=pd.read_csv(f'2022-2026/GOLD_M5_{y}.csv',header=None,encoding='utf-16',
            names=['t','o','h','l','c','v','tv'],parse_dates=['t'])
        dfs.append(d)
    df=pd.concat(dfs).sort_values('t').reset_index(drop=True).set_index('t')
    return df.resample('1h').agg({'o':'first','h':'max','l':'min','c':'last','v':'sum'}).dropna()

h1=load(); C=h1['c'].values; H=h1['h'].values; L=h1['l'].values; n=len(C)
pc=np.concatenate([[C[0]],C[:-1]])
TR=np.maximum(H-L,np.maximum(abs(H-pc),abs(L-pc)))
def rma(x,p):
    out=np.empty(len(x)); out[0]=x[0]; a=1.0/p
    for i in range(1,len(x)): out[i]=out[i-1]+a*(x[i]-out[i-1])
    return out
ATR=rma(TR,100)

def kama(src,N=10,F=2,S=30):
    c=np.asarray(src,float); m=len(c); out=np.empty(m); out[0]=c[0]
    d1=np.abs(np.diff(c,prepend=c[0]))
    for i in range(1,m):
        if i>=N:
            chg=abs(c[i]-c[i-N]); spd=d1[i-N+1:i+1].sum()
            er=chg/spd if spd>0 else 0.0
        else: er=0.5
        k=2.0/(S+1)*(er*(S/(F+1)-1.0/(S+1))+1.0/(S+1))
        out[i]=out[i-1]+k*(c[i]-out[i-1])
    return out

def evaluate(line,bw,dur=1):
    up=line+bw*ATR; dn=line-bw*ATR
    prev=0; flips=0; touch=0; run=0; shortflips=0; pend=0
    for i in range(n):
        cur=prev
        if C[i]>up[i]: raw=1
        elif C[i]<dn[i]: raw=-1
        else: raw=0
        if raw!=0 and prev!=0 and raw==-prev:
            # candidate flip; require dur bars confirmation via simple approach: count then accept
            flips+=1; cur=raw; run=0
        elif raw!=0: cur=raw
        if prev==1 and L[i]<line[i]: touch+=1
        if prev==-1 and H[i]>line[i]: touch+=1
        prev=cur; run+=1
    return flips,touch/years/12

years=n/24/365
print("=== KAMA(close,10,2,30) + hysteresis band b*ATR ===")
for S in [30,60]:
    line=kama(C,10,2,S)
    for b in [2.0,2.2,2.5,2.8,3.0,3.5]:
        f,t=evaluate(line,b)
        dist=np.abs(C-line)/ATR
        print(f"S={S} band={b:.1f}: flips/yr={f/years:6.1f} touches/mo={t:5.1f} med|C-L|/ATR={np.median(dist):.2f}")

# quality of trend state: forward return after flip vs noise
print("\n=== качество направления (KAMA30 band2.5): средняя величина «выезженного» движения после флипа ===")
line=kama(C,10,2,30); b=2.5
up=line+b*ATR; dn=line-b*ATR
prev=0; events=[]
for i in range(n):
    if C[i]>up[i]: raw=1
    elif C[i]<dn[i]: raw=-1
    else: raw=0
    if raw!=0 and prev!=0 and raw==-prev:
        j=min(i+24,n-1)
        move=(C[j]-C[i])*raw
        events.append(move/ATR[i])
    if raw!=0: prev=raw
ev=np.array(events)
print(f"events={len(ev)}, mean move after 24h = {ev.mean():.2f} ATR, median={np.median(ev):.2f}, share>0={(ev>0).mean()*100:.0f}%")
