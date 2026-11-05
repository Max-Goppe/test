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

line=kama(C,10,2,30)
years=n/24/365

# state machine: flip only when close crosses line AND then closes beyond b*ATR from line (full traversal of hysteresis band)
# "отбой" = price touched line but returned without crossing it by confirmDist*ATR -> no flip
def sim(band,confirm,src=C):
    prev=0; flips=0; rebounds=0; whips=0; lastdir=0; pend=0; pendbar=0
    states=[]
    for i in range(n):
        cur=prev
        if prev==0:
            if src[i]>band*ATR[i]: cur=1
            elif src[i]<-band*ATR[i]: cur=-1
        else:
            # candidate cross: close on opposite side of line
            if (prev==1 and src[i]<line[i]) or (prev==-1 and src[i]>line[i]):
                if pend==0: pendbar=i
                pend+=1
            else:
                pend=0
            # confirmed flip: closed beyond line +/- confirm*ATR on other side
            if prev==1 and src[i] < line[i]-confirm*ATR[i]:
                cur=-1
            elif prev==-1 and src[i] > line[i]+confirm*ATR[i]:
                cur=1
        if cur!=prev and cur!=0:
            flips+=1
            if flips>1 and (i-pendbar)<3: pass
        prev=cur; states.append(cur)
    return flips,states

# simpler & standard: Schmitt trigger around LINE itself with half-width h*ATR
def schmitt(hw, minbars=0):
    st=0; flips=0; runs=[]; run=0
    for i in range(n):
        d=C[i]-line[i]
        if st>=0 and d> hw*ATR[i]: newst=1
        elif st<=0 and d< -hw*ATR[i]: newst=-1
        else: newst=st
        if newst!=st:
            if minbars>0 and run<minbars:
                flips_whip=1
            flips+=1; run=0
        st=newst; run+=1; 
        runs.append(run)
    return flips

print("Schmitt-trigger по линии KAMA, порог ±h*ATR от самой линии:")
for h in [0.5,0.8,1.0,1.2,1.5,2.0]:
    f=schmitt(h)
    print(f"  h={h}: смен тренда/год = {f/years:.1f}")

# add min-duration filter on top
def schmitt2(h,minbars):
    st=0; pending=0; pdir=0; flips=0; run=0; short=0
    for i in range(n):
        d=C[i]-line[i]
        want=st
        if st<=0 and d> h*ATR[i]: want=1
        if st>=0 and d< -h*ATR[i]: want=-1
        if want!=0 and want!=st:
            if want==pdir: pending+=1
            else: pdir=want; pending=1
            if pending>=minbars:
                flips+=1; st=want; pending=0
        else:
            pending=0
    return flips
print("\n+ фильтр мин. длительности (N бар подряд за порогом):")
for h in [0.8,1.0,1.2]:
    for mb in [1,3,6]:
        f=schmitt2(h,mb)
        print(f"  h={h} minbars={mb}: смен/год = {f/years:.1f}")
