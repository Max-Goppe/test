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
def rma(x,n):
    out=np.empty(len(x)); out[0]=x.mean() if False else x[0]
    a=1.0/n
    for i in range(1,len(x)): out[i]=out[i-1]+a*(x[i]-out[i-1])
    return out
ATR=rma(TR,100)

def kama(src,N,F,S):
    c=np.asarray(src,dtype=float); m=len(c); out=np.empty(m); out[0]=c[0]
    d1=np.abs(np.diff(c,prepend=c[0]))
    for i in range(1,m):
        if i>=N:
            chg=abs(c[i]-c[i-N]); spd=d1[i-N+1:i+1].sum()
            er=chg/spd if spd>0 else 0.0
        else: er=0.5
        k=2.0/(S+1)*(er*(S/(F+1)-1.0/(S+1))+1.0/(S+1))
        out[i]=out[i-1]+k*(c[i]-out[i-1])
    return out

def supertrend(src,h,l,p,mult):
    c=np.asarray(src,float); m=len(c)
    atr_=rma(TR,p)
    hl2=(np.asarray(h,float)+np.asarray(l,float))/2
    ub=hl2+mult*atr_; lb=hl2-mult*atr_
    fub=ub.copy(); flb=lb.copy()
    st=np.zeros(m); dirn=np.ones(m,int)
    for i in range(1,m):
        fub[i]=ub[i] if (ub[i]<fub[i-1] or c[i-1]>fub[i-1]) else fub[i-1]
        flb[i]=lb[i] if (lb[i]>flb[i-1] or c[i-1]<flb[i-1]) else flb[i-1]
        if dirn[i-1]==1:
            dirn[i]=-1 if c[i]<flb[i-1] else 1
        else:
            dirn[i]=1 if c[i]>fub[i-1] else -1
        st[i]=flb[i] if dirn[i]==1 else fub[i]
    return st,dirn

def donchian_mid(h,l,p):
    m=len(h); mid=np.empty(m)
    for i in range(m):
        s=max(0,i-p+1)
        mid[i]=(h[s:i+1].max()+l[s:i+1].min())/2
    return mid

def evaluate(line,bw_atr):
    # state machine with hysteresis around line; flip = close crosses line by bw beyond opposite side
    up=line+bw_atr*ATR; dn=line-bw_atr*ATR
    prev=0; flips=0; touch=0; daysince=0; maxrun=0; run=0
    states=[]
    for i in range(n):
        cur=prev
        if C[i]>up[i]: cur=1
        elif C[i]<dn[i]: cur=-1
        if prev!=0 and cur==-prev: flips+=1; run=0
        elif cur!=0: run+=1; maxrun=max(maxrun,run)
        if prev==1 and L[i]<line[i]: touch+=1
        if prev==-1 and H[i]>line[i]: touch+=1
        prev=cur; states.append(cur)
    return flips,touch,maxrun,states

years=n/24/365
rows=[]
for S in [20,30,40,60]:
    line=kama(C,10,2,S)
    for b in [1.0,1.5,2.0,2.5,3.0]:
        f,t,mr,_=evaluate(line,b)
        rows.append(('KAMA%d band%.1f'%(S,b),f,t,mr))
for p,mu in [(10,2),(10,3),(15,2),(15,3),(20,2),(20,3)]:
    st,_=supertrend(C,H,L,p,mu)
    rows.append(('ST%d,%d'%(p,mu),*evaluate(st,0)[:3]))
for p in [24,48,96,150]:
    dm=donchian_mid(H,L,p)
    for b in [0.5,1.0]:
        f,t,mr,_=evaluate(dm,b)
        rows.append(('DonchMid%d band%.1f'%(p,b),f,t,mr))
# KAMA on HL2
for S in [30,60]:
    line=kama((H+L)/2,10,2,S)
    for b in [1.5,2.0,2.5]:
        f,t,mr,_=evaluate(line,b)
        rows.append(('KAMAHL2-%d band%.1f'%(S,b),f,t,mr))

print(f"{'variant':24s} {'flips/yr':>9s} {'touch/mo':>9s} {'maxrun_h':>9s}")
for name,f,t,mr in rows:
    print(f"{name:24s} {f/years:9.1f} {t/years/12:9.1f} {mr:9d}")
