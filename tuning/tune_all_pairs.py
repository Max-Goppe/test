import numpy as np, pandas as pd, glob, os

def load_h1(path):
    raw=open(path,'rb').read()
    for enc in ['utf-8','cp1251','latin1']:
        try: txt=raw.decode(enc); break
        except: pass
    import io
    df=pd.read_csv(io.StringIO(txt),sep=r'\s+',header=0,
        names=['DATE','TIME','OPEN','HIGH','LOW','CLOSE','TICKVOL','VOL','SPREAD'],
        engine='python',encoding=None)
    df=df[df['DATE'].astype(str).str.match(r'^\d{4}\.')]
    df['dt']=pd.to_datetime(df['DATE']+' '+df['TIME'].str[:5],format='%Y.%m.%d %H:%M')
    df=df.set_index('dt')[['OPEN','HIGH','LOW','CLOSE']].astype(float)
    return df

def rma(x,p):
    out=np.empty(len(x)); out[0]=x[0]; a=1.0/p
    for i in range(1,len(x)): out[i]=out[i-1]+a*(x[i]-out[i-1])
    return out

def kama(c,N=10,F=2,S=30):
    m=len(c); out=np.empty(m); out[0]=c[0]
    d1=np.abs(np.diff(c,prepend=c[0]))
    er=np.full(m,0.5)
    chg=np.abs(c-np.roll(c,N)); chg[:N]=0
    spd=np.array([d1[max(0,i-N+1):i+1].sum() if i>=N else 1 for i in range(m)])
    good=(np.arange(m)>=N)&(spd>0)
    er[good]=chg[good]/spd[good]
    sc=2.0/(F+1)-2.0/(S+1)
    k=(er*sc+2.0/(S+1))**2
    for i in range(1,m): out[i]=out[i-1]+k[i]*(c[i]-out[i-1])
    return out

def sim(C,H,L,ATR,ln,cm,cb):
    n=len(C); up=ln+cm*ATR; dn=ln-cm*ATR
    trend=0; pend=0; runs=[]
    for i in range(n):
        prev=trend
        if C[i]<dn[i] and trend!=-1: pend = pend-1 if pend<0 else -1
        elif C[i]>up[i] and trend!=1: pend = pend+1 if pend>0 else 1
        else: pend=0
        if pend>=cb: trend=1
        if pend<=-cb: trend=-1
        if trend!=prev and trend!=0: runs.append(i)
    arr=np.array(runs)
    gaps=np.diff(arr) if len(arr)>1 else np.array([],dtype=int)
    whips=int((gaps<12).sum()) if len(gaps) else 0
    medgap=float(np.median(gaps)) if len(gaps) else 0
    return len(runs),whips,medgap

pairs={'GOLD':'pine/Графики/Gold/GOLD_H1.csv','EURUSD':'pine/Графики/EURUSD/EURUSD_H1.csv',
       'GBPUSD':'pine/Графики/GBPUSD/GBPUSD_H1.csv','EURGBP':'pine/Графики/EURGBP/EURGBP_H1.csv',
       'USDCAD':'pine/Графики/USDCAD/USDCAD_H1.csv'}
years_span=None
for name,path in pairs.items():
    df=load_h1(path); C=df['CLOSE'].values; H=df['HIGH'].values; L=df['LOW'].values
    pc=np.concatenate([[C[0]],C[:-1]])
    TR=np.maximum(H-L,np.maximum(abs(H-pc),abs(L-pc)))
    ATR=rma(TR,100)
    yrs=len(C)/24/365
    ln=kama(C,10,2,30)
    print(f"\n=== {name}: {len(C)} H1 баров (~{yrs:.1f} года), диапазон {C[0]:.5g}-{C[-1]:.5g} ===")
    best=None
    for cm in [1.0,1.2,1.5,1.8]:
        for cb in [2,3]:
            f,w,g=sim(C,H,L,ATR,ln,cm,cb)
            tag=f"chan={cm} conf={cb}"
            ok = (f/yrs<=60 and w/yrs<=3)
            print(f"  {tag:18s}: смен/год={f/yrs:6.1f} ерш/год={w/yrs:4.1f} медиан.длитность={g:5.0f}ч {'OK' if ok else ''}")
