import numpy as np, pandas as pd, itertools, json

def load():
    dfs=[]
    for y in [2022,2023,2024,2025,2026]:
        d=pd.read_csv(f'2022-2026/GOLD_M5_{y}.csv',header=None,
            names=['t','o','h','l','c','v','tv'],encoding='utf-16',parse_dates=['t'])
        dfs.append(d)
    df=pd.concat(dfs).sort_values('t').reset_index(drop=True)
    df=df.set_index('t')
    h1=df.resample('1h').agg({'o':'first','h':'max','l':'min','c':'last','v':'sum'}).dropna()
    return h1

h1=load()
print("H1 bars:",len(h1))

def atr(series_h,series_l,series_c,n):
    pc=series_c.shift(1)
    tr=np.maximum(series_h-series_l,np.maximum((series_h-pc).abs(),(series_l-pc).abs()))
    return tr.ewm(alpha=1/n,adjust=False).mean()

def kama(close,N,F,S):
    c=close.values; n=len(c)
    er=np.zeros(n); ma=np.zeros(n)
    ma[0]=c[0]
    vol=np.abs(np.diff(c,prepend=c[0]))
    for i in range(1,n):
        if i>=N:
            chg=abs(c[i]-c[i-N])
            spd=vol[i-N+1:i+1].sum()
            er[i]=chg/spd if spd>0 else 0
        else:
            er[i]=1.0 if i==0 else er[i-1]
        k=(er[i]*2/(F+S-1))**2 if False else (er[i]*(2/(F+S-1)-2/(S+2)))**2*0+ (er[i]*(2/(F+1)-2/(S+1))+2/(S+1))**2*0 + (2/(S+1)*(er[i]*(S/(F+1)-1/(S+1))+1/(S+1)))
        k=min(max(k,0),1)
        ma[i]=ma[i-1]+k*(c[i]-ma[i-1])
    return pd.Series(ma,index=close.index)

# simpler KAMA formula
def kama2(close,N=10,F=2,S=30):
    c=close.values; n=len(c)
    out=np.empty(n); out[0]=c[0]
    diff1=np.abs(np.diff(c,prepend=c[0]))
    for i in range(n):
        if i>=N:
            chg=abs(c[i]-c[i-N]); spd=diff1[i-N+1:i+1].sum()
            er=chg/spd if spd>0 else 0.0
        else:
            er=0.5
        k=2.0/(S+1)*(er*(S/(F+1)-1.0/(S+1))+1.0/(S+1))
        if i>0: out[i]=out[i-1]+k*(c[i]-out[i-1])
    return pd.Series(out,index=close.index)

def hma(close,n):
    # simple EMA-based HMA
    half=close.ewm(span=max(n//2,2),adjust=False).mean()
    full=close.ewm(span=n,adjust=False).mean()
    raw=2*half-full
    return raw.ewm(span=max(int(np.sqrt(n)),2),adjust=False).mean()

def evaluate(line,close,h,l,maxcross=3,minwidth_atr=1.0,atr=None):
    # crosses with hysteresis band: line +/- w*atr
    w=minwidth_atr
    up=line+w*atr; dn=line-w*atr
    state=0  # 1 above band, -1 below, 0 inside
    flips=0; touches=0; prev=0
    cl=close.values; U=up.values; D=dn.values
    for i in range(len(cl)):
        s=prev
        if cl[i]>U[i]: cur=1
        elif cl[i]<D[i]: cur=-1
        else: cur=s
        if cur!=s and s!=0:
            # flip only counted when going through? count direction changes
            if cur==-s: flips+=1
        if s!=0 and ((s==1 and l[i]<line.values[i]) or (s==-1 and h[i]>line.values[i])):
            touches+=1
        prev=cur
    return flips,touches

atrs={}
for n in [50,100,200]:
    atrs[n]=atr(h1['h'],h1['l'],h1['c'],n)

results=[]
cands=[]
for N,F,S in itertools.product([10],[2],[20,30,40,60,80]):
    for band_mult in [0.8,1.0,1.3,1.6,2.0]:
        line=kama2(h1['c'],N=N,F=F,S=S)
        a=atrs[100]
        rel_width=((line+band_mult*a)-(line-band_mult*a))/h1['c']
        flips,touches=evaluate(line,h1['c'],h1['h'],h1['l'],atr=a,minwidth_atr=band_mult)
        cands.append(('KAMA S=%d band=%.1f'%(S,band_mult),flips,touches,len(h1)/ (24*260)*0+flips))
        results.append(dict(name='KAMA S=%d band=%.1fxATR'%(S,band_mult),flips=flips,touches=touches,
                            per_year=flips/ (len(h1)/24/365)))
for n in [80,120,200,300]:
    for band_mult in [0.8,1.0,1.3,1.6,2.0]:
        line=hma(h1['c'],n)
        a=atrs[100]
        flips,touches=evaluate(line,h1['c'],h1['h'],h1['l'],atr=a,minwidth_atr=band_mult)
        results.append(dict(name='HMA n=%d band=%.1fxATR'%(n,band_mult),flips=flips,touches=touches,
                            per_year=flips/(len(h1)/24/365)))

res=pd.DataFrame(results).sort_values('per_year')
print(res.to_string(index=False))
