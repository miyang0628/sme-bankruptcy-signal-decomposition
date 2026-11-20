import warnings; warnings.filterwarnings("ignore")
from pathlib import Path
import numpy as np, pandas as pd, torch
import torch.nn.functional as F
from torch import nn
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from scipy.stats import norm

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT/"data"/"processed"; TAB=ROOT/"results"/"tables"; SEEDS=list(range(10))

def compute_midrank(x):
    J=np.argsort(x); Z=x[J]; N=len(x); T=np.zeros(N); i=0
    while i<N:
        j=i
        while j<N and Z[j]==Z[i]: j+=1
        T[i:j]=0.5*(i+j-1)+1; i=j
    T2=np.empty(N); T2[J]=T; return T2

def delong_two(y, s1, s2):
    # correct DeLong for two correlated ROC AUCs on same sample
    y=np.asarray(y); pos=y==1; neg=y==0; m=int(pos.sum()); n=int(neg.sum())
    preds=[s1,s2]; k=2
    vx=np.empty([k,m]); vy=np.empty([k,n]); aucs=np.empty(k)
    for r in range(k):
        sp=preds[r][pos]; sn=preds[r][neg]
        tzp=compute_midrank(preds[r])[pos]; 
        txp=compute_midrank(sp); tyn=compute_midrank(sn)
        aucs[r]=tzp.sum()/m/n-(m+1.0)/2.0/n
        vx[r]=(compute_midrank(preds[r])[pos]-txp)/n
        vy[r]=1.0-(compute_midrank(preds[r])[neg]-tyn)/m
    sx=np.cov(vx); sy=np.cov(vy); cov=sx/m+sy/n
    L=np.array([[1.0,-1.0]]); var=float((L@cov@L.T)[0,0])
    z=(aucs[0]-aucs[1])/np.sqrt(var) if var>0 else 0.0
    return aucs[0],aucs[1],2*(1-norm.cdf(abs(z)))

data=torch.load(PROCESSED/"smesd_hetero.pt",weights_only=False)
comp=data["company"]; N=comp.num_nodes; PDIM=data["person"].x.shape[1]
Xbase=comp.x.numpy().astype(np.float64); Xbase[:,:2]=np.log1p(np.clip(Xbase[:,:2],0,None))
y=comp.y; trm,vam,tem=comp.train_mask,comp.val_mask,comp.test_mask
ET=[et for et in data.edge_types if int(et[1].split("_")[1])<=11]
FF={6,7,10,11}; PM={0,1,2,3,4,5,8,9}; dff=np.zeros(N); dpm=np.zeros(N)
for et in data.edge_types:
    s,r,t=et; rel=int(r.split("_")[1]); ei=data[et].edge_index.numpy()
    for side,nt in [(0,s),(1,t)]:
        if nt=="company":
            nd=ei[side]
            if rel in FF: np.add.at(dff,nd,1.0)
            elif rel in PM: np.add.at(dpm,nd,1.0)
dt=dff+dpm
degf=np.column_stack([np.log1p(dff),np.log1p(dpm),np.log1p(dt),(dt>0).astype(float)]).astype(np.float32)
Xdeg=np.concatenate([Xbase,degf],1)

class G(nn.Module):
    def __init__(s_,ets,ind,h=64,o=2,L=2,d=0.3):
        super().__init__()
        from torch_geometric.nn import HeteroConv,SAGEConv
        s_.d=d; s_.li=nn.ModuleDict({nt:nn.Linear(dd,h) for nt,dd in ind.items()})
        s_.cv=nn.ModuleList([HeteroConv({et:SAGEConv((h,h),h) for et in ets},aggr="sum") for _ in range(L)])
        s_.hd=nn.Sequential(nn.Linear(h,h),nn.ReLU(),nn.Dropout(d),nn.Linear(h,o))
    def forward(s_,xd,eid):
        hh={nt:F.relu(s_.li[nt](x)) for nt,x in xd.items()}
        for c in s_.cv:
            hh=c(hh,eid); hh={nt:F.dropout(F.relu(v),p=s_.d,training=s_.training) for nt,v in hh.items()}
        return s_.hd(hh["company"])
class M(nn.Module):
    def __init__(s_,d,h=64,o=2,dr=0.3):
        super().__init__(); s_.n=nn.Sequential(nn.Linear(d,h),nn.ReLU(),nn.Dropout(dr),nn.Linear(h,h),nn.ReLU(),nn.Dropout(dr),nn.Linear(h,o))
    def forward(s_,x): return s_.n(x)
def cw():
    cl=torch.bincount(y[trm],minlength=2).float(); return cl.sum()/(2*cl.clamp(min=1))
def prob(kind,seed,feat=None,eid=None,cf=None):
    torch.manual_seed(seed); np.random.seed(seed); w=cw()
    if kind=="gnn":
        Xc=StandardScaler().fit(cf[trm.numpy()]).transform(cf).astype(np.float32)
        xd={"company":torch.tensor(Xc),"person":data["person"].x}; model=G(list(eid.keys()),{"company":Xc.shape[1],"person":PDIM}); fwd=lambda:model(xd,eid)
    else:
        sc=StandardScaler().fit(feat[trm.numpy()]); X=torch.tensor(sc.transform(feat).astype(np.float32)); model=M(X.size(1)); fwd=lambda:model(X)
    opt=torch.optim.Adam(model.parameters(),lr=5e-3,weight_decay=5e-4); bv,bs,wt=-1,None,0
    for _ in range(200):
        model.train(); opt.zero_grad(); F.cross_entropy(fwd()[trm],y[trm],weight=w).backward(); opt.step(); model.eval()
        with torch.no_grad(): p=F.softmax(fwd(),1)[:,1].numpy()
        va=roc_auc_score(y[vam].numpy(),p[vam.numpy()])
        if va>bv: bv,bs,wt=va,{k:v.clone() for k,v in model.state_dict().items()},0
        else:
            wt+=1
            if wt>=30: break
    model.load_state_dict(bs); model.eval()
    with torch.no_grad(): p=F.softmax(fwd(),1)[:,1].numpy()
    return p[tem.numpy()]
yt=y[tem.numpy()].numpy(); eidr={et:data[et].edge_index for et in ET}; pv=[]
for sd in SEEDS:
    pa=prob("mlp",sd,feat=Xdeg); pb=prob("gnn",sd,eid=eidr,cf=Xdeg)
    ag,am,p=delong_two(yt,pb,pa); pv.append(p)
    print(f"seed {sd}: gnn+deg {ag:.4f} mlp+deg {am:.4f} DeLong p {p:.4f}",flush=True)
pv=np.array(pv); print(f"\nDeLong median p = {np.median(pv):.4f} | seeds p<0.05: {(pv<0.05).sum()}/10")
pd.DataFrame({"seed":SEEDS,"delong_p":pv.round(4)}).to_csv(TAB/"13_matched_delong.csv",index=False)
