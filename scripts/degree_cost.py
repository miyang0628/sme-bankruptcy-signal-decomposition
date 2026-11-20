# M1: decision analysis for the DEGREE model (the configuration the paper recommends),
# using the exact pipeline of notebook 10 (isotonic recalibration on val, prevalence-
# independent cost sweeps). Adds a third model "degree" alongside intra and contagion.
import warnings; warnings.filterwarnings("ignore")
from pathlib import Path
import numpy as np, pandas as pd, torch
import torch.nn.functional as F
from torch import nn
from sklearn.preprocessing import StandardScaler
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score, brier_score_loss, roc_curve
from torch_geometric.nn import HeteroConv, SAGEConv

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT/"data"/"processed"; TAB = ROOT/"results"/"tables"
SEEDS = list(range(5))

data = torch.load(PROCESSED/"smesd_hetero.pt", weights_only=False)
comp = data["company"]; N=comp.num_nodes; PDIM=data["person"].x.size(1)
Xraw = comp.x.numpy().astype(np.float64); Xraw[:, :2]=np.log1p(np.clip(Xraw[:, :2],0,None))
y=comp.y; trm,vam,tem=comp.train_mask,comp.val_mask,comp.test_mask
EDGE=[et for et in data.edge_types if int(et[1].split("_")[1])<=11]
ei_dict={et:data[et].edge_index for et in EDGE}

# degree features (identical to nb12)
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

# standardized company feats for GNN (as nb10)
Xc_std = StandardScaler().fit(Xraw[trm.numpy()]).transform(Xraw).astype(np.float32)
comp_x_gnn = torch.tensor(Xc_std)
IN_DIMS={"company":comp_x_gnn.size(1),"person":PDIM}

class HeteroGNN(nn.Module):
    def __init__(s_,h=64,o=2,L=2,d=0.3):
        super().__init__(); s_.drop=d
        s_.lin_in=nn.ModuleDict({nt:nn.Linear(dd,h) for nt,dd in IN_DIMS.items()})
        s_.convs=nn.ModuleList([HeteroConv({et:SAGEConv((h,h),h) for et in EDGE},aggr="sum") for _ in range(L)])
        s_.head=nn.Sequential(nn.Linear(h,h),nn.ReLU(),nn.Dropout(d),nn.Linear(h,o))
    def forward(s_,xd,eid):
        hh={nt:F.relu(s_.lin_in[nt](x)) for nt,x in xd.items()}
        for c in s_.convs:
            hh=c(hh,eid); hh={nt:F.dropout(F.relu(v),p=s_.drop,training=s_.training) for nt,v in hh.items()}
        return s_.head(hh["company"])
class MLP(nn.Module):
    def __init__(s_,d,h=64,o=2,dr=0.3):
        super().__init__(); s_.net=nn.Sequential(nn.Linear(d,h),nn.ReLU(),nn.Dropout(dr),nn.Linear(h,h),nn.ReLU(),nn.Dropout(dr),nn.Linear(h,o))
    def forward(s_,x): return s_.net(x)
def cw():
    cl=torch.bincount(y[trm],minlength=2).float(); return cl.sum()/(2*cl.clamp(min=1))

# MLP feature sets (standardized on train)
def mlp_feat(mat):
    sc=StandardScaler().fit(mat[trm.numpy()]); return torch.tensor(sc.transform(mat).astype(np.float32))
X_intra = mlp_feat(Xraw)
X_degree = mlp_feat(np.concatenate([Xraw,degf],1))
GNN_XD = {"company":comp_x_gnn, "person":data["person"].x}

def train_probs(kind, seed):
    torch.manual_seed(seed); np.random.seed(seed); w=cw()
    if kind=="contagion":
        model=HeteroGNN(); fwd=lambda:model(GNN_XD, ei_dict)
    elif kind=="degree":
        model=MLP(X_degree.size(1)); fwd=lambda:model(X_degree)
    else:
        model=MLP(X_intra.size(1)); fwd=lambda:model(X_intra)
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
    return p[vam.numpy()], p[tem.numpy()]

yv=y[vam.numpy()].numpy(); yt=y[tem.numpy()].numpy()
probs={}
for kind in ["intra","contagion","degree"]:
    pv=np.mean([train_probs(kind,s)[0] for s in SEEDS],axis=0)
    pt=np.mean([train_probs(kind,s)[1] for s in SEEDS],axis=0)
    probs[kind]=(pv,pt)

# isotonic recal
cal={}; rows=[]
for kind in ["intra","contagion","degree"]:
    pv,pt=probs[kind]; iso=IsotonicRegression(out_of_bounds="clip").fit(pv,yv); ptc=iso.predict(pt)
    cal[kind]=ptc
    rows.append({"model":kind,"Brier_raw":round(brier_score_loss(yt,pt),4),
                 "Brier_isotonic":round(brier_score_loss(yt,ptc),4),"AUC":round(roc_auc_score(yt,pt),4)})
cal_df=pd.DataFrame(rows); cal_df.to_csv(TAB/"14_degree_recalibration.csv",index=False)
print("=== recalibration ==="); print(cal_df.to_string(index=False))

def mec(p,pi,cr):
    fpr,tpr,_=roc_curve(yt,p); fnr=1-tpr; ec=pi*cr*fnr+(1-pi)*fpr; return float(ec.min())

# base-rate sweep: degree vs intra  AND  degree relative to contagion
PI=0.05; CR=5
pis=np.linspace(0.02,0.65,40); rel=[]
for pi in pis:
    ec_i=mec(cal["intra"],pi,CR); ec_d=mec(cal["degree"],pi,CR); ec_c=mec(cal["contagion"],pi,CR)
    rel.append({"base_rate":round(float(pi),4),"cost_ratio":CR,
                "intra_cost":round(ec_i,5),"degree_cost":round(ec_d,5),"contagion_cost":round(ec_c,5),
                "degree_saving_vs_intra_pct":round(100*(ec_i-ec_d)/ec_i if ec_i>0 else 0,3),
                "degree_minus_contagion_cost":round(ec_d-ec_c,5)})
rdf=pd.DataFrame(rel); rdf.to_csv(TAB/"14_degree_cost_vs_baserate.csv",index=False)
at5=rdf.iloc[(rdf["base_rate"]-PI).abs().argmin()]
print(f"\n=== degree-model decision analysis (isotonic-calibrated, 5 seeds) ===")
print(f"  AUC: intra {cal_df.set_index('model').loc['intra','AUC']}, "
      f"degree {cal_df.set_index('model').loc['degree','AUC']}, "
      f"contagion {cal_df.set_index('model').loc['contagion','AUC']}")
print(f"  at base rate {PI:.0%}, {CR}:1 -> degree saving vs intra = {at5['degree_saving_vs_intra_pct']:.1f}%")
print(f"  degree saving positive across {rdf['base_rate'].min():.0%}-{rdf['base_rate'].max():.0%}: "
      f"{'yes' if (rdf['degree_saving_vs_intra_pct']>0).all() else 'no'} "
      f"(min {rdf['degree_saving_vs_intra_pct'].min():.1f}%, max {rdf['degree_saving_vs_intra_pct'].max():.1f}%)")
# does degree cost <= contagion cost at every swept base rate?
dmc=rdf["degree_minus_contagion_cost"]
print(f"  degree expected cost <= contagion at all base rates: {'yes' if (dmc<=1e-9).all() else 'no'} "
      f"(max degree-minus-contagion = {dmc.max():.5f})")
# cost-ratio sweep at pi=5%
ratios=np.arange(1,21); crows=[]
for kind in ["intra","contagion","degree"]:
    for r in ratios:
        crows.append({"model":kind,"base_rate":PI,"cost_ratio":int(r),"min_expected_cost":round(mec(cal[kind],PI,r),5)})
pd.DataFrame(crows).to_csv(TAB/"14_degree_cost_vs_ratio.csv",index=False)
print("\nsaved: 14_degree_recalibration.csv, 14_degree_cost_vs_baserate.csv, 14_degree_cost_vs_ratio.csv")
