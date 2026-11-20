# Matched-input experiment: give BOTH the non-graph MLP and the GNN the same
# connection-count (degree) features, compare under identical protocol (10 seeds, mask A).
# Reuses notebook 12's model defs, feature construction, and training loop verbatim.
import warnings; warnings.filterwarnings("ignore")
from pathlib import Path
import numpy as np, pandas as pd, torch
import torch.nn.functional as F
from torch import nn
from scipy.stats import ttest_rel
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, f1_score, roc_curve

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
TAB = ROOT / "results" / "tables"
SEEDS = list(range(10))

def ks_stat(y, s):
    fpr, tpr, _ = roc_curve(y, s); return float(np.max(tpr - fpr))
def metrics(y, s, thr=0.5):
    return {"AUC": roc_auc_score(y, s), "F1": f1_score(y, (s >= thr).astype(int), zero_division=0), "KS": ks_stat(y, s)}

data = torch.load(PROCESSED / "smesd_hetero.pt", weights_only=False)
comp = data["company"]; N_COMP = comp.num_nodes
person_x = data["person"].x.numpy(); PDIM = person_x.shape[1]
Xbase = comp.x.numpy().astype(np.float64)
Xbase[:, :2] = np.log1p(np.clip(Xbase[:, :2], 0, None))
y = comp.y
trm, vam, tem = comp.train_mask, comp.val_mask, comp.test_mask
ET_0_11 = [et for et in data.edge_types if int(et[1].split("_")[1]) <= 11]

# Degree features (identical to nb12)
FF = {6, 7, 10, 11}; PM = {0, 1, 2, 3, 4, 5, 8, 9}
deg_ff = np.zeros(N_COMP); deg_pm = np.zeros(N_COMP)
for et in data.edge_types:
    s, r, t = et; rel = int(r.split("_")[1]); ei = data[et].edge_index.numpy()
    for side, nt in [(0, s), (1, t)]:
        if nt == "company":
            nodes = ei[side]
            if rel in FF: np.add.at(deg_ff, nodes, 1.0)
            elif rel in PM: np.add.at(deg_pm, nodes, 1.0)
deg_total = deg_ff + deg_pm
degree_feats = np.column_stack([np.log1p(deg_ff), np.log1p(deg_pm), np.log1p(deg_total),
                                (deg_total > 0).astype(float)]).astype(np.float32)

class HeteroGNN(nn.Module):
    def __init__(self, edge_types, in_dims, hidden=64, out=2, layers=2, dropout=0.3):
        super().__init__()
        from torch_geometric.nn import HeteroConv, SAGEConv
        self.drop = dropout
        self.lin_in = nn.ModuleDict({nt: nn.Linear(d, hidden) for nt, d in in_dims.items()})
        self.convs = nn.ModuleList([HeteroConv({et: SAGEConv((hidden, hidden), hidden)
                     for et in edge_types}, aggr="sum") for _ in range(layers)])
        self.head = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden, out))
    def forward(self, xd, eid):
        h = {nt: F.relu(self.lin_in[nt](x)) for nt, x in xd.items()}
        for conv in self.convs:
            h = conv(h, eid); h = {nt: F.dropout(F.relu(v), p=self.drop, training=self.training) for nt, v in h.items()}
        return self.head(h["company"])

class MLP(nn.Module):
    def __init__(self, d, hidden=64, out=2, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, hidden), nn.ReLU(), nn.Dropout(dropout),
                                 nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden, out))
    def forward(self, x): return self.net(x)

def cw():
    cls = torch.bincount(y[trm], minlength=2).float(); return cls.sum() / (2 * cls.clamp(min=1))

# run supports arbitrary company feature matrix for the GNN (company_feat), else Xbase.
def run(kind, seed, feat=None, eid=None, company_feat=None):
    torch.manual_seed(seed); np.random.seed(seed); w = cw()
    if kind == "gnn":
        base = company_feat if company_feat is not None else Xbase
        Xc = StandardScaler().fit(base[trm.numpy()]).transform(base).astype(np.float32)
        xd = {"company": torch.tensor(Xc), "person": data["person"].x}
        in_dims = {"company": Xc.shape[1], "person": PDIM}
        model = HeteroGNN(list(eid.keys()), in_dims); fwd = lambda: model(xd, eid)
    else:
        sc = StandardScaler().fit(feat[trm.numpy()]); X = torch.tensor(sc.transform(feat).astype(np.float32))
        model = MLP(X.size(1)); fwd = lambda: model(X)
    opt = torch.optim.Adam(model.parameters(), lr=5e-3, weight_decay=5e-4)
    bv, bs, wait = -1, None, 0
    for _ in range(200):
        model.train(); opt.zero_grad()
        loss = F.cross_entropy(fwd()[trm], y[trm], weight=w); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad(): p = F.softmax(fwd(), 1)[:, 1].numpy()
        va = roc_auc_score(y[vam].numpy(), p[vam.numpy()])
        if va > bv: bv, bs, wait = va, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            wait += 1
            if wait >= 30: break
    model.load_state_dict(bs); model.eval()
    with torch.no_grad(): p = F.softmax(fwd(), 1)[:, 1].numpy()
    return metrics(y[tem.numpy()].numpy(), p[tem.numpy()])

Xdeg = np.concatenate([Xbase, degree_feats], 1)
Xcompany_deg = np.concatenate([Xbase, degree_feats], 1)  # GNN company input with degree
real_eid = {et: data[et].edge_index for et in ET_0_11}

rows = []
for seed in SEEDS:
    rows.append({"config": "mlp_intra+degree", "seed": seed, **run("mlp", seed, feat=Xdeg)})
    rows.append({"config": "gnn_0_11+degree", "seed": seed, **run("gnn", seed, eid=real_eid, company_feat=Xcompany_deg)})
    print(f"seed {seed} done", flush=True)

raw = pd.DataFrame(rows)
raw.to_csv(TAB / "13_matched_input_raw.csv", index=False)
summ = raw.groupby("config").agg(AUC_mean=("AUC","mean"), AUC_std=("AUC","std"),
        F1_mean=("F1","mean"), KS_mean=("KS","mean")).round(4)
summ.to_csv(TAB / "13_matched_input.csv")
print("\n=== Matched-input (10 seeds, mask A) ===")
print(summ)
a = raw[raw.config=="mlp_intra+degree"].sort_values("seed")["AUC"].values
b = raw[raw.config=="gnn_0_11+degree"].sort_values("seed")["AUC"].values
t, p = ttest_rel(b, a)
print(f"\nGNN+degree - MLP+degree: delta AUC = {(b-a).mean():+.4f}, paired t p = {p:.4f}")
print(f"  MLP+degree  AUC {a.mean():.4f} +/- {a.std(ddof=1):.4f}")
print(f"  GNN+degree  AUC {b.mean():.4f} +/- {b.std(ddof=1):.4f}")
