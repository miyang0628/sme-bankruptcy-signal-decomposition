# Relational Signal Attribution for SME Bankruptcy Prediction from Non-Financial Information

Code and evaluation protocol accompanying a manuscript currently under
double-blind peer review. **All author- and institution-identifying information
has been removed for anonymity.** Please do not attempt to de-anonymize the
authors from commit metadata or otherwise.

## What this repository contains

A controlled, leakage-free study of where the predictive signal comes from when
predicting small- and medium-sized enterprise (SME) bankruptcy from
**non-financial information only** — lawsuit events and an inter-firm
relationship graph, with no financial ratios. Rather than proposing a new
architecture, we treat a heterogeneous graph neural network as a measurement
instrument and attribute the predictive signal under a strict protocol.

The study is organized around three questions:

- **RQ1 — Predictability.** Can non-financial lawsuit and relationship data
  predict bankruptcy above a basic-information floor and above strong non-graph
  tabular competitors (logistic regression, MLP, LightGBM, XGBoost)?
- **RQ2 — Decomposition.** Holding node features fixed and varying only message
  passing, how is the signal split between intra-risk (a firm's own lawsuit
  record) and contagion (relational propagation)? An edge-shuffle permutation
  null, a DeLong test, and bootstrap confidence intervals test whether the
  contagion uplift reflects real structure. A degree-destroying and a
  degree-preserving (configuration-model) null, plus a connection-count baseline,
  separate a genuine *connectivity* effect from specific *wiring*; a
  **matched-input** test then gives both the non-graph model and the GNN the same
  connection counts to isolate what the wiring adds once connectivity is
  controlled.
- **RQ3 — LLM enrichment.** Does enriching the *coded* lawsuit fields with
  LLM-derived semantics and an explicit risk direction beat simple counts? We
  report a controlled, power-analyzed negative result.

A decision-analysis notebook translates the relational signal into calibration,
cost-sensitive expected cost, and net-benefit terms.

## Repository structure

```
data/
  external/ComRisk/   SMEsD dataset and the original method code (see Data)
  processed/          built graph tensors and frozen LLM-derived code vectors
notebooks/
  00_build_graph.ipynb              heterogeneous graph + dual leakage masks
  01_eda.ipynb                      graph statistics, edge counts, label balance
  02_baselines.ipynb                RQ1 floor + LightGBM/XGBoost competitors
  03_architecture_comparison.ipynb  SAGE vs relation-attention (instrument validity)
  04_decomposition.ipynb            RQ2 + permutation null + DeLong + bootstrap
  05_llm_enrichment.ipynb           RQ3 ablation + power/equivalence analysis
  06_decision_analysis.ipynb        calibration, expected cost, decision curves
  07_robustness.ipynb               consolidation, mask A/B, stratified analysis
  08_confound_inference.ipynb       attention-operator decomposition + inference
  09_protocol_generalization.ipynb  cross-dataset discrimination (Taiwan)
  10_decision_sweep.ipynb           base-rate / cost-ratio decision sweeps
  11_round2_robustness.ipynb        person-pooling + degree-null robustness
  12_degree_baseline.ipynb          connection-count baseline + 50-draw null
scripts/                            revision (round-3) add-on experiments
  matched_input.py                  matched-input test: GNN vs MLP, both given counts
  delong_fix.py                     per-seed DeLong for the matched-input test
  degree_cost.py                    decision analysis for the recommended degree model
  confound_check.py                 degree vs firm size/age (confound check)
results/
  figures/            grayscale figures (png + pdf, 600 dpi)
  tables/             result tables (csv); 13_* matched-input, 14_* degree-model cost
```

## Data

The study uses the public **SMEsD** dataset, released with the original method
paper and obtained from its authors' repository (`ComRisk`). The dataset is used
under the terms set by its authors and is **not redistributed** here beyond the
copy required to reproduce the build; users should obtain it from the original
source. No financial ratios are used at any point.

## Reproducing the results

Requirements: Python 3.11+, `torch`, `torch_geometric`, `scikit-learn`,
`lightgbm`, `xgboost`, `pandas`, `numpy`, `scipy`, `seaborn`, `matplotlib`.

```
pip install torch torch_geometric scikit-learn lightgbm xgboost \
            pandas numpy scipy seaborn matplotlib
```

Run the notebooks in order (`00` → `12`). Each notebook is self-contained in a
single cell and writes its tables to `results/tables/` and figures to
`results/figures/`. Notebook `00` must run first to build the shared graph.

The `scripts/` directory holds the revision (round-3) add-on experiments, which
reuse the model definitions, feature construction, and training protocol of the
notebooks and read the same `data/processed/smesd_hetero.pt`. After `00` has built
the graph, they can be run directly:

```
python scripts/matched_input.py     # 13_matched_input*.csv  (GNN vs MLP, both given counts)
python scripts/delong_fix.py        # 13_matched_delong.csv  (per-seed DeLong)
python scripts/degree_cost.py       # 14_degree_*.csv        (degree-model decision analysis)
python scripts/confound_check.py    # degree vs firm size/age correlation
```

These produce the matched-input comparison (the GNN given the same connection
counts reaches AUC 0.818 vs the degree model's 0.809; DeLong non-significant in all
10 seeds) and the decision analysis for the recommended degree model (isotonic
Brier 0.180; about 23% expected-cost saving at a 5% base rate, no higher than the
graph model's cost at any swept base rate).

**LLM-derived features are frozen.** The semantic embeddings and risk-direction
scalars for the 19 lawsuit codes are content-hashed and cached in
`data/processed/`, so RQ3 (`05`) reproduces exactly **without any API calls**.
Regenerating them from scratch would require an API key; this is not needed to
reproduce any reported number.

## Notes

- The strict split (mask A) removes companies that recur across the time-based
  train/validation/test boundary and is the primary evaluation; the as-shipped
  split (mask B) is reported for reference.
- All preprocessing (log transforms, standardization, PCA) is fit on training
  companies only.
- `.env` and any credentials are intentionally excluded from this repository.

## License

To be released under an open-source license upon publication.
