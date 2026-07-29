# SME Bankruptcy Prediction via Intra-Risk / Contagion Signal Decomposition

Code and experiments for a study on predicting small- and medium-enterprise
(SME) bankruptcy from **non-financial information only** — lawsuit events and an
enterprise relationship graph — with no financial ratios. The work decomposes
predictive signal into **intra-risk** (a firm's own lawsuit history) and
**contagion risk** (propagation through inter-firm relations), and tests whether
LLM-based semantic enrichment of lawsuit codes adds value.

> **Note:** This repository is anonymized for double-blind peer review. Author,
> affiliation, funding, and contact information are intentionally omitted.

## Summary of findings

All results use a strict, leakage-free temporal split (bankruptcy year:
train 2014–2018 / validation 2019 / test 2020–2021), reported as the mean ±
standard deviation over 10 random seeds on the test set (mask A).

| Research question | Setup | Test AUC |
|---|---|---|
| **RQ1** — Can non-financial data predict bankruptcy? | Basic-info floor (logistic regression) | 0.624 |
| | Heterogeneous GNN (lawsuit + relations) | **0.765 ± 0.012** (ΔAUC +0.141) |
| **RQ2** — Where does the signal come from? | Intra-only (no graph) | 0.687 |
| | + Contagion (relations 0–11) | **0.772 ± 0.008** (ΔAUC +0.085, *p* < 0.001) |
| | + Contagion (relations 0–19) | 0.765 ± 0.012 (no significant gain) |
| **RQ3** — Does LLM enrichment of lawsuit codes help? | Counts vs + semantic + direction | +0.003 (*p* = 0.63, **not significant**) |

**Takeaways.** (1) Non-financial lawsuit and relationship data predict SME
bankruptcy well above a basic-information floor. (2) The dominant source of
predictive value is **contagion** — relational propagation — not the individual
firm's lawsuit features. (3) LLM-based semantic/direction enrichment of coded
lawsuit fields yields **no robust improvement**, a negative result that
reinforces the central finding: value is *relational*, not in per-lawsuit
semantics. Widening the relation set to include the sparse financial relations
(12–19) adds no significant signal, consistent with the original dataset paper's
choice to use only relations 0–11.

## Dataset

Experiments use the public **SMEsD** dataset (3,976 Chinese SMEs and related
persons, 2014–2021; lawsuit events + a multiplex enterprise knowledge graph +
bankruptcy labels; no financial ratios). The dataset is released by the authors
of the original method paper and is **not redistributed here** — see
`00_data_acquisition` for how to obtain it.

The pipeline also touches several auxiliary public bankruptcy datasets used in a
broader research portfolio; only SMEsD is required to reproduce the results
above.

## Repository structure

```
notebooks/
  00_data_acquisition      # download / verify datasets
  01_build_graph           # parse SMEsD, build the PyG heterogeneous graph
  02_baseline_basic        # condition (1): basic-info floor  (RQ1)
  03_gnn_reproduce         # condition (2): hetero-GNN over the graph (RQ1)
  04_llm_code_enrich       # LLM semantic + risk-direction embeddings of codes
  05_gnn_plus_enrich       # ablation: counts vs +semantic vs +direction (RQ3)
  06_signal_decomp         # intra vs contagion decomposition (RQ2, core)
  07_case_study            # qualitative contagion illustration (figures)
data/
  raw/        external/    processed/    # (git-ignored; created locally)
artifacts/
  tables/     figures/     models/       # results, plots, checkpoints
paper_C/                                  # publication-ready figures (pdf/eps/png)
```

## Method overview

1. **Graph construction.** The enterprise knowledge graph is built as a PyG
   `HeteroData` object with `company` and `person` node types and 20 typed
   relations. Company nodes carry 3 basic attributes (registered capital,
   paid-in capital, setup time) plus a 20-dimensional lawsuit-aggregate feature
   (cause / court / result-category counts + mean timestamp). Person nodes use
   pretrained metapath2vec embeddings.
2. **Strict evaluation.** A small number of firms recur across temporal splits;
   we build a strict, disjoint mask (**mask A**, each firm kept only in its
   latest split) as the primary evaluation, and retain the as-shipped mask
   (**mask B**) for reference. All preprocessing (log-scaling of capital,
   standardization, PCA) is fit on training rows only.
3. **LLM enrichment (RQ3).** Each lawsuit code (11 causes, 4 court levels,
   4 result categories) is described by an LLM in terms of its bankruptcy-risk
   implication; the description is embedded, and an explicit signed
   risk-direction scalar is extracted. These are pooled per firm and appended to
   node features. Embeddings are cached and content-hash **frozen** for
   reproducibility.
4. **Signal decomposition (RQ2).** Node features are held fixed while only
   message passing is varied: no graph (intra) → relations 0–11 → relations
   0–19, isolating the contagion contribution.

## Reproducing the results

```bash
# 1. environment (CPU is sufficient; the graph is small, ~4k nodes)
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install torch_geometric ucimlrepo kaggle pandas pyarrow scikit-learn \
            matplotlib networkx openai python-dotenv

# 2. obtain SMEsD (see notebooks/00_data_acquisition), place under data/external/

# 3. run notebooks 00 -> 07 in order
```

**LLM step (04).** Requires an API key in a local `.env` file
(`OPENAI_API_KEY`, plus model names). The 04 notebook caches all calls and
freezes the resulting embeddings via a SHA-256 content hash, so downstream
notebooks consume identical embeddings and results do not drift. The `.env` file
is git-ignored and never committed.

**Determinism.** Results are averaged over 10 seeds. The GNN is seeded per run;
PCA and scalers are fit on training data only. Raw per-seed outputs are written
alongside the aggregated tables under `artifacts/tables/`.

## Notes and limitations

- This is an **equivalent** modern-PyG reimplementation of the original
  heterogeneous GNN, not a bit-exact reproduction of the original method paper;
  it is intended to measure uplift over an intra-only floor, not to match the
  original's exact numbers.
- The RQ3 enrichment result is a deliberately reported **negative result**; the
  frozen embeddings and 10-seed protocol are provided so it can be verified.
- Real firm names (Chinese) in the graph are identifiers only, used for
  case-study figures under an anonymization flag; they are not model features.

## License

Code released under the MIT License. The SMEsD dataset is subject to the terms
set by its original authors and is not included in this repository.
