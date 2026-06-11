# tdc-suite

| Field | Value |
|-------|-------|
| Profile (suite-level) | (P=0, T=1, H=1, C=3, Ty=3) |
| Regime | event |
| Origin | natural |
| Stability | churning |
| Member of | [tdc-cross-task-suite](../pairs/tdc-cross-task-suite.json) |
| Source | Huang et al. 2021 Nature Chemical Biology |
| License | MIT |

> **Suite** — 66+ datasets across ADMET (22 endpoints), DTI, bioactivity, toxicity. [tdc-caco2](./tdc-caco2.md) broken out as a core entry within the suite.

## Facts

- 66+ datasets; 4.26M compounds, 34k genes, 60k peptides, 1.99M reactions; built-in scaffold / cold-start / temporal splits.

## Hypotheses

1. No single ADMET *generalist* method wins across all 22 endpoints; per-endpoint specialists dominate.
2. The temporal (publication-year) split gives a more honest estimate of real-world generalization for novel compounds than the scaffold split.

## Intuitions

1. A current TDC leaderboard with 2024–25 state-of-the-art results likely exists.
2. Cross-TDC molecular-foundation-model results have probably been published.
3. TDC has likely expanded to new task types beyond the original release.

## Unknowns

1. TDC bundles heterogeneous tasks (small-molecule properties, peptide, DTI, reactions) as "drug discovery"; whether they cohere structurally is presupposed by curation, not justified.
2. The "cold-start" split definition (novel compound / target / assay) differs per task, and inter-task consistency cannot be determined from the suite.

## Boundary

The suite spans very heterogeneous content; a single suite-level method ordering does not apply across the bundled tasks.

## Curation

Huang et al. (Harvard/Stanford/MIT), Nature Chemical Biology 2021 — per-task split rules (scaffold/cold-start/temporal); motivation: standardized therapeutic-task ML benchmarks.
