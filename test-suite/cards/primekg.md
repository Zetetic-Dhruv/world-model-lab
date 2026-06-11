# primekg

| Field | Value |
|-------|-------|
| Profile | (P=0, T=0, H=2, C=0, Ty=3) |
| Regime | KG |
| Origin | natural (integrated from 20 source DBs) |
| Stability | churning |
| Pairs with | [hetionet](../manifests/hetionet/) |
| Member of | [primekg-source-split](../pairs/primekg-source-split.json) |
| Source | Chandak et al. 2023 *Scientific Data*; [DOI 10.7910/DVN/IXA7BM](https://doi.org/10.7910/DVN/IXA7BM) |
| License | MIT |

## Facts

- 129,375 nodes / 4,050,249 edges / 10 node types / 30 edge types / 20 source resources; TxGNN (Huang 2023 *Nat Med*) zero-shot drug-repurposing baseline.

## Hypotheses

1. The typed-edge conditional distribution is not genuinely 20-way distinct across source DBs: sources mostly agree, collapsing the effective environment cardinality to ~3–5.
2. TxGNN's held-out-disease shift manifests only at the node-neighborhood level, not at the edge-type level.

## Intuitions

1. The full TxGNN holdout protocol and replication attempts likely exist in the literature.
2. Head-to-head PrimeKG vs Hetionet benchmarks probably exist for 2024–25.
3. PrimeKG version-diff papers (Dec 2023 update vs original) likely exist.

## Unknowns

1. "Disease" is an unstable cross-resource primitive — different source DBs use incompatible disease ontologies for the same physical phenomenon, and there is no established label for this ontology drift across resources as a shift type.
2. Whether the integration step *creates* or *averages out* the cross-resource shift cannot be determined from the integrated KG alone.

## Boundary

T=0 (snapshot, no dynamics) — a version-diff cannot supply this, since the deltas are too discrete to constitute a temporal axis.

## Representational commitments

- Integrated ontology — disease/drug/gene equivalences resolved by curation
- 10-node-type × 30-edge-type closure is the curator's selection
- Source-resource provenance preserved per edge — the environment handle hinges on this
- PrimeKG's specific version is one snapshot of evolving integration
