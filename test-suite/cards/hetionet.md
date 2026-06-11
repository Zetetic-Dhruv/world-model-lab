# hetionet

| Field | Value |
|-------|-------|
| Profile | (P=0, T=0, H=2, C=0, Ty=3) |
| Regime | KG |
| Origin | natural (29 source DBs) |
| Stability | stable (frozen 2017) |
| Pairs with | [primekg](../manifests/primekg/) |
| Member of | hetionet-source-split · primekg-vs-hetionet-cross-kg |
| Source | Himmelstein et al. 2017 eLife |
| License | CC0-1.0 |

## Facts

- 47k nodes / 2.25M edges / 24 edge types / 29 source resources; Project Rephetio drug-repurposing canonical.

## Hypotheses

1. The integration choices that separate Hetionet from PrimeKG change *which methods* generalize, not merely by how much.
2. Hetionet's 29 source resources are sufficient to predict the cross-source generalization gap from graph structure alone, as is also expected for PrimeKG.

## Intuitions

1. Clinical validation of Rephetio's 2017 predictions likely exists in 2017–2025 follow-up work.
2. Cross-KG transfer benchmarks (train-Hetionet / test-PrimeKG and reverse) probably exist in the 2024–25 literature.

## Unknowns

1. Hetionet's frozen-2017 snapshot vs PrimeKG's Dec-2023 update is a *discrete version axis* the schema does not surface; there is no established label for "version-as-env" at the corpus level.
2. The 29 source resources are not independent — many curate each other (DrugBank ← PubMed ← …); env-axis independence is presupposed but unverified.

## Boundary

T=0; the cross-KG comparison with PrimeKG is the pair axis, not internal.

## Representational commitments

- Integrated 2017 snapshot, frozen
- 11-node-type × 24-edge-type closure
- Source-resource provenance preserved per edge
- Metapath abstraction is Project Rephetio's choice
