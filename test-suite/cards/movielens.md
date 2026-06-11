# movielens

| Field | Value |
|-------|-------|
| Profile | (P=0, T=2, H=1, C=0, Ty=2) |
| Regime | event |
| Origin | natural |
| Stability | churning |
| Member of | [movielens-cross-snapshot](../pairs/movielens-cross-snapshot.json) |
| Source | Harper & Konstan 2015 TIIS (MovieLens) |
| License | GroupLens-research |

## Facts

- ML-1M, 10M, 20M, 25M, 32M snapshots at different time windows.

## Hypotheses

1. Cross-snapshot shift on MovieLens is *sample-size* driven (small ML-1M differs from large ML-32M for statistical reasons) rather than genuine *temporal* user-behavior drift.
2. The genre taxonomy is stable enough across snapshots to factor out rather than drifting.

## Intuitions

1. Literature on MovieLens stability likely exists.
2. Recent recommendation benchmark surveys probably exist.

## Unknowns

1. Whether MovieLens-research is useful data or a *legacy artifact* — heavy use may have shaped which methods "work" via publication bias on the benchmark, and there is no established label for "benchmark overuse as shift confounder".

## Boundary

Recommendation-only; classic but limited scale and modality.

## Representational commitments

- GroupLens curation
- 5 snapshot sizes
- Star-rating scale
- Genre taxonomy is GroupLens's choice
