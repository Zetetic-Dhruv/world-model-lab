# amazon-reviews-2023

| Field | Value |
|-------|-------|
| Profile | (P=1, T=3, H=2, C=0, Ty=3) |
| Regime | event |
| Origin | natural |
| Stability | churning |
| Member of | [amazon-cross-category](../pairs/amazon-cross-category.json) |
| Source | Hou et al. 2024 (Amazon Reviews 2023); McAuley group UCSD |
| License | Research-use-only |

## Facts

- 571.54M reviews from 54.51M users across 48.19M items, May 1996 – Sept 2023; 33 categories.

## Hypotheses

1. Category-cluster is a better environment unit than raw "category" (e.g., fashion = Clothing+Shoes+Bags), because several of the 33 categories are near-duplicates.
2. Temporal shift within a category dominates cross-category shift, because platform and reviewer behaviour drift faster over 1996–2023 than item semantics differ across categories.

## Intuitions

1. Cross-comparisons between the 2018 and 2023 releases likely exist, alongside cross-domain recommendation (CDR) leaderboards for 2024–25.
2. The McAuley group likely has post-2023 releases extending this collection.

## Unknowns

1. Whether user activity linking categories carries genuine cross-domain signal or merely correlation (people who buy X also buy Y) cannot be determined from the data.
2. The 33 categories are Amazon's commercial taxonomy, not a semantic taxonomy; whether ML-relevant categorical distinctions match commercial ones cannot be decided from the data.

## Boundary

The dataset has strong temporal (T=3) and typed (Ty=3) axes but C=0, so it cannot show within-review compositional shift.

## Representational commitments

- 33-category commercial taxonomy
- 1996–2023 temporal range
- User-item-review-text bundling
- Cross-category transfer, not within-category cold-start
