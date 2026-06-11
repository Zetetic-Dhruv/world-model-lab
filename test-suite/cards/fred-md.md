# fred-md

| Field | Value |
|-------|-------|
| Profile | (P=1, T=3, H=2, C=0, Ty=2) |
| Regime | event |
| Origin | natural |
| Stability | live-updating |
| Member of | [fred-regime-shift](../pairs/fred-regime-shift.json) |
| Source | McCracken & Ng 2016 JBES; St. Louis Fed FRED |
| License | Public domain (US govt work) |

## Facts

- 134+ monthly US macro series 1959–present + ALFRED vintage archive (real-time snapshots by release date).

## Hypotheses

1. {pre-2008, 2008–2020, post-2020} are three distinct envs rather than two, because the post-2020 period reflects a persistent shift and not merely a transient COVID episode.
2. Vintage-as-env (released-by date) is categorically different from measurement-date-as-env — the same data under a different shift type.

## Intuitions

1. Goulet Coulombe 2022 and Medeiros 2021 likely have follow-ups extending to post-2024 data.
2. A nowcasting literature using the ALFRED vintage archive likely exists.
3. Macroeconomic foundation-model work from 2024–25 may exist, if any.

## Unknowns

1. Whether "regime" is a discrete latent state (recession / expansion / ZLB) or a continuous latent variable cannot be determined — the data does not commit, but every method does, and there is no established label for "implicit regime labeling chosen by the method, not the data".
2. Series-selection (the curated 134) is itself a representational commitment that biases all comparisons.

## Boundary

C=0; aggregation-decomposition (national inflation → sector inflation) is implicit in the variable list, not exposed as a compositional handle.

## Representational commitments

- 134-series curated selection (McCracken-Ng)
- Monthly grain fixed
- Vintage-as-snapshot policy (ALFRED)
- 8-category grouping is a researcher convention
