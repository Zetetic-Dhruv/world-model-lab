# ednet

| Field | Value |
|-------|-------|
| Profile | (P=2, T=3, H=3, C=0, Ty=2) |
| Regime | event |
| Origin | natural |
| Stability | stable |
| Member of | [ednet-cross-subset](../pairs/ednet-cross-subset.json) |
| Source | Choi et al. 2020 AIED (EdNet) |
| License | CC-BY-NC-4.0 |

## Facts

- 131,441,538 interactions from 784,309 students across KT1/KT2/KT3/KT4 subsets (TOEIC-prep Santa platform, 2017–19).

## Hypotheses

1. KT1→KT4 is a *schema-expansion* shift (more action types tracked) rather than a *behavior* shift, because the progression primarily adds instrumentation rather than changing student conduct.
2. Knowledge-tracing models trained on KT1 (simplest) transfer to KT4 (richest) only partially and require re-training for the richer action set.

## Intuitions

1. Knowledge-tracing benchmark surveys from 2024–25 likely cover EdNet.
2. Foundation education-model attempts using EdNet probably exist.

## Unknowns

1. Whether learning-process generalization here transfers to other domains cannot be decided — EdNet is TOEIC-prep-specific (Korean students, English assessment).
2. The Santa platform UI shapes student behavior via UI/UX — UI-as-confounder is not exposed.

## Boundary

Single-platform, single-subject (TOEIC) — cannot evidence cross-domain learning shift.

## Representational commitments

- TOEIC-prep / Korean-student / English-assessment specificity
- Santa platform UI shapes interactions
- KT1–KT4 progression is curator's nested scheme
- Action-type taxonomy fixed by platform
