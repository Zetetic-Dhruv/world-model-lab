# lhc-olympics-2020

| Field | Value |
|-------|-------|
| Profile | (P=2, T=0, H=2, C=2, Ty=2) |
| Regime | event |
| Origin | **simulator** (PYTHIA + DELPHES) |
| Stability | stable |
| Member of | [lhc-olympics-blind-bb](../pairs/lhc-olympics-blind-bb.json) |
| Source | Kasieczka et al. 2021 RPP |
| License | CC-BY-4.0 |

## Facts

- 1M background + 1M signal R&D events + 3 black-box datasets with unknown injected anomalies.

## Hypotheses

1. Black-box anomaly scores do not correlate with R&D training: the blind injection succeeds in being method-invariant.
2. Anomaly detection does not generalize *across* the 3 black boxes (BB1→BB2, BB3); anomaly discovery is dataset-specific.

## Intuitions

1. LHC Olympics retrospective papers from 2020–2024 likely exist.
2. ATLAS/CMS anomaly detection on actual Run 2/3 physical data probably exists.

## Unknowns

1. MC-to-data shift in HEP is *compound* — parton showers + hadronization + detector response + pileup — but the black boxes don't decompose, and there is no established label for a "compound shift mechanism".
2. Detector-side and physics-side anomalies share data signature with categorically different generation processes.

## Boundary

Simulated data; C=2 is event-level only, not within-particle.

## Representational commitments

- PYTHIA + DELPHES simulation chain
- Up to 700 particles per event (padding/truncation)
- Jet-substructure representation choice
- BB1/BB2/BB3 specific anomaly types are the organizers' choice
