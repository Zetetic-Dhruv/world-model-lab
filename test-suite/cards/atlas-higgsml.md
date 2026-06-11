# atlas-higgsml

| Field | Value |
|-------|-------|
| Profile | (P=2, T=0, H=2, C=2, Ty=2) |
| Regime | event |
| Origin | **simulator** (Geant4 + ATLAS reconstruction) |
| Stability | stable |
| Member of | [higgsml-ams](../pairs/higgsml-ams.json) |
| Source | Adam-Bourdarios et al. 2015 (HiggsML challenge); CERN Open Data |
| License | CC0-1.0 |

## Facts

- 818,238 simulated ATLAS H→ττ events; AMS metric with reweighting simulating realistic analysis sample.

## Hypotheses

1. AMS-optimized classification overfits to the specific reweighting scheme rather than generalizing, because methods tuned to one weight distribution track that distribution's particulars.
2. Simulator-to-data gaps for HiggsML and LHC-Olympics differ, because ATLAS-specific reconstruction shapes the gap beyond what generic CMS-style events show.

## Intuitions

1. A HiggsML retrospective documenting which methods were deployed in real ATLAS analyses likely exists, alongside CERN Open Data follow-ups using subsequent ATLAS releases.
2. Recent (2024–25) ATLAS measurements that use HiggsML-era technology with real data probably provide a direct sim-to-real test.

## Unknowns

1. The AMS metric is itself a representational commitment (Asimov significance under a specific signal/background assumption); whether AMS-optimization tracks physics-discovery-utility outside this assumption cannot be determined from the data.
2. The dataset's reweighting scheme implicitly defines what "shift" means; whether this captures real analysis-time shift cannot be decided from the data.

## Boundary

Being simulator-generated and limited to a single channel (H→ττ only), the dataset is silent on cross-channel transfer.

## Representational commitments

- Geant4 + ATLAS reconstruction chain
- 30 high-level kinematic features (curator's selection)
- AMS metric framing
- Single-channel H→ττ
