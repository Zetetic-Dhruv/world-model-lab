# noaa-ghcn-daily

| Field | Value |
|-------|-------|
| Profile | (P=2, T=3, H=3, C=0, Ty=2) |
| Regime | event |
| Origin | natural |
| Stability | live-updating |
| Member of | [ghcn-cross-decade](../pairs/ghcn-cross-decade.json) |
| Source | Menne et al. 2012 (GHCN-Daily) |
| License | WMO Resolution 40 (free-for-research) |

## Facts

- >100k stations / 180 countries / ~1.4B daily values / up to 175-year records; documented thermometer-type changes provide instrument-shift ground truth.

## Hypotheses

1. The correct env unit for cross-station shift is (station × instrument-era) rather than station alone, because documented instrument changes alter the measurement mechanism within a station.
2. Across the century-scale cross-decade shift, instrument and network-density changes dominate over observed climate drift.

## Intuitions

1. The GHCN homogenization literature (Menne, Williams et al.) likely records what is already corrected versus what remains.
2. Cross-network benchmarks (GHCN-D vs CRUTEM vs HadISD vs others) probably exist.

## Unknowns

1. "Instrument-era" is sometimes documented and sometimes not, so whether the corpus admits an implicit-vs-explicit metadata split cannot be determined from the dataset alone.
2. Continuous live-update means the dataset's own temporal stability drifts, and whether pinning to a snapshot is methodologically equivalent to retrospective re-analysis cannot be determined.

## Boundary

H=3 (deep) but Ty=2 (limited variables — temp, precip mostly); station-density biased toward developed countries.

## Representational commitments

- Station-as-entity (network-topology imposed)
- Daily grain (sub-daily lost)
- 6 standard elements selected
- Homogenization corrections applied to some stations
