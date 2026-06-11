# weatherbench2

| Field | Value |
|-------|-------|
| Profile | (P=3, T=3, H=2, C=1, Ty=3) |
| Regime | array (Zarr-native) |
| Origin | **mixed** (ERA5 reanalysis = observation + assimilation) |
| Stability | churning |
| Member of | [weatherbench-temporal](../pairs/weatherbench-temporal.json) |
| Source | Rasp et al. 2024 JAMES |
| License | CC-BY-4.0 |

## Facts

- ERA5 reanalysis 1959–2023, 6-hourly, 0.25°; canonical splits (≤2018 / 2019 / 2020); GraphCast/Pangu/FourCastNet/IFS baselines.

## Hypotheses

1. The train/val/test temporal split (≤2018/2019/2020) is a *natural* shift driven by climate-change baseline drift rather than a mere *convenience* split.
2. ML weather prediction learns ERA5-specific artifacts rather than generalizing from ERA5 training to IFS-analyses test, so cross-source transfer within climate science degrades.

## Intuitions

1. A WeatherBench 2 leaderboard with 2024–25 results (GraphCast, GenCast, Aurora) likely exists, alongside operational deployment papers (ECMWF AIFS).
2. Cross-reanalysis benchmarks comparing ERA5 with MERRA-2 and JRA-3Q probably exist.

## Unknowns

1. ERA5 is itself a data-assimilation product blending observation and model; whether OOD generalization on ERA5 measures *physical* generalization or *reanalysis-product* generalization cannot be determined from the data alone.
2. The 0.25° resolution + 6-hour grain is a fixed representational commitment; whether the relevant atmospheric dynamics are captured at this resolution cannot be determined from ERA5 itself.

## Boundary

Array-regime data — no traces.parquet; manifest declares Zarr-native. The temporal shift is the dominant axis; cross-source (ERA5↔IFS) is a separate pair.

## Representational commitments

- ERA5 reanalysis (specific assimilation model)
- 0.25° / 6-hourly grain
- 13 pressure levels
- Multi-variable bundling
