import TestSuite

/-!
# Examples.CamelsDk

CAMELS-DK as a concrete Lean4 manifest value, mirroring
`manifests/camels-dk/manifest.json`. The fact that this file typechecks
is the schema-conformance test: every JSON field has a corresponding
Lean4 structure field, and the type system caught any drift between
the two.

The `example : Axioms.WellFormed camelsDk := by decide` verifies the
manifest satisfies the content-bearing axioms (A₁, A₄, A₆, A₇, A₁₀).
-/

open TestSuite

/-- CAMELS-DK manifest (Liu et al. 2025 ESSD, doi:10.5194/essd-17-1551-2025). -/
def camelsDk : DatasetManifest := {
  schemaVersion := "0.1.0"
  name          := "camels-dk"
  version       := "2025.04.essd"
  snapshotDate  := "2026-05-26"
  source := {
    url       := "https://dataverse.geus.dk/dataset.xhtml?persistentId=doi:10.22008/FK2/AZXSYP"
    doi       := some "10.22008/FK2/AZXSYP"
    hostType  := "geus-dataverse"
    license   := "CC0-1.0"
    stability := .stable
  }
  domain     := "earth-science.hydrology"
  regime     := .event
  -- BP₁/BP₂: the regime-dependent schema. The `.event` constructor pins the
  -- regime, and `A6_RegimeClassified`/`regimeMatchesSchema` checks it equals
  -- the flat `regime` field above. Entity/event/static types are kept as
  -- name-lists (the JSON's per-field keys/units are out of mirror scope, BP₇).
  schema := .event {
    entityTypes      := ["catchment"]
    eventTypes       := ["daily_observation"]
    staticAttributes := ["drainage_area_km2", "mean_elevation_m",
                          "land_use_class", "soil_type", "geology_class"]
  }
  -- BP₄: the dependent payload. `.natural` carries `Unit`; only `.mixed`
  -- could carry a blend `String`. `(natural, some "…")` is unconstructible.
  dataOrigin := .natural
  asymmetryProfile := {
    process       := 3
    temporal      := 3
    hierarchical  := 2
    compositional := 0
    typed         := 2
  }
  envAxis := {
    name           := "country"
    field          := some "country_code"
    values         := ["DK"]
    shiftType      := .covariate
    shiftMechanism := "geographic-cohort with potential mechanism component (snow-melt and groundwater regimes differ from US median)"
    pairWith       := ["camels-us", "camels-ind"]
  }
  oodProtocol := {
    name   := "pub-cross-country"
    ref    := "Liu et al. 2025 ESSD doi:10.5194/essd-17-1551-2025; Kratzert et al. 2019 HESS (PUB methodology)"
    splits := [
      { envId := "us", purpose := "train",    ref := some "camels-us:traces" },
      { envId := "dk", purpose := "ood-test", ref := some "self:traces" }
    ]
  }
  scale := {
    entities   := some 3330
    events     := none
    rawBytes   := some 2934993414
    nEnvLevels := some 1
  }
  epistemicState := {
    facts := ["3,330 Danish catchments at daily resolution, 1989-2019; schema-aligned with CAMELS-US per Liu et al. 2025 ESSD."]
    hypotheses := [
      "The matched (P=3, T=3) structural profile of CAMELS-US and CAMELS-DK predicts transfer-method ordering, consistent with the published PUB ranking.",
      "The country axis is a mechanism shift rather than a pure covariate shift: snow-melt and groundwater regimes in Denmark differ systematically from the US median.",
      "The cross-country generalization gap is predictable from the datasets' typed-process structure alone, before any model training."
    ]
    intuitions := [
      "Nearing et al.'s 2024 Nature global-flood model likely underperforms on CAMELS-DK specifically.",
      "A combined US+DK+IND meta-analysis ('META-CAMELS') likely already exists or is emerging in the 2024-25 literature."
    ]
    unknowns := [
      "Measurement-network change and climate change cannot be disentangled as joint sources of within-DK temporal drift.",
      "Whether schema-alignment with CAMELS-US lossily projects DK-specific signal."
    ]
    boundary := "C=0 — the dataset is silent on compositional shift; its contribution is the country (cross-population) axis, not expanded structural coverage."
  }
  representationalCommitments := [
    "Catchment partition is an imposed cartographic decision, not discovered from the data",
    "Daily temporal resolution coarsens away sub-daily mechanism (storm hydrograph shape)",
    "Forcing variables chosen to be Daymet-equivalent — biases the schema toward measurements the US release also has",
    "Static attribute set is schema-aligned with US, potentially lossy for groundwater-driven DK hydrology"
  ]
  memberOf := ["camels-cross-country"]
  -- BP₅: refined checksum. `mkChecksum` discharges the `isSha256` proof by
  -- `native_decide` at construction; a malformed digest fails the build.
  checksums := [
    mkChecksum
      "snapshots/camels-dk/2025.04.essd/camels-dk-raw.tar.gz"
      "sha256:c2678c647c26fe0ecf651a4a130bb506a3f0cee99703ffc50feec30597c789df"
  ]
  baselines := [{
    name   := "EA-LSTM in-country PUB (US reference)"
    ref    := "Kratzert et al. 2019 HESS"
    metric := "median_nse"
    value  := 0.74
    env    := some "us"
  }]
}

-- The reframed annotation strings are long; `decide` reduces `String` equality
-- character by character, so raise the recursion limit for the checks below.
set_option maxRecDepth 4000

/-- **Test 1**: CAMELS-DK is `natural`, not synthetic (A₁). -/
example : Axioms.A1_NaturalShift camelsDk := by decide

/-- **Test 2**: CAMELS-DK's boundary is non-empty (A₄). -/
example : Axioms.A4_KUInManifest camelsDk := by decide

/-- **Test 3**: the declared `regime` (`event`) matches the `schema` variant (A₆). -/
example : Axioms.A6_RegimeClassified camelsDk := by decide

/-- **Test 4**: CAMELS-DK's source URL, license, and host type are populated (A₇). -/
example : Axioms.A7_ProvenancePreserved camelsDk := by decide

/-- **Test 5**: CAMELS-DK has representational commitments listed (A₁₀). -/
example : Axioms.A10_RepresentationalCommitments camelsDk := by decide

/-- **Test 6**: CAMELS-DK satisfies all content-bearing axioms — well-formed. -/
example : Axioms.WellFormed camelsDk := by decide

/-- **Test 7**: CAMELS-DK is materialized (has at least one checksum). -/
example : camelsDk.Materialized := by decide

/-- **Test 8**: CAMELS-DK's profile is (3,3,2,0,2). -/
example : camelsDk.asymmetryProfile.totalScore = 10 := by decide

/-- **Test 9**: CAMELS-DK's regime is `event`. -/
example : camelsDk.regime = Regime.event := rfl
