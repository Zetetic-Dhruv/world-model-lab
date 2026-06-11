# Test Suite scripts

Three subdirectories, one role each. All scripts are stubs; populated as datasets are materialized.

## `fetch/<dataset>.py`

Download dataset from its canonical host (URL pinned in `manifests/<dataset>/manifest.json`).

- Verifies `source.url` is reachable.
- Downloads to object storage under `snapshots/<dataset>/<version>/` (or local `raw/` during dev).
- Writes SHA256 into `manifest.checksums` and commits the manifest delta.

For datasets at canonical hosts with stable URLs, `fetch` may be a no-op that simply records the canonical URL. We do not mirror what's already reproducibly hosted.

## `validate/<dataset>.py`

Checksum + schema conformance.

- Verifies all `manifest.checksums` entries.
- Validates `manifest.schema` declarations against actual file contents.
- Validates the JSON manifest itself against `schemas/manifest.schema.json`.
- Validates `traces.parquet` (if materialized) against `schemas/trace.schema.json`.
- Validates `edges.parquet` (if materialized) against `schemas/edges.schema.json`.

Exit code 0 ⇒ schema + checksum conform.

## `build/<dataset>.py`

Coerce raw data to the regime-appropriate canonical form.

- **regime=event** → `traces.parquet` matching `trace.schema.json`.
- **regime=KG** → `edges.parquet` (and optional `nodes.parquet`) matching `edges.schema.json`.
- **regime=blob** → metadata-only `traces.parquet` (one row per file, payload has path + labels).
- **regime=array** → no canonical form; manifest declares Zarr-native; scripts here only validate structure.

Output: object storage under `derived/<dataset>/`.

## Cross-dataset utilities

To be added under `scripts/`:

- `materialize_splits.py` — given a pair manifest, materialize the environment partitions.
- `route.py` — given a method's target profile, list profile-matching pairs.
- `bridge.py` — compute cross-domain transfer diagnostics between two pair members.
- `diff.py` — diff a manifest's `epistemic_state` against its previous Git commit (the audit-trail reader).
