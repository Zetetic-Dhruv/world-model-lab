"""Build atlas-higgsml traces.parquet from data.csv.gz.

Each row = one ATLAS H→ττ reconstructed event. KaggleSet provides the
train/val/test split native to the HiggsML challenge.
"""
from __future__ import annotations
import gzip, sys, hashlib
from pathlib import Path
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "raw" / "atlas-higgsml"
OUT = ROOT / "derived" / "atlas-higgsml"
OUT.mkdir(parents=True, exist_ok=True)

KAGGLESET_MAP = {"t": "train-nominal", "b": "val-public", "v": "val-private", "u": "unused"}
PURPOSE_MAP   = {"t": "train", "b": "val", "v": "test", "u": "unused"}


def main():
    print(f"Reading {RAW / 'data.csv.gz'} …", file=sys.stderr)
    with gzip.open(RAW / "data.csv.gz", "rt") as f:
        df = pd.read_csv(f)
    print(f"  loaded {len(df):,} events", file=sys.stderr)

    feature_cols = [c for c in df.columns if c.startswith(("DER_", "PRI_"))]
    df["env_id"] = df["KaggleSet"].map(KAGGLESET_MAP).fillna("unknown")
    df["split"] = df["KaggleSet"].map(PURPOSE_MAP).fillna("unused")

    out = pd.DataFrame({
        "trace_id": ["atlas-higgsml:" + str(eid) for eid in df["EventId"]],
        "dataset": "atlas-higgsml",
        "entity_id": df["EventId"].astype(str),
        "entity_type": "event",
        "parent_id": None,
        "parent_type": None,
        "event_type": "atlas_reconstructed_event",
        "t": None,
        "t_order": df["EventId"].astype(int),
        "env_id": df["env_id"],
        "env_axis": "reweighting",
        "split": df["split"],
        "payload": [
            {**{c: row[c] for c in feature_cols}, "label": row["Label"], "weight": row["Weight"], "kaggle_weight": row["KaggleWeight"]}
            for _, row in df.iterrows()
        ],
    })

    out_path = OUT / "traces.parquet"
    print(f"Writing {out_path} ({len(out):,} rows)…", file=sys.stderr)
    pq.write_table(pa.Table.from_pandas(out, preserve_index=False), out_path, compression="snappy")
    print(f"DONE. events={len(out):,} envs={df['env_id'].nunique()}")


if __name__ == "__main__":
    main()
