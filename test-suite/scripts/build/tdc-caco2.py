"""Build tdc-caco2 traces.parquet from PyTDC-downloaded caco2_wang.tab."""
from __future__ import annotations
import sys
from pathlib import Path
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import hashlib

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "raw" / "tdc-caco2"
OUT = ROOT / "derived" / "tdc-caco2"
OUT.mkdir(parents=True, exist_ok=True)


def main():
    df = pd.read_csv(RAW / "caco2_wang.tab", sep="\t")
    df.columns = [c.strip() for c in df.columns]
    print(f"loaded {len(df):,} compounds", file=sys.stderr)

    out = pd.DataFrame({
        "trace_id": ["tdc-caco2:" + hashlib.sha256(s.encode()).hexdigest()[:16] for s in df["Drug"].astype(str)],
        "dataset": "tdc-caco2",
        "entity_id": df["Drug_ID"].astype(str),
        "entity_type": "compound",
        "parent_id": None,
        "parent_type": None,
        "event_type": "permeability_measurement",
        "t": None,
        "t_order": None,
        "env_id": "all",
        "env_axis": "scaffold",
        "split": "train",
        "payload": [{"smiles": s, "log_permeability": float(y)} for s, y in zip(df["Drug"], df["Y"])],
    })
    out_path = OUT / "traces.parquet"
    pq.write_table(pa.Table.from_pandas(out, preserve_index=False), out_path, compression="snappy")
    print(f"wrote {out_path} ({len(out):,} rows)")


if __name__ == "__main__":
    main()
