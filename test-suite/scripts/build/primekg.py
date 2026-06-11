"""Build PrimeKG edges.parquet + nodes.parquet from raw kg.csv + nodes.tab.

Conforms to schemas/edges.schema.json. Source columns mapped per the manifest's
schema.edge_types declaration.

Representational commitment: PrimeKG's CSV has x_source + y_source per row (node
sources), but no explicit edge_source. We use 'x_source||y_source' as the
source_resource handle to preserve both endpoints' provenance — flagged in the
manifest's representational_commitments.
"""
from __future__ import annotations
import hashlib
import sys
from pathlib import Path
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "raw" / "primekg"
OUT = ROOT / "derived" / "primekg"
OUT.mkdir(parents=True, exist_ok=True)


def edge_id(head_id: str, relation: str, tail_id: str, source: str) -> str:
    """Stable 16-hex-char hash for edge identity."""
    h = hashlib.sha256(f"{head_id}|{relation}|{tail_id}|{source}".encode()).hexdigest()
    return f"primekg:{h[:16]}"


def build_edges() -> tuple[int, int]:
    """Build edges.parquet. Returns (n_rows, n_relations)."""
    print(f"Reading {RAW / 'kg.csv'} …", file=sys.stderr)
    df = pd.read_csv(RAW / "kg.csv")
    print(f"  loaded {len(df):,} rows", file=sys.stderr)

    # Stable string types for all id columns
    for col in ["x_id", "y_id", "x_type", "y_type", "x_source", "y_source", "relation"]:
        df[col] = df[col].astype(str)

    source_resource = df["x_source"] + "||" + df["y_source"]

    out = pd.DataFrame({
        "edge_id": [
            edge_id(h, r, t, s)
            for h, r, t, s in zip(df["x_id"], df["relation"], df["y_id"], source_resource)
        ],
        "dataset": "primekg",
        "head_id": df["x_id"],
        "head_type": df["x_type"],
        "relation": df["relation"],
        "tail_id": df["y_id"],
        "tail_type": df["y_type"],
        "source_resource": source_resource,
        "env_id": None,
        "payload": [
            {"display_relation": d, "x_name": xn, "y_name": yn}
            for d, xn, yn in zip(df["display_relation"], df["x_name"], df["y_name"])
        ],
    })

    n_rel = out["relation"].nunique()
    out_path = OUT / "edges.parquet"
    print(f"Writing {out_path} ({len(out):,} rows, {n_rel} relations)…", file=sys.stderr)
    pq.write_table(pa.Table.from_pandas(out, preserve_index=False), out_path, compression="snappy")
    return len(out), n_rel


def build_nodes() -> int:
    print(f"Reading {RAW / 'nodes.tab'} …", file=sys.stderr)
    df = pd.read_csv(RAW / "nodes.tab", sep="\t")
    for col in df.columns:
        df[col] = df[col].astype(str).str.strip('"')

    out = pd.DataFrame({
        "node_id": df["node_id"],
        "node_type": df["node_type"],
        "node_name": df["node_name"],
        "source_resource": df["node_source"],
        "dataset": "primekg",
        "node_index": df["node_index"].astype(int),
    })
    out_path = OUT / "nodes.parquet"
    print(f"Writing {out_path} ({len(out):,} rows)…", file=sys.stderr)
    pq.write_table(pa.Table.from_pandas(out, preserve_index=False), out_path, compression="snappy")
    return len(out)


if __name__ == "__main__":
    n_edges, n_rel = build_edges()
    n_nodes = build_nodes()
    print(f"\nDONE. edges={n_edges:,}  relations={n_rel}  nodes={n_nodes:,}")
