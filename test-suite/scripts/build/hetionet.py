"""Build Hetionet edges.parquet + nodes.parquet from SIF/TSV distribution.

NOTE on representational commitment: The Hetionet SIF/TSV distribution does NOT
preserve per-edge source_resource. The JSON distribution may; if needed, parse
hetionet-v1.0.json.bz2 to extract per-edge provenance. For now we use
'hetionet-integrated' as a placeholder — manifest flags this as a gap.

Per-edge source provenance is needed to predict the cross-source generalization
gap from structure (see the hetionet hypotheses). Resolving this is a follow-up.
"""
from __future__ import annotations
import hashlib
import gzip
import sys
from pathlib import Path
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "raw" / "hetionet"
OUT = ROOT / "derived" / "hetionet"
OUT.mkdir(parents=True, exist_ok=True)


def edge_id(head_id: str, relation: str, tail_id: str) -> str:
    h = hashlib.sha256(f"{head_id}|{relation}|{tail_id}".encode()).hexdigest()
    return f"hetionet:{h[:16]}"


def parse_typed_id(s: str) -> tuple[str, str]:
    """Hetionet uses 'Type::Identifier' format. Returns (type, identifier)."""
    if "::" in s:
        t, i = s.split("::", 1)
        return t, i
    return "Unknown", s


def build_edges() -> int:
    print(f"Reading {RAW / 'edges.sif.gz'} …", file=sys.stderr)
    with gzip.open(RAW / "edges.sif.gz", "rt") as f:
        df = pd.read_csv(f, sep="\t")
    print(f"  loaded {len(df):,} rows", file=sys.stderr)

    head_types, head_ids = zip(*[parse_typed_id(s) for s in df["source"]])
    tail_types, tail_ids = zip(*[parse_typed_id(s) for s in df["target"]])

    out = pd.DataFrame({
        "edge_id": [edge_id(h, r, t) for h, r, t in zip(df["source"], df["metaedge"], df["target"])],
        "dataset": "hetionet",
        "head_id": list(head_ids),
        "head_type": list(head_types),
        "relation": df["metaedge"],
        "tail_id": list(tail_ids),
        "tail_type": list(tail_types),
        "source_resource": "hetionet-integrated",
        "env_id": None,
        "payload": [{"raw_source": s, "raw_target": t} for s, t in zip(df["source"], df["target"])],
    })

    out_path = OUT / "edges.parquet"
    print(f"Writing {out_path} ({len(out):,} rows)…", file=sys.stderr)
    pq.write_table(pa.Table.from_pandas(out, preserve_index=False), out_path, compression="snappy")
    return len(out)


def build_nodes() -> int:
    print(f"Reading {RAW / 'nodes.tsv'} …", file=sys.stderr)
    df = pd.read_csv(RAW / "nodes.tsv", sep="\t")
    node_types, node_ids = zip(*[parse_typed_id(s) for s in df["id"]])

    out = pd.DataFrame({
        "node_id": list(node_ids),
        "node_type": list(node_types),
        "node_name": df["name"].astype(str),
        "source_resource": "hetionet-integrated",
        "dataset": "hetionet",
        "raw_id": df["id"],
    })
    out_path = OUT / "nodes.parquet"
    print(f"Writing {out_path} ({len(out):,} rows)…", file=sys.stderr)
    pq.write_table(pa.Table.from_pandas(out, preserve_index=False), out_path, compression="snappy")
    return len(out)


if __name__ == "__main__":
    n_e = build_edges()
    n_n = build_nodes()
    print(f"\nDONE. edges={n_e:,} nodes={n_n:,}")
