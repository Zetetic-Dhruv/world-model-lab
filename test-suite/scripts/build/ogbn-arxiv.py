"""Build ogbn-arxiv edges.parquet + nodes.parquet from OGB-downloaded raw files.

OGB stores edges in COO format (edge.csv.gz: source\ttarget) and node features
separately. Year-based split is in split/time/{train,valid,test}.csv.gz.
"""
from __future__ import annotations
import sys, gzip
from pathlib import Path
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "raw" / "ogbn-arxiv" / "ogbn_arxiv"
OUT = ROOT / "derived" / "ogbn-arxiv"
OUT.mkdir(parents=True, exist_ok=True)


def main():
    # Load edges (COO: source, target per row)
    print(f"Reading {RAW / 'raw' / 'edge.csv.gz'} …", file=sys.stderr)
    edges = pd.read_csv(RAW / "raw" / "edge.csv.gz", header=None, names=["src", "dst"])
    print(f"  {len(edges):,} edges", file=sys.stderr)

    # Load node years
    years = pd.read_csv(RAW / "raw" / "node_year.csv.gz", header=None, names=["year"])
    print(f"  {len(years):,} nodes, years range {years['year'].min()}-{years['year'].max()}", file=sys.stderr)

    # Year of each edge = year of source paper (the citing paper)
    edge_years = years["year"].values[edges["src"].values]

    # Year-based env: ≤2017 / 2018 / ≥2019
    env = np.where(edge_years <= 2017, "≤2017",
          np.where(edge_years == 2018, "2018", "≥2019"))

    edge_ids = "ogbn-arxiv:" + edges["src"].astype(str) + ":cites:" + edges["dst"].astype(str)
    out_e = pd.DataFrame({
        "edge_id": edge_ids,
        "dataset": "ogbn-arxiv",
        "head_id": edges["src"].astype(str),
        "head_type": "paper",
        "relation": "cites",
        "tail_id": edges["dst"].astype(str),
        "tail_type": "paper",
        "source_resource": "microsoft-academic-graph",
        "env_id": env,
        "payload": [{"src_year": int(y)} for y in edge_years],
    })
    out_path_e = OUT / "edges.parquet"
    print(f"Writing {out_path_e} …", file=sys.stderr)
    pq.write_table(pa.Table.from_pandas(out_e, preserve_index=False), out_path_e, compression="snappy")

    # Nodes: paper_id + year + label (subject area)
    labels = pd.read_csv(RAW / "raw" / "node-label.csv.gz", header=None, names=["label"])
    out_n = pd.DataFrame({
        "node_id": [str(i) for i in range(len(years))],
        "node_type": "paper",
        "source_resource": "microsoft-academic-graph",
        "dataset": "ogbn-arxiv",
        "year": years["year"].astype(int),
        "subject_label": labels["label"].astype(int),
    })
    out_path_n = OUT / "nodes.parquet"
    print(f"Writing {out_path_n} …", file=sys.stderr)
    pq.write_table(pa.Table.from_pandas(out_n, preserve_index=False), out_path_n, compression="snappy")

    print(f"\nDONE. edges={len(out_e):,} nodes={len(out_n):,}")


if __name__ == "__main__":
    main()
