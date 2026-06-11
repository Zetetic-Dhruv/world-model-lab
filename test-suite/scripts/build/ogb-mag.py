"""Build ogb-mag edges.parquet + nodes.parquet — heterogeneous citation KG.

4 node types (paper, author, institution, field_of_study), 4 relation types.
Year information is on paper nodes only; edge env_id uses src paper's year
when src is a paper, else null.
"""
from __future__ import annotations
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "raw" / "ogb-mag" / "ogbn_mag" / "raw"
OUT = ROOT / "derived" / "ogb-mag"
OUT.mkdir(parents=True, exist_ok=True)

NODE_COUNTS = {"paper": 736389, "author": 1134649, "institution": 8740, "field_of_study": 59965}

RELATIONS = [
    ("author___affiliated_with___institution", "author", "affiliated_with", "institution"),
    ("author___writes___paper", "author", "writes", "paper"),
    ("paper___cites___paper", "paper", "cites", "paper"),
    ("paper___has_topic___field_of_study", "paper", "has_topic", "field_of_study"),
]


def main():
    paper_year = pd.read_csv(RAW / "node-feat" / "paper" / "node_year.csv.gz", header=None, names=["year"])["year"].values
    print(f"paper years: {paper_year.min()}-{paper_year.max()}", file=sys.stderr)

    all_edges = []
    for dirname, h_type, rel, t_type in RELATIONS:
        path = RAW / "relations" / dirname / "edge.csv.gz"
        df = pd.read_csv(path, header=None, names=["src", "dst"])
        print(f"{dirname}: {len(df):,} edges", file=sys.stderr)

        if h_type == "paper":
            env = np.where(paper_year[df["src"].values] <= 2017, "≤2017",
                  np.where(paper_year[df["src"].values] == 2018, "2018", "≥2019"))
        else:
            env = np.array([None] * len(df), dtype=object)

        all_edges.append(pd.DataFrame({
            "edge_id": f"ogb-mag:{rel}:" + df["src"].astype(str) + ":" + df["dst"].astype(str),
            "dataset": "ogb-mag",
            "head_id": h_type + ":" + df["src"].astype(str),
            "head_type": h_type,
            "relation": rel,
            "tail_id": t_type + ":" + df["dst"].astype(str),
            "tail_type": t_type,
            "source_resource": "microsoft-academic-graph",
            "env_id": env,
            "payload": [{"src_idx": int(s), "dst_idx": int(d)} for s, d in zip(df["src"], df["dst"])],
        }))

    edges = pd.concat(all_edges, ignore_index=True)
    print(f"Total edges: {len(edges):,}", file=sys.stderr)
    pq.write_table(pa.Table.from_pandas(edges, preserve_index=False), OUT / "edges.parquet", compression="snappy")

    # Nodes table
    node_rows = []
    for t, n in NODE_COUNTS.items():
        for i in range(n):
            node_rows.append({"node_id": f"{t}:{i}", "node_type": t, "source_resource": "microsoft-academic-graph", "dataset": "ogb-mag"})
    # Use vectorized construction instead
    node_dfs = []
    for t, n in NODE_COUNTS.items():
        df = pd.DataFrame({
            "node_id": [f"{t}:{i}" for i in range(n)],
            "node_type": t,
            "source_resource": "microsoft-academic-graph",
            "dataset": "ogb-mag",
        })
        if t == "paper":
            df["year"] = paper_year
        node_dfs.append(df)
    nodes = pd.concat(node_dfs, ignore_index=True)
    pq.write_table(pa.Table.from_pandas(nodes, preserve_index=False), OUT / "nodes.parquet", compression="snappy")

    print(f"\nDONE. edges={len(edges):,}  nodes={len(nodes):,}  by-type: {NODE_COUNTS}")


if __name__ == "__main__":
    main()
