"""Build movielens traces.parquet from ML-32M + ML-1M snapshots.

Two snapshots materialized separately for the cross-snapshot pair. Vectorized
build (no iterrows) — ML-32M is 32M rows, must be fast. Movie metadata kept in
a separate movies.parquet so the main traces table stays narrow.
"""
from __future__ import annotations
import sys
from pathlib import Path
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "raw" / "movielens"
OUT = ROOT / "derived" / "movielens"
OUT.mkdir(parents=True, exist_ok=True)


def build_snapshot(snap: str, ratings_df: pd.DataFrame) -> int:
    ratings_df["userId"] = ratings_df["userId"].astype(str)
    ratings_df["movieId"] = ratings_df["movieId"].astype(str)

    out = pd.DataFrame({
        "trace_id": "movielens:" + snap + ":" + ratings_df["userId"] + ":" + ratings_df["movieId"] + ":" + ratings_df["timestamp"].astype(str),
        "dataset": "movielens",
        "entity_id": ratings_df["userId"],
        "entity_type": "user",
        "parent_id": ratings_df["movieId"],
        "parent_type": "movie",
        "event_type": "rating",
        "t": pd.to_datetime(ratings_df["timestamp"], unit="s"),
        "t_order": ratings_df["timestamp"].astype(int),
        "env_id": snap,
        "env_axis": "snapshot",
        "split": "train",
        "payload": [{"rating": float(r)} for r in ratings_df["rating"]],
    })
    out_path = OUT / f"traces_{snap}.parquet"
    print(f"Writing {out_path} ({len(out):,} rows)…", file=sys.stderr)
    pq.write_table(pa.Table.from_pandas(out, preserve_index=False), out_path, compression="snappy")
    return len(out)


def build_movies(snap: str, movies_df: pd.DataFrame) -> int:
    out = pd.DataFrame({
        "movie_id": movies_df["movieId"].astype(str),
        "title": movies_df["title"].astype(str),
        "genres": movies_df["genres"].astype(str),
        "snapshot": snap,
    })
    out_path = OUT / f"movies_{snap}.parquet"
    pq.write_table(pa.Table.from_pandas(out, preserve_index=False), out_path, compression="snappy")
    return len(out)


def main():
    # ML-32M
    p = RAW / "ml-32m"
    print(f"Reading {p / 'ratings.csv'} …", file=sys.stderr)
    ratings_32m = pd.read_csv(p / "ratings.csv")
    print(f"  loaded {len(ratings_32m):,} ratings", file=sys.stderr)
    movies_32m = pd.read_csv(p / "movies.csv")
    n_32m = build_snapshot("ml-32m", ratings_32m)
    n_m32 = build_movies("ml-32m", movies_32m)
    del ratings_32m

    # ML-1M (different .dat format)
    p1 = RAW / "ml-1m"
    if p1.exists():
        ratings_1m = pd.read_csv(p1 / "ratings.dat", sep="::", engine="python",
                                 names=["userId", "movieId", "rating", "timestamp"])
        movies_1m = pd.read_csv(p1 / "movies.dat", sep="::", engine="python",
                                names=["movieId", "title", "genres"], encoding="latin-1")
        n_1m = build_snapshot("ml-1m", ratings_1m)
        n_m1 = build_movies("ml-1m", movies_1m)
    else:
        n_1m = n_m1 = 0

    print(f"\nDONE. ml-32m ratings={n_32m:,} movies={n_m32:,}   ml-1m ratings={n_1m:,} movies={n_m1:,}")


if __name__ == "__main__":
    main()
