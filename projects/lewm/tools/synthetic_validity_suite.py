#!/usr/bin/env python3
"""
Synthetic validity suite — minimal, ground-truth constructions that each
reproduce one measurement artifact from the four-check protocol, run over many
seeds so each artifact is reported as mean [2.5%, 97.5%] rather than a single
number. No training, no checkpoints, no GPU: fully reproducible from this file.

Each construction has a KNOWN truth, so the artifact is demonstrable, not merely
observed:
  - Estimator geometry (Check 2): an invertible diagonal rescaling provably
    preserves I(z;s); any KSG movement is therefore pure estimator artifact,
    while a standardized decoding probe stays flat.
  - Episode/group leakage (Check 3): state is buried under a per-group offset;
    a frame split can read the offset, a group split cannot.
  - Power / group baseline (Check 4): an input that genuinely carries the state
    is undecodable cross-group with few held-out groups and decodable with many.

Usage: python3 tools/synthetic_validity_suite.py [--seeds N] [--out CSV]
"""
from __future__ import annotations
import argparse, csv, sys
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import ShuffleSplit, GroupShuffleSplit
from sklearn.neighbors import NearestNeighbors
from scipy.special import digamma
import warnings
warnings.filterwarnings("ignore")            # degenerate group-splits are expected (the point)
np.seterr(all="ignore")


def ksg_mi(X, Y, k=5):
    """Joint Kraskov-Stoegbauer-Grassberger MI estimator (estimator 1), nats."""
    X = np.atleast_2d(X);  Y = np.atleast_2d(Y)
    if X.shape[0] == 1: X = X.T
    if Y.shape[0] == 1: Y = Y.T
    n = X.shape[0]
    XY = np.hstack([X, Y])
    nn = NearestNeighbors(metric="chebyshev").fit(XY)
    d, _ = nn.kneighbors(XY, n_neighbors=k + 1)
    eps = d[:, k]

    def count(Z):
        z = NearestNeighbors(metric="chebyshev").fit(Z)
        c = np.empty(n)
        for i in range(n):
            idx = z.radius_neighbors(Z[i:i + 1], radius=max(eps[i] - 1e-12, 0),
                                     return_distance=False)[0]
            c[i] = len(idx) - 1
        return c

    nx, ny = count(X), count(Y)
    mi = digamma(k) + digamma(n) - np.mean(digamma(nx + 1) + digamma(ny + 1))
    return float(max(mi, 0.0))


def probe_r2(Z, S, groups=None, seed=0, n_splits=5, test_frac=0.3):
    """Mean held-out R^2 of a standardized ridge probe; group- or frame-split."""
    Z = np.atleast_2d(Z); S = np.asarray(S, float)
    if Z.shape[0] == 1: Z = Z.T
    if groups is None:
        sp = ShuffleSplit(n_splits=n_splits, test_size=test_frac, random_state=seed)
        it = sp.split(Z)
    else:
        sp = GroupShuffleSplit(n_splits=n_splits, test_size=test_frac, random_state=seed)
        it = sp.split(Z, S, groups)
    out = []
    for tr, te in it:
        sc = StandardScaler().fit(Z[tr])
        m = Ridge(alpha=1.0).fit(sc.transform(Z[tr]), S[tr])
        out.append(m.score(sc.transform(Z[te]), S[te]))
    return float(np.mean(out))


def ci(vals):
    a = np.asarray(vals, float)
    return float(a.mean()), float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))


# ---- constructions -------------------------------------------------------

def geometry(rng, d_noise=7, n=600, sigma=0.1, stretch=8.0):
    s = rng.normal(size=n)
    z = np.column_stack([s + rng.normal(scale=sigma, size=n), rng.normal(size=(n, d_noise))])
    scale = np.r_[1.0, np.full(d_noise, stretch)]          # invertible diagonal map
    z_iso, z_aniso = z, z * scale
    return (ksg_mi(z_iso, s), ksg_mi(z_aniso, s),          # true MI identical; KSG should move
            probe_r2(z_iso, s, seed=int(rng.integers(1 << 30))),
            probe_r2(z_aniso, s, seed=int(rng.integers(1 << 30))))


def geometry_at(rng, stretch, d_noise=7, n=600, sigma=0.1):
    # KSG MI and probe R^2 at one anisotropy level; stretch=1 is the isometric base.
    s = rng.normal(size=n)
    z = np.column_stack([s + rng.normal(scale=sigma, size=n), rng.normal(size=(n, d_noise))])
    zt = z * np.r_[1.0, np.full(d_noise, float(stretch))]   # invertible diagonal map
    return ksg_mi(zt, s), probe_r2(zt, s, seed=int(rng.integers(1 << 30)))


def leakage(rng, n_groups=40, per=20, d_z=8, sigma_g=1.0, sigma_w=0.3, sig=0.15, sigma_n=0.15):
    # State is dominated by a between-group component the latent encodes only as
    # group identity (B[g]); a frame split learns B[g]->s, a group split cannot.
    G, n = n_groups, n_groups * per
    groups = np.repeat(np.arange(G), per)
    group_effect = rng.normal(scale=sigma_g, size=G)       # dominant between-group state
    within = rng.normal(scale=sigma_w, size=n)             # small within-group state
    s = group_effect[groups] + within
    B = rng.normal(size=(G, d_z))                          # per-group latent identity/offset
    wdir = rng.normal(size=(1, d_z))
    z = B[groups] + sig * within[:, None] * wdir + rng.normal(scale=sigma_n, size=(n, d_z))
    sd = int(rng.integers(1 << 30))
    return (probe_r2(z, s, seed=sd),                       # frame split: learns B[g]->s
            probe_r2(z, s, groups=groups, seed=sd))        # group split: unseen B[g]


def power(rng, G, per=15, d_x=6, sigma_b=1.5, sigma_n=0.4):
    n = G * per
    s = rng.normal(size=(n, 1))
    groups = np.repeat(np.arange(G), per)
    b = rng.normal(scale=sigma_b, size=(G, d_x))[groups]
    x = s @ rng.normal(size=(1, d_x)) + b + rng.normal(scale=sigma_n, size=(n, d_x))  # x carries s
    return probe_r2(x, s.ravel(), groups=groups, seed=int(rng.integers(1 << 30)))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=30)
    ap.add_argument("--out", default=None)
    ap.add_argument("--geom-sweep-out", default=None)
    a = ap.parse_args()
    S = a.seeds

    if a.geom_sweep_out:
        with open(a.geom_sweep_out, "w", newline="") as fh:
            cw = csv.writer(fh)
            cw.writerow(["stretch", "ksg_mean", "ksg_ep", "ksg_em",
                         "probe_mean", "probe_ep", "probe_em"])
            for st in (1, 2, 4, 8, 16, 32):
                ks, pr = [], []
                for i in range(S):
                    k, p = geometry_at(np.random.default_rng(4000 + st * 100 + i), st)
                    ks.append(k); pr.append(p)
                km, klo, khi = ci(ks); pm, plo, phi = ci(pr)
                cw.writerow([st, round(km, 4), round(khi - km, 4), round(km - klo, 4),
                             round(pm, 4), round(phi - pm, 4), round(pm - plo, 4)])
        print(f"wrote {a.geom_sweep_out}")
        return 0

    rows = []

    geo = [geometry(np.random.default_rng(1000 + i)) for i in range(S)]
    ksg_iso, ksg_an, pr_iso, pr_an = map(list, zip(*geo))
    rows += [("geometry", "KSG MI iso (nats)", *ci(ksg_iso)),
             ("geometry", "KSG MI anisotropic (nats)", *ci(ksg_an)),
             ("geometry", "KSG drift (aniso-iso)", *ci(np.array(ksg_an) - np.array(ksg_iso))),
             ("geometry", "probe R2 iso", *ci(pr_iso)),
             ("geometry", "probe R2 anisotropic", *ci(pr_an))]

    lk = [leakage(np.random.default_rng(2000 + i)) for i in range(S)]
    fr, gp = map(list, zip(*lk))
    rows += [("leakage", "frame-split R2", *ci(fr)),
             ("leakage", "group-split R2", *ci(gp)),
             ("leakage", "gap (frame-group)", *ci(np.array(fr) - np.array(gp)))]

    for G in (6, 80):
        pw = [power(np.random.default_rng(3000 + G * 100 + i), G) for i in range(S)]
        rows.append(("power", f"raw-input R2, {G} groups", *ci(pw)))

    w = max(len(r[1]) for r in rows)
    print(f"\n{'construction':12s} {'quantity':{w}s} {'mean':>8s}  [ 2.5%,   97.5%]")
    print("-" * (12 + w + 28))
    for name, q, m, lo, hi in rows:
        print(f"{name:12s} {q:{w}s} {m:8.3f}  [{lo:7.3f}, {hi:7.3f}]")

    if a.out:
        with open(a.out, "w", newline="") as fh:
            cw = csv.writer(fh)
            cw.writerow(["construction", "quantity", "mean", "ci_lo", "ci_hi"])
            cw.writerows(rows)
        print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
