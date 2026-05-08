"""Information-geometric diagnostics for the resolution × env × MI sweep.

Three statistically-grounded primitives:

  1. effective_rank_pr(Z): participation ratio of singular values
     pr = (Σσᵢ)² / Σσᵢ²     — bounded in [1, D]; collapsed → 1, isotropic → D
     References: Roy & Vetterli (2007), Recanatesi et al. (2019)

  2. ksg_mi(X, Y, k=3): Kraskov-Stögbauer-Grassberger MI estimator
     I(X,Y) ≈ ψ(k) + ψ(N) - <ψ(nx+1) + ψ(ny+1)>
     where ε = k-th NN distance in joint (Chebyshev), nx/ny count points within
     ε in marginal spaces. Algorithm 1 from Kraskov, Stögbauer & Grassberger
     (2004), Phys. Rev. E 69, 066138.

  3. twonn_intrinsic_dim(X, fraction=0.9): Facco et al. (2017) estimator
     For each point, µ = r₂/r₁ (ratio of distances to 1st and 2nd neighbors).
     Under locally-uniform density, F(µ) ≈ 1 - µ^(-d). Linear regression of
     -log(1-F) on log(µ) through origin → d.
     Reference: Facco, d'Errico, Rodriguez & Laio, Sci. Rep. 7, 12140 (2017).

LeWM Limitation #3 connection:
  The paper states that "matching the isotropic Gaussian prior in a high-
  dimensional latent space becomes challenging" in low-intrinsic-dim envs.
  This module operationalizes that claim:
    - effective_rank_pr quantifies whether SIGReg achieves the isotropic prior
    - twonn_intrinsic_dim quantifies the low-intrinsic-dim claim numerically
    - ksg_mi quantifies "limited data diversity" as I(z, env_state)

Caveats (flagged in functions):
  * KSG MI variance grows with dimensionality — always report alongside k sweep
  * Euclidean assumption distorts estimates on angular state spaces
  * TwoNN on D >> N is unreliable due to distance concentration
  * Trajectories cluster — use fraction<1 to discard top-µ outliers
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# 1. Effective rank — participation ratio of singular values
# ---------------------------------------------------------------------------

def effective_rank_pr(Z: np.ndarray) -> float:
    """Participation ratio of singular values: pr = (Σσᵢ)² / Σσᵢ².

    Bounded in [1, min(N, D)]. Behaves as the "effective number of dimensions
    used". Collapsed Z (all rows ≈ same point) gives pr ≈ 1; isotropic Gaussian
    Z gives pr ≈ min(N, D); rank-r structure gives pr between 1 and r.

    Args:
        Z: array of shape (N, D); NaN/Inf are filtered.

    Returns:
        Float in [1, min(N, D)]. Returns 0.0 if Z is empty.
    """
    Z = np.asarray(Z, dtype=np.float64)
    if Z.size == 0:
        return 0.0
    if Z.ndim != 2:
        raise ValueError(f"Z must be (N, D); got shape {Z.shape}")
    Z = Z[np.isfinite(Z).all(axis=1)]
    if len(Z) < 2:
        return 1.0
    Z_centered = Z - Z.mean(axis=0, keepdims=True)
    s = np.linalg.svd(Z_centered, compute_uv=False)
    s = s[s > 1e-12]  # drop numerical zeros
    if len(s) == 0:
        return 1.0
    pr = float(s.sum() ** 2 / (s ** 2).sum())
    return pr


# ---------------------------------------------------------------------------
# 2. KSG MI estimator (Kraskov-Stögbauer-Grassberger Algorithm 1)
# ---------------------------------------------------------------------------

def ksg_mi(X: np.ndarray, Y: np.ndarray, k: int = 3) -> float:
    """KSG Algorithm 1 MI estimator using Chebyshev distance.

    I(X;Y) ≈ ψ(k) + ψ(N) - <ψ(nx+1) + ψ(ny+1)>

    Args:
        X: shape (N, dx) or (N,)
        Y: shape (N, dy) or (N,)
        k: kNN parameter; higher k → lower variance, higher bias

    Returns:
        Estimated I(X;Y) in nats. May be slightly negative due to finite-N bias.

    Notes:
        Cost is O(N² log N) for kNN. For N>5000 consider subsampling.
        High-D inputs (D > ~30) suffer from distance concentration; bias grows
        even at large N. For (latent, low-D ground truth) pairs in our use case,
        this is fine; for (latent, latent), prefer PCA-reduced inputs.
    """
    from sklearn.neighbors import NearestNeighbors
    from scipy.special import digamma

    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    if X.ndim == 1:
        X = X[:, None]
    if Y.ndim == 1:
        Y = Y[:, None]
    if X.shape[0] != Y.shape[0]:
        raise ValueError(f"X and Y must have same N; got {X.shape[0]} vs {Y.shape[0]}")

    # Filter rows with NaN/Inf
    mask = np.isfinite(X).all(axis=1) & np.isfinite(Y).all(axis=1)
    X, Y = X[mask], Y[mask]
    N = X.shape[0]
    if N < k + 2:
        return float('nan')

    # k-th NN distance in joint space (Chebyshev = max-norm)
    XY = np.concatenate([X, Y], axis=1)
    nn_xy = NearestNeighbors(n_neighbors=k + 1, metric='chebyshev').fit(XY)
    eps = nn_xy.kneighbors(XY)[0][:, k]  # k-th distance per point (inclusive of self at idx 0)

    # Count neighbors strictly within eps in each marginal space.
    # sklearn's radius_neighbors takes a scalar radius; for per-query radii
    # we use KDTree.query_radius (supports array). Chebyshev distance is
    # supported by KDTree.
    from sklearn.neighbors import KDTree
    tree_x = KDTree(X, metric='chebyshev')
    tree_y = KDTree(Y, metric='chebyshev')
    eps_query = np.maximum(eps - 1e-12, 1e-12)
    nx_counts = tree_x.query_radius(X, r=eps_query, count_only=True)
    ny_counts = tree_y.query_radius(Y, r=eps_query, count_only=True)
    # subtract self-count
    nx = np.maximum(np.asarray(nx_counts) - 1, 0)
    ny = np.maximum(np.asarray(ny_counts) - 1, 0)
    # KSG can produce nx/ny = 0 when eps is tiny — guard against digamma(1) = -γ
    nx = np.maximum(nx, 0)
    ny = np.maximum(ny, 0)

    mi_nats = float(digamma(k) + digamma(N) -
                    (digamma(nx + 1) + digamma(ny + 1)).mean())
    return mi_nats


# ---------------------------------------------------------------------------
# 3. TwoNN intrinsic dimension (Facco et al. 2017)
# ---------------------------------------------------------------------------

def twonn_intrinsic_dim(X: np.ndarray, fraction: float = 0.9) -> float:
    """TwoNN intrinsic-dimension estimator (Facco et al. 2017).

    For each point, compute distances r₁ < r₂ to its 1st and 2nd nearest
    neighbors. Define µᵢ = r₂ᵢ/r₁ᵢ. Under locally-uniform density on a d-dim
    manifold, F(µ) ≈ 1 - µ^(-d). Linear-regress -log(1 - F) on log(µ) through
    the origin → d.

    Args:
        X: shape (N, D). For best results use N >> D.
        fraction: keep the bottom `fraction` of µ values (drops outliers from
                  density inhomogeneity). Facco et al. recommend 0.9.

    Returns:
        Estimated intrinsic dimension d (typically 0 < d ≤ D).

    Notes:
        Unreliable when D >> N (distance concentration in high-D). For latent
        spaces (D=192), apply TwoNN only after PCA-reducing to top-16 dims.
        Robust on Euclidean state spaces; biased on cyclic/angular spaces
        (Reacher qpos near ±π — but typical trajectories don't span the wrap).
    """
    from sklearn.neighbors import NearestNeighbors

    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2:
        raise ValueError(f"X must be (N, D); got shape {X.shape}")
    X = X[np.isfinite(X).all(axis=1)]
    N = len(X)
    if N < 4:
        return float('nan')

    # Distances to 1st and 2nd nearest neighbors (excluding self)
    nn = NearestNeighbors(n_neighbors=3).fit(X)  # idx 0 = self
    d, _ = nn.kneighbors(X)
    r1, r2 = d[:, 1], d[:, 2]
    # Filter degenerate (r1 == 0): coincident points; rare but possible
    nonzero = r1 > 1e-12
    r1, r2 = r1[nonzero], r2[nonzero]
    if len(r1) < 4:
        return float('nan')

    mu = r2 / r1
    mu_sorted = np.sort(mu)
    n_keep = max(2, int(len(mu_sorted) * fraction))
    mu_use = mu_sorted[:n_keep]

    # Empirical CDF F(µ) = i/N at the i-th order statistic; use only kept points
    F = np.arange(1, n_keep + 1) / (len(mu_sorted) + 1)  # +1 to avoid F=1 at end
    log_mu = np.log(mu_use)
    log_one_minus_F = -np.log(1 - F)

    # Linear fit through origin: d = (log_µ · -log(1-F)) / (log_µ · log_µ)
    num = float((log_mu * log_one_minus_F).sum())
    den = float((log_mu * log_mu).sum())
    if den < 1e-12:
        return float('nan')
    return num / den


# ---------------------------------------------------------------------------
# Bundled diagnostic suite (JSON-friendly outputs)
# ---------------------------------------------------------------------------

def diagnostic_suite(
    Z: np.ndarray,
    env_state: np.ndarray | None = None,
    Z_next: np.ndarray | None = None,
    pca_dim_for_twonn: int = 16,
    ksg_k: int = 3,
) -> dict:
    """Run effective_rank, intrinsic-dim, and MI on a (Z, env_state) pair.

    Args:
        Z:           (N, D) latent embeddings sampled from heldout val set
        env_state:   (N, S) corresponding ground-truth env states (optional)
        Z_next:      (N, D) latents at the NEXT stride-step (optional, for I(z_t, z_t+1))
        pca_dim_for_twonn: dim cap for twonn_intrinsic_dim(Z) (avoids high-D
                           distance concentration). 16 is the LeWM-style 16×16
                           grid analog for latent dims.
        ksg_k:       neighborhood size for KSG MI

    Returns:
        Dict of float scalars (JSON-serializable). Missing inputs yield NaN.
    """
    out: dict = {}
    out["latent_effective_rank_pr"] = effective_rank_pr(Z)

    # TwoNN on latents — apply via PCA-reduction to avoid high-D bias
    Z_centered = Z - Z.mean(axis=0, keepdims=True)
    if Z.shape[1] > pca_dim_for_twonn:
        u, s, vt = np.linalg.svd(Z_centered, full_matrices=False)
        Z_reduced = (Z_centered @ vt.T[:, :pca_dim_for_twonn])
    else:
        Z_reduced = Z_centered
    out["latent_twonn_intrinsic_dim"] = twonn_intrinsic_dim(Z_reduced)

    if env_state is not None:
        out["env_state_twonn_intrinsic_dim"] = twonn_intrinsic_dim(env_state)
        out["env_state_effective_rank_pr"] = effective_rank_pr(env_state)
        out["mi_z_envstate_nats"] = ksg_mi(Z_reduced, env_state, k=ksg_k)

    if Z_next is not None:
        Zn_centered = Z_next - Z_next.mean(axis=0, keepdims=True)
        if Z_next.shape[1] > pca_dim_for_twonn:
            _, _, vt2 = np.linalg.svd(Zn_centered, full_matrices=False)
            Zn_reduced = Zn_centered @ vt2.T[:, :pca_dim_for_twonn]
        else:
            Zn_reduced = Zn_centered
        out["mi_z_znext_nats"] = ksg_mi(Z_reduced, Zn_reduced, k=ksg_k)

    return out


# ---------------------------------------------------------------------------
# Sanity tests — run via `python -m src.diagnostics`
# ---------------------------------------------------------------------------

def _sanity_check():
    """Verify each estimator on synthetic baselines with known answers."""
    rng = np.random.default_rng(0)

    print("=== effective_rank_pr ===")
    Z_iso = rng.standard_normal((1000, 16))
    # genuinely collapsed: all rows are scalar multiples of one direction → rank 1
    Z_collapsed = np.outer(rng.standard_normal(1000), rng.standard_normal(16))
    Z_rank2 = rng.standard_normal((1000, 2)) @ rng.standard_normal((2, 16))
    print(f"  isotropic 16-D:        pr={effective_rank_pr(Z_iso):.2f}  (expect ~16)")
    print(f"  rank-1 collapsed:      pr={effective_rank_pr(Z_collapsed):.2f}  (expect ~1)")
    print(f"  rank-2 in 16-D:        pr={effective_rank_pr(Z_rank2):.2f}  (expect ~2)")

    print("\n=== ksg_mi ===")
    N = 2000
    X = rng.standard_normal((N, 4))
    Y_indep = rng.standard_normal((N, 4))
    Y_copy = X.copy() + rng.standard_normal((N, 4)) * 0.01
    Y_linear = X @ rng.standard_normal((4, 4))  # invertible map ⇒ I = h(X)
    print(f"  X, Y independent (4D): I={ksg_mi(X, Y_indep):.3f} nats  (expect ≈ 0)")
    print(f"  X, Y near-identical:   I={ksg_mi(X, Y_copy):.3f} nats  (expect large positive)")
    print(f"  X, Y linear bijection: I={ksg_mi(X, Y_linear):.3f} nats  (expect ~h(X) ≈ 5-6 nats for unit-Gaussian 4D)")

    print("\n=== twonn_intrinsic_dim ===")
    X_3d = rng.standard_normal((2000, 3))
    X_3d_in_10d = np.concatenate([X_3d, np.zeros((2000, 7))], axis=1)
    X_8d = rng.standard_normal((2000, 8))
    print(f"  3-D Gaussian:           d={twonn_intrinsic_dim(X_3d):.2f}  (expect ~3)")
    print(f"  3-D embedded in 10-D:   d={twonn_intrinsic_dim(X_3d_in_10d):.2f}  (expect ~3)")
    print(f"  8-D Gaussian:           d={twonn_intrinsic_dim(X_8d):.2f}  (expect ~8)")

    print("\n=== diagnostic_suite (smoke) ===")
    Z = rng.standard_normal((500, 192))  # latent-shaped
    state = rng.standard_normal((500, 6))  # state-shaped
    Z_next = Z + rng.standard_normal((500, 192)) * 0.1  # near-identity dynamics
    out = diagnostic_suite(Z, env_state=state, Z_next=Z_next)
    for k, v in out.items():
        print(f"  {k}: {v:.4f}")


if __name__ == "__main__":
    _sanity_check()
