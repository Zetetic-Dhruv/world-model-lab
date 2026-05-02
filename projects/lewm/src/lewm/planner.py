"""Path C canonical CEM planner + MPC runner.

Canonical mechanism (matches stable-worldmodel/solver/cem.py + le-wm/jepa.py):
  - Sample shape (N, H, action_block * action_dim)
  - Each "horizon step" emits action_block raw env actions via unroll
  - n_iters defaults to 30 (canonical universal)
  - First sample of each iter forced to current mean (monotonicity guard)
  - No internal action clipping (env clips at execution)
  - Per-solver torch.Generator for deterministic sampling
  - Real past actions filled into action history at plan onset
  - Warm-start: previous plan's tail re-used as next-plan init
  - Receding horizon decoupled from planning horizon
"""

from __future__ import annotations

import torch
import torch.nn as nn
import numpy as np

from .model import LeWM


class CEMPlanner:
    """Path C canonical CEM in latent space.

    Args:
      model: trained LeWM
      horizon: plan length in HORIZON-STEPS (each step = action_block env steps)
      action_block: env actions per horizon step (canonical PushT: 5)
      action_dim: raw env action dim (canonical PushT: 2)
      n_samples: candidate sequences per iter (canonical: 300)
      n_elites: top-K kept per iter (canonical: 30)
      n_iters: CEM iterations (canonical: 30 universal)
      sigma_init: initial proposal std (canonical: 1.0)
      history_size: predictor history-window size (canonical PushT: 3)
      seed: RNG seed for the per-solver Generator
    """

    def __init__(
        self,
        model: LeWM,
        horizon: int = 5,
        action_block: int = 5,
        action_dim: int = 2,
        n_samples: int = 300,
        n_elites: int = 30,
        n_iters: int = 30,
        sigma_init: float = 1.0,
        history_size: int = 3,
        seed: int = 1234,
    ):
        self.model = model
        self.horizon = horizon
        self.action_block = action_block
        self.action_dim = action_dim
        self.n_samples = n_samples
        self.n_elites = n_elites
        self.n_iters = n_iters
        self.sigma_init = sigma_init
        self.history_size = history_size
        self.token_dim = action_block * action_dim
        self.seed = seed
        self._gen: torch.Generator | None = None

    def _generator(self, device: torch.device) -> torch.Generator:
        if self._gen is None or self._gen.device != device:
            self._gen = torch.Generator(device=device)
            self._gen.manual_seed(self.seed)
        return self._gen

    @torch.no_grad()
    def _rollout_cost(
        self,
        z_history: torch.Tensor,
        a_history: torch.Tensor,
        candidates: torch.Tensor,
        z_goal: torch.Tensor,
    ) -> torch.Tensor:
        """Score N candidate plans.

        z_history:  (history_size, D)             — encoded past obs (stride=action_block)
        a_history:  (history_size, token_dim)     — real past action tokens
        candidates: (N, horizon, token_dim)
        z_goal:     (D,)
        Returns cost: (N,) — squared L2 of final predicted emb to z_goal.
        """
        N, H, _ = candidates.shape
        # Replicate (z_history, a_history) N times along batch
        ctx_z = z_history.unsqueeze(0).expand(N, -1, -1).contiguous()
        ctx_a = a_history.unsqueeze(0).expand(N, -1, -1).contiguous()

        for k in range(H):
            window_z = ctx_z[:, -self.history_size:, :]
            window_a = ctx_a[:, -self.history_size:, :]
            preds = self.model.predict(window_z, window_a)  # (N, history_size, D)
            next_z = preds[:, -1:, :]
            next_a = candidates[:, k:k + 1, :]
            ctx_z = torch.cat([ctx_z, next_z], dim=1)
            ctx_a = torch.cat([ctx_a, next_a], dim=1)

        z_final = ctx_z[:, -1, :]
        diff = z_final - z_goal.unsqueeze(0)
        return diff.pow(2).sum(dim=-1)

    @torch.no_grad()
    def plan(
        self,
        z_history: torch.Tensor,
        a_history: torch.Tensor,
        z_goal: torch.Tensor,
        init_action: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run CEM. Returns mu: (horizon, token_dim) best plan in token form."""
        device = next(self.model.parameters()).device
        z_history = z_history.to(device)
        a_history = a_history.to(device)
        z_goal = z_goal.to(device)
        gen = self._generator(device)

        H, T = self.horizon, self.token_dim
        if init_action is not None:
            mu = init_action.to(device, dtype=z_history.dtype)
        else:
            mu = torch.zeros(H, T, device=device, dtype=z_history.dtype)
        sigma = torch.full((H, T), self.sigma_init,
                           device=device, dtype=z_history.dtype)

        for it in range(self.n_iters):
            samples = mu.unsqueeze(0) + sigma.unsqueeze(0) * torch.randn(
                self.n_samples, H, T, device=device, dtype=z_history.dtype,
                generator=gen,
            )
            # Canonical trick: first sample is the current mean (monotonicity guard)
            samples[0] = mu
            # No internal clipping — env clips at execution

            costs = self._rollout_cost(z_history, a_history, samples, z_goal)
            elite_idx = torch.topk(costs, self.n_elites, largest=False).indices
            elites = samples[elite_idx]

            mu = elites.mean(dim=0)
            sigma = elites.std(dim=0)
            # No sigma_min floor (canonical)

        return mu  # (H, T)

    @staticmethod
    def unroll_plan(mu: torch.Tensor, action_block: int, action_dim: int) -> np.ndarray:
        """Expand plan tokens (H, action_block * action_dim) → env actions (H * action_block, action_dim)."""
        H, T = mu.shape
        if T != action_block * action_dim:
            raise ValueError(f"token_dim={T} != action_block * action_dim "
                             f"= {action_block * action_dim}")
        return mu.reshape(H, action_block, action_dim).reshape(-1, action_dim).cpu().numpy()


class MPCRunner:
    """Path C MPC: receding-horizon execution with warm-start + real action history.

    The runner expects the env to be already in the desired init state. It:
      1. Bootstraps `history_size * action_block` env steps of small random actions
         to fill the action-history buffer (predictor needs real action context).
      2. Plans via CEM, executes `receding_horizon * action_block` env steps,
         replans (with warm-start from prior plan tail), repeats until budget
         or `success_fn` triggers.
    """

    def __init__(
        self,
        planner: CEMPlanner,
        history_size: int = 3,
        action_block: int = 5,
        action_dim: int = 2,
        receding_horizon: int | None = None,
        budget_env_steps: int = 50,
        success_fn=None,
    ):
        self.planner = planner
        self.history_size = history_size
        self.action_block = action_block
        self.action_dim = action_dim
        self.token_dim = action_block * action_dim
        self.receding_horizon = receding_horizon if receding_horizon is not None else planner.horizon
        self.budget_env_steps = budget_env_steps
        self.success_fn = success_fn

    def run(
        self,
        env,
        z_goal: torch.Tensor,
        log_diagnostics: bool = True,
    ) -> dict:
        device = next(self.planner.model.parameters()).device
        model = self.planner.model

        # Phase 0 — bootstrap action/obs history with small random actions
        action_history_raw: list[np.ndarray] = []
        obs_history: list[np.ndarray] = [env.observe()]
        rng = np.random.default_rng(self.planner.seed + 1)
        warmup_steps = self.history_size * self.action_block
        for _ in range(warmup_steps):
            a = rng.uniform(-0.3, 0.3, size=self.action_dim).astype(np.float32)
            env.step(a)
            action_history_raw.append(a)
            obs_history.append(env.observe())
        env_steps_taken = warmup_steps

        success_step: int | None = None
        last_plan_tail: torch.Tensor | None = None
        diagnostics = {"states": [env.state().copy()],
                       "predicted_final_costs": []}

        while env_steps_taken < self.budget_env_steps:
            if self.success_fn is not None and self.success_fn(env):
                success_step = env_steps_taken
                break

            # Build z_history: encode the obs at strided positions matching training.
            # Take obs at offsets (env_steps - history_size*block, env_steps - (history_size-1)*block, ..., env_steps).
            stride_offsets = [env_steps_taken - h * self.action_block for h in range(self.history_size, 0, -1)]
            ctx_obs = [obs_history[max(0, off)] for off in stride_offsets]
            ctx_arr = np.stack(ctx_obs)  # (history_size, H, W, C)
            ctx_t = torch.from_numpy(ctx_arr).permute(0, 3, 1, 2).float().unsqueeze(0).to(device)
            with torch.no_grad():
                z_hist = model.encoder(ctx_t)[0]  # (history_size, D)

            # Build a_history: most recent (history_size * action_block) raw actions
            recent_actions = np.stack(
                action_history_raw[-self.history_size * self.action_block:]
            ).astype(np.float32)
            a_hist = recent_actions.reshape(self.history_size, self.token_dim)
            a_hist_t = torch.from_numpy(a_hist).to(device)

            # CEM plan (warm-start from prior tail when receding < horizon)
            mu = self.planner.plan(z_hist, a_hist_t, z_goal, init_action=last_plan_tail)
            plan_env_actions = self.planner.unroll_plan(
                mu, self.action_block, self.action_dim
            )

            # Execute receding_horizon * action_block env steps
            exec_count = self.receding_horizon * self.action_block
            steps_to_run = min(exec_count, self.budget_env_steps - env_steps_taken)
            for k in range(steps_to_run):
                a = plan_env_actions[k]
                env.step(a)
                action_history_raw.append(a.astype(np.float32))
                obs_history.append(env.observe())
                env_steps_taken += 1
                if log_diagnostics:
                    diagnostics["states"].append(env.state().copy())
                if self.success_fn is not None and self.success_fn(env):
                    success_step = env_steps_taken
                    break
            if success_step is not None:
                break

            # Warm-start setup
            if self.receding_horizon < self.planner.horizon:
                tail = mu[self.receding_horizon:]
                pad = torch.zeros(self.receding_horizon, self.token_dim,
                                  device=device, dtype=mu.dtype)
                last_plan_tail = torch.cat([tail, pad], dim=0)
            else:
                last_plan_tail = None

        return {
            "success_step": success_step,
            "env_steps_taken": env_steps_taken,
            "final_state": env.state().copy(),
            "final_obs": env.observe(),
            "diagnostics": diagnostics,
        }
