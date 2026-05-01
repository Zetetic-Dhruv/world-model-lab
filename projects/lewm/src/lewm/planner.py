"""Cross-Entropy Method (CEM) planner in latent space, with MPC wrapper.

Plans action sequences a*_{1:H} that minimize ‖predict(z_init, a) − z_goal‖²
under the trained LeWM model. Predictor + pred_proj are frozen during planning.

The MPC wrapper executes the first K_exec actions of each plan and replans
from the new observation, until the goal is reached or the budget runs out.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .model import LeWM


class CEMPlanner:
    """Latent-space CEM over action sequences.

    Parameters
    ----------
    model : LeWM
        Trained model (frozen during planning).
    horizon : int
        Plan length H (number of action steps to optimize).
    n_samples : int
        Candidate sequences per CEM iteration (N).
    n_elites : int
        Top-K kept per iteration (K ≤ N).
    n_iters : int
        CEM refinement iterations (T).
    sigma_init : float
        Initial standard deviation of the proposal distribution.
    sigma_min : float
        Floor on the proposal std to avoid premature collapse.
    action_dim : int
    action_low, action_high : float
        Per-dim clamp range for sampled actions (matches env action space).
    sub_len : int
        Predictor's history-window size (must equal training-time sub_len).
        Plans are scored using the last `sub_len` positions of an
        autoregressively-rolled-out latent sequence.
    """

    def __init__(
        self,
        model: LeWM,
        horizon: int = 5,
        n_samples: int = 300,
        n_elites: int = 30,
        n_iters: int = 10,
        sigma_init: float = 1.0,
        sigma_min: float = 0.1,
        action_dim: int = 2,
        action_low: float = -1.0,
        action_high: float = 1.0,
        sub_len: int = 4,
    ):
        self.model = model
        self.horizon = horizon
        self.n_samples = n_samples
        self.n_elites = n_elites
        self.n_iters = n_iters
        self.sigma_init = sigma_init
        self.sigma_min = sigma_min
        self.action_dim = action_dim
        self.action_low = action_low
        self.action_high = action_high
        self.sub_len = sub_len

    @torch.no_grad()
    def _rollout_cost(
        self,
        z_history: torch.Tensor,
        actions: torch.Tensor,
        z_goal: torch.Tensor,
    ) -> torch.Tensor:
        """Score N candidate action sequences in batch.

        Parameters
        ----------
        z_history : (sub_len, D)
            Encoder embeddings for the last sub_len observations (post-projector).
            We use the LAST emb as the rollout starting point and feed
            (sub_len, D) into the predictor each step (same window topology as
            training).
        actions : (N, H, action_dim)
        z_goal : (D,)

        Returns
        -------
        cost : (N,) — squared distance from final predicted emb to z_goal.
        """
        N, H, A = actions.shape
        D = z_history.size(-1)
        device = z_history.device

        # Initialize per-candidate context: replicate z_history N times along batch.
        # Shape: (N, sub_len, D)
        ctx = z_history.unsqueeze(0).expand(N, -1, -1).contiguous()

        # We need the predictor's history-window action input too. We use the
        # most recent (sub_len-1) actions (zero-padded) plus the candidate's
        # current action at position H.
        # For simplicity and matching training topology: we feed the full
        # (sub_len) window of latents and a (sub_len) window of actions, and
        # the predictor outputs predictions at each position. We take position
        # -1 as the next-emb prediction.
        # We'll roll forward H steps; at each step k:
        #   - window_z = ctx[:, -sub_len:]
        #   - window_a = pad-and-shift to align: actions for the last sub_len
        #     latent positions.
        # For a fresh rollout we maintain a parallel "action window" buffer
        # that we shift each step.

        # Action window: (N, sub_len, A). We initialize to zeros and shift in
        # candidate actions one by one. This matches training-time pattern
        # where each predictor call takes the same-length history.
        a_win = torch.zeros(N, self.sub_len, A, device=device, dtype=actions.dtype)

        for k in range(H):
            # Shift action window left, place new action at the right
            a_win = torch.roll(a_win, shifts=-1, dims=1)
            a_win[:, -1, :] = actions[:, k, :]
            # Predict from the current context window
            window_z = ctx[:, -self.sub_len :, :]
            preds = self.model.predict(window_z, a_win)  # (N, sub_len, D)
            next_z = preds[:, -1:, :]  # (N, 1, D)
            ctx = torch.cat([ctx, next_z], dim=1)

        # Final predicted emb after H rollouts: ctx[:, -1, :]
        z_final = ctx[:, -1, :]
        # Cost = ‖z_final − z_goal‖²
        diff = z_final - z_goal.unsqueeze(0)
        cost = diff.pow(2).sum(dim=-1)
        return cost

    @torch.no_grad()
    def plan(
        self,
        obs_history: torch.Tensor,
        obs_goal: torch.Tensor,
    ) -> torch.Tensor:
        """Plan an action sequence of length H.

        Parameters
        ----------
        obs_history : (sub_len, C, H, W) — last sub_len frames in chronological order.
        obs_goal : (C, H, W) — single goal observation.

        Returns
        -------
        actions : (H, action_dim) — best plan (mean of final CEM distribution).
        """
        device = next(self.model.parameters()).device
        obs_history = obs_history.to(device)
        obs_goal = obs_goal.to(device)

        # Encode history and goal once.
        self.model.eval()
        z_history = self.model.encoder(obs_history.unsqueeze(0))[0]  # (sub_len, D)
        z_goal = self.model.encoder(obs_goal.unsqueeze(0).unsqueeze(0))[0, 0]  # (D,)

        # Initialize CEM proposal distribution
        H, A = self.horizon, self.action_dim
        mu = torch.zeros(H, A, device=device, dtype=z_history.dtype)
        sigma = torch.full((H, A), self.sigma_init,
                           device=device, dtype=z_history.dtype)

        for _ in range(self.n_iters):
            # Sample N candidates
            samples = mu.unsqueeze(0) + sigma.unsqueeze(0) * torch.randn(
                self.n_samples, H, A, device=device, dtype=z_history.dtype
            )
            samples = samples.clamp_(self.action_low, self.action_high)

            costs = self._rollout_cost(z_history, samples, z_goal)  # (N,)
            # Top-K elites (lowest cost)
            elite_idx = torch.topk(costs, self.n_elites, largest=False).indices
            elites = samples[elite_idx]  # (K, H, A)

            mu = elites.mean(dim=0)
            sigma = elites.std(dim=0).clamp_min(self.sigma_min)

        return mu  # (H, A)


class MPCRunner:
    """Receding-horizon MPC: plan H actions, execute K_exec, replan.

    Stops on goal-reach or budget exhaustion. Returns success flag and
    per-step trajectory (states/observations) for analysis.
    """

    def __init__(
        self,
        planner: CEMPlanner,
        env_factory,  # callable returning a fresh env
        sub_len: int = 4,
        k_exec: int | None = None,
        budget_steps: int = 50,
        success_fn=None,  # callable(env) -> bool; default: env.is_success()
    ):
        self.planner = planner
        self.env_factory = env_factory
        self.sub_len = sub_len
        self.k_exec = k_exec if k_exec is not None else planner.horizon
        self.budget_steps = budget_steps
        self.success_fn = success_fn or (lambda env: env.is_success())

    def run(self, env=None) -> dict:
        import numpy as np
        env = env or self.env_factory()
        obs_history = [env.observe()]
        # Pad initial history by repeating the first frame.
        while len(obs_history) < self.sub_len:
            obs_history.append(obs_history[-1])

        states = [env.state()]
        steps_taken = 0
        success = False

        while steps_taken < self.budget_steps:
            if self.success_fn(env):
                success = True
                break
            # Build context tensor (sub_len, C, H, W)
            ctx_imgs = np.stack(obs_history[-self.sub_len :])
            ctx_imgs = np.transpose(ctx_imgs, (0, 3, 1, 2))
            ctx = torch.from_numpy(ctx_imgs).float()

            goal_img = env.goal_observation()
            goal = torch.from_numpy(goal_img).permute(2, 0, 1).float()

            plan = self.planner.plan(ctx, goal).cpu().numpy()  # (H, A)

            # Execute up to k_exec steps
            for k in range(min(self.k_exec, self.budget_steps - steps_taken)):
                obs = env.step(plan[k])
                obs_history.append(obs)
                states.append(env.state())
                steps_taken += 1
                if self.success_fn(env):
                    success = True
                    break
            if success:
                break

        return {
            "success": success,
            "steps_taken": steps_taken,
            "final_state": env.state(),
            "states": np.stack(states),
        }
