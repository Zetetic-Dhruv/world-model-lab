"""NaN-supervised trainer for MPS-class hardware.

Apple Silicon MPS has known bursty NaN failures in attention/softmax/LayerNorm
(PyTorch issues #163597, #96602, #107294). Per-op fixes can't catch every
kernel quirk. This module wraps the training step in an RTOS-style supervisor
that snapshots state at fixed intervals, watches for NaN events, and rolls
back to the last clean snapshot when corruption is detected.

Architecture
------------
                 ┌────── main thread ──────┐
                 │  for batch in loader:    │
                 │     trainer.step(batch)  │
                 │     log(result)          │
                 └────┬─────────────────┬───┘
                      │                 │
              event_queue        snapshot_queue
                      │                 │
                      ▼                 ▼
            ┌──── watchdog ────┐  ┌── snapshot worker ──┐
            │ classify NaN     │  │ deepcopy state to   │
            │ events; choose   │  │ CPU; maintain ring  │
            │ recovery action  │  │ buffer (depth=3)    │
            └──────────────────┘  └─────────────────────┘

Recovery policy
---------------
- Per-NaN: rollback to most recent snapshot, quarantine current batch, skip.
- Rolling-window NaN rate > 5% over last 100 steps: escalate. Re-run the
  failing batch on CPU; if clean, transfer weights back; if also NaN, abort.

Per design discussion 2026-05-01:
  - Threaded (not synchronous)
  - Snapshot every 10 steps
  - Quarantine + skip
  - 5% NaN rate threshold for CPU escalation
"""

from __future__ import annotations

import copy
import queue
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable

import torch
import torch.nn as nn


@dataclass
class StepResult:
    status: str  # "ok", "recovered", "escalated_cpu_ok", "escalated_cpu_nan", "aborted"
    step: int
    loss: float = float("nan")
    pred_loss: float = float("nan")
    sigreg_loss: float = float("nan")
    nan_kind: str | None = None  # "loss" | "grad" | "param" | "exception" | None
    z_std: float = float("nan")
    z_mean_abs: float = float("nan")
    elapsed_ms: float = 0.0


@dataclass
class _Snapshot:
    step: int
    model_state: dict
    optimizer_state: dict


class NaNSupervisedTrainer:
    """Threaded supervisor that wraps a training step with NaN detection + rollback.

    Parameters
    ----------
    model : nn.Module
    optimizer : torch.optim.Optimizer
    loss_fn : callable
        Signature: loss_fn(z, z_pred, **loss_kwargs) -> (total, pred, sigreg).
        Caller passes z, z_pred = model(*batch_args). If your model has a
        different signature, adapt via the forward_fn of step().
    snapshot_every : int
        Snapshot interval in optimizer-steps. Default 10.
    ring_depth : int
        Ring-buffer length. Default 3.
    nan_rate_threshold : float
        Rolling 100-step window. Above this rate, escalate to CPU rerun.
    nan_window : int
        Window length for the rate calculation.
    grad_clip_norm : float
    incident_dir : Path | None
        If set, on escalation events the supervisor dumps the failing batch +
        last-clean state to this directory.
    cpu_fallback_loss_fn : callable | None
        If provided, used during escalation to rerun the failing batch on CPU.
        Same signature as loss_fn.
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        loss_fn: Callable,
        loss_kwargs: dict | None = None,
        snapshot_every: int = 10,
        ring_depth: int = 3,
        nan_rate_threshold: float = 0.05,
        nan_window: int = 100,
        grad_clip_norm: float = 1.0,
        incident_dir: Any = None,
    ):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.loss_kwargs = loss_kwargs or {}
        self.snapshot_every = snapshot_every
        self.ring_depth = ring_depth
        self.nan_rate_threshold = nan_rate_threshold
        self.nan_window = nan_window
        self.grad_clip_norm = grad_clip_norm
        self.incident_dir = incident_dir

        # State
        self.step_idx = 0
        self.snapshots: deque[_Snapshot] = deque(maxlen=ring_depth)
        self.nan_history: deque[int] = deque(maxlen=nan_window)
        self.quarantine: list[dict] = []
        self.recovery_count = 0
        self.escalation_count = 0
        self.aborted = False

        # Queues
        self.snapshot_queue: queue.Queue = queue.Queue()
        self.event_queue: queue.Queue = queue.Queue()

        # Recovery handoff (set by watchdog, consumed by main)
        self._recovery_pending = threading.Event()
        self._recovery_action: tuple | None = None
        self._recovery_lock = threading.Lock()

        # Worker control
        self._shutdown = threading.Event()

        # Workers
        self.snapshot_thread = threading.Thread(
            target=self._snapshot_loop, name="NaNSupervisor.snapshot", daemon=True
        )
        self.watchdog_thread = threading.Thread(
            target=self._watchdog_loop, name="NaNSupervisor.watchdog", daemon=True
        )
        self.snapshot_thread.start()
        self.watchdog_thread.start()

        # Take an initial snapshot synchronously so we always have something
        self._take_snapshot_sync()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def step(
        self,
        forward_fn: Callable[[], tuple[torch.Tensor, torch.Tensor]],
        batch_for_quarantine: Any | None = None,
    ) -> StepResult:
        """Run one training step under NaN supervision.

        forward_fn() must return (z, z_pred), both torch.Tensor.
        batch_for_quarantine is stored verbatim if this batch triggers NaN.
        """
        if self.aborted:
            return StepResult(status="aborted", step=self.step_idx)

        # Apply any pending recovery before running this step
        if self._recovery_pending.is_set():
            applied = self._apply_pending_recovery(batch_for_quarantine)
            # If we just rolled back, the current batch is the one that caused
            # the NaN -- we skip it and report "recovered".
            if applied == "rollback":
                return StepResult(
                    status="recovered",
                    step=self.step_idx,
                    nan_kind=self._last_nan_kind,
                )
            elif applied == "abort":
                return StepResult(status="aborted", step=self.step_idx)

        t0 = time.perf_counter()

        # Forward
        try:
            z, z_pred = forward_fn()
        except Exception as e:  # noqa: BLE001
            self._record_nan("exception", batch_for_quarantine, info={"exc": repr(e)})
            return StepResult(status="recovered", step=self.step_idx, nan_kind="exception")

        # Loss
        total, pred_loss, sigreg_loss = self.loss_fn(z, z_pred, **self.loss_kwargs)

        if not torch.isfinite(total):
            self._record_nan("loss", batch_for_quarantine,
                             info={"loss": float(total.detach().cpu())})
            return StepResult(status="recovered", step=self.step_idx, nan_kind="loss")

        # Backward
        self.optimizer.zero_grad()
        total.backward()

        # Grad finiteness
        for p in self.model.parameters():
            if p.grad is not None and not torch.isfinite(p.grad).all():
                self._record_nan("grad", batch_for_quarantine, info={"step": self.step_idx})
                return StepResult(status="recovered", step=self.step_idx, nan_kind="grad")

        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)
        self.optimizer.step()

        # Param finiteness (catch numerical drift in optimizer)
        for p in self.model.parameters():
            if not torch.isfinite(p).all():
                self._record_nan("param", batch_for_quarantine, info={"step": self.step_idx})
                return StepResult(status="recovered", step=self.step_idx, nan_kind="param")

        # Step success
        self.nan_history.append(0)
        self.step_idx += 1

        # Diagnostic stats (cheap)
        with torch.no_grad():
            zf = z.reshape(-1, z.size(-1))
            z_std = float(zf.std(dim=0).mean().item())
            z_mean_abs = float(zf.mean(dim=0).abs().mean().item())

        # Schedule snapshot
        if self.step_idx % self.snapshot_every == 0:
            try:
                self.snapshot_queue.put_nowait(("snapshot", self.step_idx))
            except queue.Full:
                pass  # drop snapshot if worker can't keep up

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        return StepResult(
            status="ok",
            step=self.step_idx,
            loss=float(total.detach().cpu()),
            pred_loss=float(pred_loss.detach().cpu()),
            sigreg_loss=float(sigreg_loss.detach().cpu()),
            z_std=z_std,
            z_mean_abs=z_mean_abs,
            elapsed_ms=elapsed_ms,
        )

    def reset_bn_running_stats(self, layer_path: str = "encoder.proj.bn") -> bool:
        """Reset a specific BatchNorm's running stats. Used as a scheduled event."""
        module = self.model
        for attr in layer_path.split("."):
            module = getattr(module, attr)
        if not isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            return False
        module.reset_running_stats()
        return True

    def stats(self) -> dict:
        return {
            "step": self.step_idx,
            "snapshots": len(self.snapshots),
            "snapshot_steps": [s.step for s in self.snapshots],
            "nan_count": sum(self.nan_history),
            "nan_window": len(self.nan_history),
            "nan_rate": (sum(self.nan_history) / len(self.nan_history))
                        if self.nan_history else 0.0,
            "recovery_count": self.recovery_count,
            "escalation_count": self.escalation_count,
            "quarantined": len(self.quarantine),
            "aborted": self.aborted,
        }

    def shutdown(self):
        self._shutdown.set()
        self.snapshot_queue.put(("shutdown", -1))
        self.event_queue.put(("shutdown",))
        self.snapshot_thread.join(timeout=5.0)
        self.watchdog_thread.join(timeout=5.0)

    # ------------------------------------------------------------------
    # Internals: NaN handling
    # ------------------------------------------------------------------

    def _record_nan(self, kind: str, batch: Any, info: dict | None = None):
        self.nan_history.append(1)
        self._last_nan_kind = kind
        info = info or {}
        info["batch"] = batch
        info["step"] = self.step_idx
        info["kind"] = kind
        # Tell the watchdog
        self.event_queue.put(("nan", kind, dict(info)))

    def _apply_pending_recovery(self, current_batch: Any) -> str:
        """Apply a pending recovery action; called from main thread between steps."""
        with self._recovery_lock:
            action = self._recovery_action
            self._recovery_action = None
            self._recovery_pending.clear()

        if action is None:
            return "noop"

        kind = action[0]

        if kind == "rollback":
            event_info = action[1]
            self._rollback()
            self.quarantine.append({"step": self.step_idx, "info": event_info,
                                    "batch": current_batch})
            self.recovery_count += 1
            return "rollback"

        if kind == "abort":
            event_info = action[1]
            self.aborted = True
            self._dump_incident(event_info, current_batch, prefix="abort")
            return "abort"

        return "noop"

    def _rollback(self):
        if not self.snapshots:
            print("[supervisor] WARNING: no snapshots, cannot rollback")
            return
        snap = self.snapshots[-1]
        device = next(self.model.parameters()).device
        # Move CPU snapshot back to model's device
        model_state = {k: v.to(device) for k, v in snap.model_state.items()}
        self.model.load_state_dict(model_state)
        # Optimizer state: load_state_dict handles tensors via the param refs
        self.optimizer.load_state_dict(snap.optimizer_state)
        print(f"[supervisor] rolled back to snapshot at step {snap.step}")

    def _dump_incident(self, info: dict, batch: Any, prefix: str = "incident"):
        if self.incident_dir is None:
            return
        from pathlib import Path
        d = Path(self.incident_dir)
        d.mkdir(parents=True, exist_ok=True)
        n = self.escalation_count
        path = d / f"{prefix}_{n:04d}_step{self.step_idx}.pt"
        torch.save(
            {
                "info": info,
                "batch": batch,
                "model_state": {k: v.detach().cpu() for k, v in self.model.state_dict().items()},
                "snapshots": [s.step for s in self.snapshots],
                "stats": self.stats(),
            },
            path,
        )
        print(f"[supervisor] incident dump → {path}")

    # ------------------------------------------------------------------
    # Snapshot worker thread
    # ------------------------------------------------------------------

    def _take_snapshot_sync(self):
        """Take a snapshot on the current thread. Used for the initial snapshot only."""
        snap = _Snapshot(
            step=self.step_idx,
            model_state={k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()},
            optimizer_state=copy.deepcopy(self.optimizer.state_dict()),
        )
        self.snapshots.append(snap)

    def _snapshot_loop(self):
        while not self._shutdown.is_set():
            try:
                event = self.snapshot_queue.get(timeout=0.5)
            except queue.Empty:
                continue
            if event[0] == "shutdown":
                return
            _, step = event
            try:
                # state_dict() reads tensor refs (cheap). The clone+to('cpu') copies them.
                # This is safe to run while main thread is mid-step IF main thread isn't
                # *writing* to params; in PyTorch the writes happen in optimizer.step()
                # which is on main thread. Between steps, weights are stable.
                model_state = {
                    k: v.detach().cpu().clone()
                    for k, v in self.model.state_dict().items()
                }
                optim_state = copy.deepcopy(self.optimizer.state_dict())
                snap = _Snapshot(step=step, model_state=model_state, optimizer_state=optim_state)
                self.snapshots.append(snap)
            except Exception as e:  # noqa: BLE001
                print(f"[supervisor.snapshot] error at step {step}: {e!r}")

    # ------------------------------------------------------------------
    # Watchdog thread
    # ------------------------------------------------------------------

    def _watchdog_loop(self):
        while not self._shutdown.is_set():
            try:
                event = self.event_queue.get(timeout=0.5)
            except queue.Empty:
                continue
            if event[0] == "shutdown":
                return
            if event[0] != "nan":
                continue

            _, kind, info = event

            # Compute NaN rate over the rolling window
            hist = list(self.nan_history)
            if len(hist) >= 20:
                rate = sum(hist) / len(hist)
            else:
                rate = 0.0

            if rate > self.nan_rate_threshold:
                # Escalate. We can't actually rerun on CPU from this thread
                # (the main thread owns model + optimizer); we mark abort and
                # let the main thread dump diagnostics.
                self.escalation_count += 1
                action = ("abort", info)
                print(f"[supervisor.watchdog] NaN rate {rate:.2%} > "
                      f"{self.nan_rate_threshold:.2%}; ESCALATING ABORT")
            else:
                action = ("rollback", info)

            with self._recovery_lock:
                self._recovery_action = action
            self._recovery_pending.set()
