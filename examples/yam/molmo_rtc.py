"""Training-free real-time chunking (RTC) for MolmoAct2's flow-matching action expert.

Port of evo/openpi's RTC (`src/openpi/models/pi0.py::sample_actions` + `src/openpi/models/rtc.py`), i.e.
Black et al. 2025, "Real-Time Execution of Action Chunking Flow Policies": while the robot executes the
current chunk, the next one is sampled with a guidance term that pulls its first actions toward the
previous chunk's not-yet-executed tail, so the swap is continuous and the arm never waits.

MolmoAct2 integrates t from 0 (noise) to 1 (data), the paper's convention, so per Euler step

    x1_hat = x_t + (1 - t) v
    v     <- v + w(t) * (dx1_hat/dx_t)^T [W * (Y - x1_hat)]          (a VJP through the action expert)
    w(t)   = min(beta, (1 - t) / t / r2(t)),   r2(t) = (1 - t)^2 / (t^2 + (1 - t)^2)

evo writes the same thing with t' = 1 - t and a negative dt, hence its minus sign.

method="inpaint" (default) skips the VJP: each Euler step pins the weighted prefix of x_t to the
leftover noised to level t, x_t <- x_t + W * ((1 - t) eps + t Y - x_t), and the final chunk is blended
toward Y with the same weights so the executed overlap hands over smoothly from the old plan. Forward
passes only, so it costs about the same as plain inference and allows a small inference_delay. With the
guided method on a 30-step chunk the ~400 ms VJP latency forced d=14, which left the executed actions
unconstrained and made the arm lurch at every swap.

Installed on a loaded model *instance*: the cached modeling_molmoact2.py is not edited (see CLAUDE.md).
The VLM forward and everything outside the flow loop are unchanged; only RTC requests take the guided
loop, and they must run without the action CUDA graph (the graph replays the unguided loop).
"""

from __future__ import annotations

import dataclasses
import math
import threading
from typing import Any

import numpy as np
import torch


def prefix_weights(inference_delay: int, overlap: int, horizon: int, schedule: str = "linear",
                   device: Any = None) -> torch.Tensor:
    """Per-position weights, identical to evo's `rtc.get_prefix_weights`: 1 for the first
    `inference_delay` actions (executed while this chunk is computed), decaying to 0 at `overlap`
    (the end of the previous chunk's leftover), 0 after."""
    idx = torch.arange(horizon, dtype=torch.float32, device=device)
    start, end = min(inference_delay, overlap), overlap
    if schedule == "ones":
        w = torch.ones(horizon, device=device)
    elif schedule == "zeros":
        w = (idx < start).float()
    elif schedule in ("linear", "exp"):
        w = ((start - 1 - idx) / (end - start + 1) + 1).clamp(0, 1)
        if schedule == "exp":
            w = w * torch.expm1(w) / (math.e - 1)
    else:
        raise ValueError(f"unknown prefix schedule {schedule!r}")
    return torch.where(idx >= end, torch.zeros_like(w), w)


def _as_normal_tensors(x: Any) -> Any:
    """Clone tensors (recursively through dataclasses/tuples/lists) outside inference mode, so the
    action-expert context built under `torch.inference_mode` can take part in autograd."""
    if torch.is_tensor(x):
        return x.clone()
    if dataclasses.is_dataclass(x) and not isinstance(x, type):
        return dataclasses.replace(x, **{f.name: _as_normal_tensors(getattr(x, f.name)) for f in dataclasses.fields(x)})
    if isinstance(x, tuple):
        return tuple(_as_normal_tensors(v) for v in x)
    if isinstance(x, list):
        return [_as_normal_tensors(v) for v in x]
    return x


class MolmoRTC:
    """Patches `model.model` (the MolmoAct2 backbone) in place.

    - Every call records the normalized flow output (`last_trajectory`), which the client's RTC
      broker sends back as `prev_chunk_left_over` on the next request.
    - Inside `guided(...)`, the flow loop runs RTC guidance toward that leftover.
    """

    def __init__(self, model: Any, *, execution_horizon: int, inference_delay: int, action_dim: int,
                 schedule: str = "linear", max_guidance_weight: float = 10.0, method: str = "inpaint") -> None:
        if method not in ("inpaint", "guided"):
            raise ValueError(f"unknown RTC method {method!r}")
        self.method = method
        self.inner = model.model
        self.e, self.d, self.action_dim = int(execution_horizon), int(inference_delay), int(action_dim)
        self.schedule, self.beta = schedule, float(max_guidance_weight)
        self._request: tuple[np.ndarray, int] | None = None
        self.last_trajectory: torch.Tensor | None = None
        self.lock = threading.Lock()  # one RTC/plain request at a time: _request is per-call state
        self._orig_generate = self.inner.generate_actions_from_inputs
        self._orig_loop = self.inner._run_action_flow_loop
        self.inner.generate_actions_from_inputs = self._generate
        self.inner._run_action_flow_loop = self._loop

    def guided(self, prev_chunk_left_over: np.ndarray):
        """Context manager: the next predict() call is RTC-guided toward `prev_chunk_left_over`
        ((H, action_dim), normalized model space, zero-padded after the leftover)."""
        rtc = self

        class _Ctx:
            def __enter__(self):
                rtc._request = (np.asarray(prev_chunk_left_over, dtype=np.float32), rtc.d)

            def __exit__(self, *exc):
                rtc._request = None

        return _Ctx()

    def _generate(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        out = self._orig_generate(*args, **kwargs)
        self.last_trajectory = out.detach().float().cpu().clone()  # may be a CUDA-graph static buffer
        return out

    def _loop(self, inputs: Any, steps: int) -> torch.Tensor:
        if self._request is None:
            return self._orig_loop(inputs, steps)
        if self.method == "inpaint":
            return self._loop_inpaint(inputs, steps, *self._request)
        leftover, d = self._request
        inner = self.inner
        expert = inner._require_action_expert()
        mask_enabled = inner.config.mask_action_dim_padding
        with torch.inference_mode(False):
            x = inputs.trajectory.clone()
            context = _as_normal_tensors(inputs.context)
            modulations = [_as_normal_tensors(m) for m in inputs.modulations]
            pad = _as_normal_tensors(inputs.action_dim_is_pad)
            batch, horizon, width = x.shape
            target = torch.zeros((batch, horizon, width), device=x.device, dtype=torch.float32)
            n, a = min(horizon, leftover.shape[0]), min(width, leftover.shape[1], self.action_dim)
            target[:, :n, :a] = torch.as_tensor(leftover[:n, :a], device=x.device)
            overlap = horizon - self.e  # previous chunk's actions not yet executed when this one was requested
            weights = prefix_weights(d, overlap, horizon, self.schedule, device=x.device)[None, :, None]
            weights = weights * (torch.arange(width, device=x.device) < self.action_dim).float()[None, None, :]
            dt = 1.0 / steps
            for idx in range(steps):
                t = idx / steps
                with torch.enable_grad():
                    xg = x.detach().requires_grad_(True)
                    v = expert.forward_with_context(xg, modulations[idx].conditioning, context=context,
                                                    modulation=modulations[idx])
                    v = inner._mask_action_dim_tensor(v, action_dim_is_pad=pad, enabled=mask_enabled)
                    x1 = xg + (1.0 - t) * v
                    err = ((target - x1.float()) * weights).to(x1.dtype)
                    (correction,) = torch.autograd.grad(x1, xg, grad_outputs=err)
                if t > 1e-8:
                    inv_r2 = (t * t + (1 - t) ** 2) / ((1 - t) ** 2)
                    w = min((1 - t) / t * inv_r2, self.beta)
                else:
                    w = self.beta
                v = v.detach() + w * correction
                x = inner._mask_action_dim_tensor((x + dt * v).detach(), action_dim_is_pad=pad,
                                                  enabled=mask_enabled)
        return x

    def _target_and_weights(self, x: torch.Tensor, leftover: np.ndarray, d: int):
        batch, horizon, width = x.shape
        target = torch.zeros((batch, horizon, width), device=x.device, dtype=torch.float32)
        n, a = min(horizon, leftover.shape[0]), min(width, leftover.shape[1], self.action_dim)
        target[:, :n, :a] = torch.as_tensor(leftover[:n, :a], device=x.device)
        weights = prefix_weights(d, horizon - self.e, horizon, self.schedule, device=x.device)[None, :, None]
        weights = weights * (torch.arange(width, device=x.device) < self.action_dim).float()[None, None, :]
        return target.to(x.dtype), weights.to(x.dtype)

    def _loop_inpaint(self, inputs: Any, steps: int, leftover: np.ndarray, d: int) -> torch.Tensor:
        inner = self.inner
        expert = inner._require_action_expert()
        mask_enabled = inner.config.mask_action_dim_padding
        pad = inputs.action_dim_is_pad
        noise = inputs.trajectory
        target, weights = self._target_and_weights(noise, leftover, d)
        x, dt = noise, 1.0 / steps
        for idx in range(steps):
            t = idx / steps
            x = x + weights * (((1.0 - t) * noise + t * target) - x)  # pin the prefix to the leftover's noisy path
            v = expert.forward_with_context(x, inputs.modulations[idx].conditioning, context=inputs.context,
                                            modulation=inputs.modulations[idx])
            v = inner._mask_action_dim_tensor(v, action_dim_is_pad=pad, enabled=mask_enabled)
            x = inner._mask_action_dim_tensor(x + dt * v, action_dim_is_pad=pad, enabled=mask_enabled)
        # Final blend: weight 1 where the old chunk is already executing, decaying across the executed overlap.
        return inner._mask_action_dim_tensor(x + weights * (target - x), action_dim_is_pad=pad, enabled=mask_enabled)

    def raw_actions(self, n_steps: int) -> np.ndarray:
        """Normalized chunk of the last call, shaped like the returned actions: (n_steps, action_dim)."""
        if self.last_trajectory is None:
            raise RuntimeError("no trajectory recorded yet")
        return self.last_trajectory[0, :n_steps, : self.action_dim].numpy().astype(np.float32)
