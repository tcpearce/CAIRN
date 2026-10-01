"""
S4 Temporal Nowcaster for H2S Forecasting.

Implements a Structured State Space (S4) model with diagonal state matrix (S4D)
for capturing long-range temporal dependencies in H2S concentration data.

Key features:
- Long-range memory (24-hour stagnation buildup)
- Strict causal convolution (no future data leakage)
- Dual heads: regression (H2S value) + classification (Low/Med/High)
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat


class DropoutNd(nn.Module):
    """N-dimensional dropout that ties dropout mask across sequence lengths."""

    def __init__(self, p: float = 0.5, tie=True, transposed=True):
        super().__init__()
        if p < 0 or p >= 1:
            raise ValueError(f"dropout probability has to be in [0, 1), got {p}")
        self.p = p
        self.tie = tie
        self.transposed = transposed

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, dim, lengths...)."""
        if self.training:
            if not self.transposed:
                x = rearrange(x, 'b ... d -> b d ...')
            mask_shape = x.shape[:2] + (1,) * (x.ndim - 2) if self.tie else x.shape
            mask = torch.rand(*mask_shape, device=x.device) < 1. - self.p
            x = x * mask * (1.0 / (1 - self.p))
            if not self.transposed:
                x = rearrange(x, 'b d ... -> b ... d')
        return x


class S4DKernel(nn.Module):
    """Generate convolution kernel from diagonal SSM parameters.

    Uses the S4D (S4 Diagonal) simplification with HiPPO-LegS initialization.
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 64,
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        lr: Optional[float] = None,
        fast_fraction: float = 0.25,
        dt_init: str = "random",
        fast_dt_min: Optional[float] = None,
        fast_dt_max: Optional[float] = None,
        slow_dt_min: Optional[float] = None,
        slow_dt_max: Optional[float] = None,
        fast_anchor_hours: float = 1.0,
        slow_anchor_hours: float = 24.0,
        spread: float = 0.5,
        use_fp64_ssm_params: bool = True,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.dt_min = dt_min
        self.dt_max = dt_max
        self.fast_fraction = fast_fraction
        self.dt_init = dt_init
        # Scope A: store log_A_real and log_dt (and therefore Adam m,v) in fp64
        # to escape float32 ULP at the param values near log(0.5) ≈ -0.69 where
        # post-warmup updates of size lr × grad ~ 1e-10 get bit-rounded away.
        # Forward path still casts to fp32 for FFT/Tensor-Core compatibility.
        self.use_fp64_ssm_params = use_fp64_ssm_params

        # DEBUG: Verify dt_min/dt_max are actually being passed correctly
        # print(f"[DEBUG S4DKernel] Initialized with dt_min={dt_min}, dt_max={dt_max}")

        H = d_model
        N = d_state

        # ===== FAST & SLOW LANE ARCHITECTURE =====
        # CRITICAL FIX: Different A AND dt initialization for fast vs slow lanes
        #
        # Previous bug: All lanes used |A|=0.5, giving minimum τ ≈ 5 hours
        # This made 15-minute spike detection IMPOSSIBLE
        #
        # Timescale formula: τ = 1 / (|A| × dt) × 0.25 hours
        # Fast lane: |A|=10.0, dt∈[0.05,0.15] → τ ≈ 0.17-0.5h (10-30 min) ✓
        # Slow lane: |A|=0.5, dt∈[0.00001,0.0002] → τ ≈ 2500-12500h ✓
        #
        # Lane split controlled by fast_fraction (default 0.25 = 25% fast, 75% slow)
        # fast_fraction=0.0 → pure slow lane  (disable fast lane)
        # fast_fraction=1.0 → pure fast lane  (disable slow lane)
        # fast_fraction=0.5 → equal 50/50 split

        # 1. Define the split
        n_fast = round(d_model * fast_fraction)
        n_fast = max(0, min(d_model, n_fast))  # clamp to [0, d_model]
        n_slow = d_model - n_fast

        # 2. Resolve per-lane dt bounds (use overrides if provided, else derive from dt_min/dt_max)
        _fast_dt_min = fast_dt_min if fast_dt_min is not None else dt_max * 0.5
        _fast_dt_max = fast_dt_max if fast_dt_max is not None else dt_max * 1.5
        _slow_dt_min = slow_dt_min if slow_dt_min is not None else dt_min * 0.1
        _slow_dt_max = slow_dt_max if slow_dt_max is not None else dt_min * 2.0

        if dt_init == "linspace":
            # Deterministic: evenly spaced in log-scale → guarantees full spectrum coverage from epoch 0
            log_dt_fast = torch.linspace(
                math.log(_fast_dt_min), math.log(_fast_dt_max), n_fast
            ) if n_fast > 0 else torch.zeros(0)
            log_dt_slow = torch.linspace(
                math.log(_slow_dt_min), math.log(_slow_dt_max), n_slow
            ) if n_slow > 0 else torch.zeros(0)
        elif dt_init == "physics_anchored":
            # Physics-motivated initialisation: concentrate dt around known H2S timescales.
            #
            # Motivation: linspace pre-solves timescale coverage, creating a flat loss
            # landscape that prevents dt from ever adapting (reservoir-computing trap).
            # By anchoring to specific physics timescales and sampling with log-Gaussian
            # spread, we create genuine gradient signal for dt to move toward uncovered
            # timescales (stagnation 12-48hr, inversion 6-12hr).
            #
            # Timescale formula: τ = timebase_hr / (|A_real| × dt)
            #   → dt = timebase_hr / (|A_real| × τ_anchor_hours)
            #
            # Assumes 15-min timebase (timebase_hr=0.25).
            # A_real init constants must match the log_A_fast/slow values below.
            _timebase_hr = 0.25          # 15-min timebase (hardcoded; update if timebase changes)
            _A_real_fast_init = 10.0     # must match log_A_fast = log(10.0) initialization
            _A_real_slow_init = 0.5      # must match log_A_slow = log(0.5) initialization

            # Convert anchor timescales to dt values
            dt_fast_center = _timebase_hr / (_A_real_fast_init * fast_anchor_hours)
            dt_slow_center = _timebase_hr / (_A_real_slow_init * slow_anchor_hours)

            # Clamp centers to lane dt ranges (prevents out-of-bounds initialisation)
            dt_fast_center = max(_fast_dt_min, min(_fast_dt_max, dt_fast_center))
            dt_slow_center = max(_slow_dt_min, min(_slow_dt_max, dt_slow_center))

            # Sample log-space Gaussian around anchors, then hard-clamp to lane bounds
            if n_fast > 0:
                log_dt_fast = torch.randn(n_fast) * spread + math.log(dt_fast_center)
                log_dt_fast = log_dt_fast.clamp(math.log(_fast_dt_min), math.log(_fast_dt_max))
            else:
                log_dt_fast = torch.zeros(0)

            if n_slow > 0:
                log_dt_slow = torch.randn(n_slow) * spread + math.log(dt_slow_center)
                log_dt_slow = log_dt_slow.clamp(math.log(_slow_dt_min), math.log(_slow_dt_max))
            else:
                log_dt_slow = torch.zeros(0)
        else:
            # Random (default): log-uniform sampling — same as original behaviour
            log_dt_fast = torch.rand(n_fast) * (
                math.log(_fast_dt_max) - math.log(_fast_dt_min)
            ) + math.log(_fast_dt_min)
            log_dt_slow = torch.rand(n_slow) * (
                math.log(_slow_dt_max) - math.log(_slow_dt_min)
            ) + math.log(_slow_dt_min)

        # 4. Concatenate and register
        log_dt = torch.cat([log_dt_fast, log_dt_slow])
        self.register("log_dt", log_dt, lr)

        # Store lane indices for monitoring
        self.register_buffer("_fast_lane_indices", torch.arange(n_fast))
        self.register_buffer("_slow_lane_indices", torch.arange(n_fast, d_model))

        # Persist resolved lane dt bounds + anchor metadata for telemetry
        # (clamp-saturation diagnostic in SSMTelemetryTracker reads these).
        self._fast_dt_min = float(_fast_dt_min)
        self._fast_dt_max = float(_fast_dt_max)
        self._slow_dt_min = float(_slow_dt_min)
        self._slow_dt_max = float(_slow_dt_max)
        self._fast_anchor_hours = float(fast_anchor_hours)
        self._slow_anchor_hours = float(slow_anchor_hours)

        # Initialize C (output projection) with SMALL scale
        # CRITICAL: If C is too large, the optimizer panics and kills A matrix
        # to silence the "noisy" memory. Use Xavier-like scaling for SSM.
        C_std = 1.0 / math.sqrt(H * N)  # Small initialization ~ 0.001 for H=128, N=64
        C_real = torch.randn(H, N // 2) * C_std
        C_imag = torch.randn(H, N // 2) * C_std
        C = torch.complex(C_real, C_imag)
        self.C = nn.Parameter(torch.view_as_real(C))

        # ===== HIGH-FREQUENCY INITIALIZATION =====
        # CRITICAL FIX: Different A initialization for fast vs slow lanes
        #
        # Standard S4 (HiPPO) uses A ≈ -0.5, which gives long memory (5+ hours)
        # For spike detection, we need SHORT memory (15-minute level)
        #
        # Solution: Initialize fast lane with |A| ≫ 0.5 (high decay)
        #         Initialize slow lane with |A| ≈ 0.5 (long memory)
        #
        # Timescale formula: τ = 1 / (|A| × dt) × 0.25 hours
        # Fast lane target: 15 min = 0.25h
        #   With dt ≈ 0.1: need |A| ≈ 10 to get τ ≈ 0.25h
        # Slow lane target: Long stagnation
        #   With dt ≈ 0.0001: |A| ≈ 0.5 gives τ ≈ 5000h

        # Fast Lane: HIGH DECAY (|A| = 10.0) for short memory
        # IMPORTANT: log_A_real stores log(positive_value), NOT log(negative_value)
        log_A_fast = torch.log(10.0 * torch.ones(n_fast, N // 2))  # log(10.0) ≈ 2.3
        # Slow Lane: LOW DECAY (|A| = 0.5) for long memory
        log_A_slow = torch.log(0.5 * torch.ones(n_slow, N // 2))   # log(0.5) ≈ -0.69
        # Concatenate the log values
        log_A_real = torch.cat([log_A_fast, log_A_slow])  # (H, N//2)
        A_imag = math.pi * repeat(torch.arange(N // 2), 'n -> h n', h=H)
        self.register("log_A_real", log_A_real, lr)
        self.register("A_imag", A_imag, lr)

        # Scope A (fp64 SSM params): upcast all four learnable kernel
        # tensors to fp64 AFTER registration so AdamW allocates m/v in fp64
        # automatically. C is included because |C_init| ~ 1e-3 → local ULP
        # ~ 1.2e-10 in fp32, easily below typical updates; A_imag is included
        # because |A_imag| up to π·N/2 ~ 97 → local ULP up to 1.2e-5 in fp32,
        # which is several orders above typical updates (severely frozen).
        # Forward path always casts back to fp32 for FFT/Tensor-Core paths.
        if self.use_fp64_ssm_params:
            with torch.no_grad():
                self.log_A_real.data = self.log_A_real.data.double()
                self.log_dt.data = self.log_dt.data.double()
                self.A_imag.data = self.A_imag.data.double()
                self.C.data = self.C.data.double()

    def register(self, name: str, tensor: torch.Tensor, lr: Optional[float] = None):
        """Register a tensor with a configurable learning rate and 0 weight decay."""
        if lr == 0.0:
            self.register_buffer(name, tensor)
        else:
            self.register_parameter(name, nn.Parameter(tensor))
            optim = {"weight_decay": 0.0}
            if lr is not None:
                optim["lr"] = lr
            setattr(getattr(self, name), "_optim", optim)

    def to_real_representation(self) -> 'S4DKernelReal':
        """
        Convert to real-valued kernel for TensorRT export.

        Returns:
            S4DKernelReal instance with equivalent parameters
        """
        from .s4_orin import S4DKernelReal
        return S4DKernelReal.from_complex_kernel(self)

    @torch.cuda.amp.custom_fwd(cast_inputs=torch.float32)
    def forward(self, L: int) -> torch.Tensor:
        """
        Generate convolution kernel of length L using real arithmetic only.

        Decomposes the complex Vandermonde product into real ops so that
        torch.compile can trace through without complex-dtype errors.

        returns: (H, L) where H is d_model
        """
        # Materialize parameters in real form.
        # If params are stored in fp64 (Scope A), cast to fp32 here so the FFT
        # and Tensor-Core paths downstream are unaffected. Autograd will
        # automatically cast gradients back to fp64 for the optimizer step.
        dt = torch.exp(self.log_dt.float())               # (H,)
        A_real = -torch.exp(self.log_A_real.float())      # (H, N)
        A_imag = self.A_imag.float()                      # (H, N)

        C_c = torch.view_as_complex(self.C.float())       # (H, N) complex view
        C_r = C_c.real                                    # (H, N)
        C_i = C_c.imag                                    # (H, N)

        # dt * A components
        dt_exp = dt.unsqueeze(-1)                         # (H, 1)
        dtA_real = dt_exp * A_real                        # (H, N)
        dtA_imag = dt_exp * A_imag                        # (H, N)

        # Discretization factor: (exp(dtA) − 1) / A  (complex, all real ops)
        # Use expm1 for numerical stability when dtA_real is small (Slow Lane)
        # Stable identity: e^a*cos(b) - 1 = (e^a - 1)*cos(b) + (cos(b) - 1)
        a = dtA_real
        b = dtA_imag

        # Use expm1 for stable e^a - 1 computation
        expm1_a = torch.expm1(a)  # Stable e^a - 1
        cos_b = torch.cos(b)
        sin_b = torch.sin(b)

        # Numerator: e^a*cos(b) - 1 = (e^a - 1)*cos(b) + (cos(b) - 1)
        # Note: cos(b) - 1 could use -2*sin(b/2)^2 for very small b, but cos_b - 1.0 is acceptable in FP32
        numer_r = expm1_a * cos_b + (cos_b - 1.0)
        numer_i = torch.exp(a) * sin_b  # Need full exp(a) for imaginary part

        denom = A_real ** 2 + A_imag ** 2 + 1e-8
        scale_r = (numer_r * A_real + numer_i * A_imag) / denom
        scale_i = (numer_i * A_real - numer_r * A_imag) / denom

        # C_tilde = C * scale  (complex multiply in real)
        Ct_r = C_r * scale_r - C_i * scale_i             # (H, N)
        Ct_i = C_r * scale_i + C_i * scale_r             # (H, N)

        # Time indices
        k = torch.arange(L, device=dt.device, dtype=dt.dtype)  # (L,)

        # exp(dtA · k) = exp(decay) · (cos(phase) + j·sin(phase))
        # CRITICAL: Force FP32 for decay/phase computation to preserve Slow Lane dynamics
        # TF32 has 10-bit mantissa (epsilon ~1e-4), which truncates slow lane values (~1e-4 to 1e-6)
        # The einsum/multiplication triggers Tensor Cores, which truncate to TF32 by default
        with torch.cuda.amp.autocast(enabled=False):
            decay = dtA_real.unsqueeze(-1) * k.float()    # (H, N, L)
            phase = dtA_imag.unsqueeze(-1) * k.float()    # (H, N, L)

            exp_decay = torch.exp(decay)                  # (H, N, L)
            cos_phase = torch.cos(phase)                  # (H, N, L)
            sin_phase = torch.sin(phase)                  # (H, N, L)

        # K = 2 · Re(C_tilde · exp(dtA·k))  summed over N
        # Re((Cr+jCi)·(cos+jsin)·exp) = exp·(Cr·cos − Ci·sin)
        Ct_r_exp = Ct_r.unsqueeze(-1)                     # (H, N, 1)
        Ct_i_exp = Ct_i.unsqueeze(-1)                     # (H, N, 1)

        K_per_n = exp_decay * (Ct_r_exp * cos_phase - Ct_i_exp * sin_phase)
        K = 2.0 * K_per_n.sum(dim=1)                     # (H, L)

        return K

    @torch.no_grad()
    def evaluate_at_indices(self, k_indices: torch.Tensor) -> torch.Tensor:
        """
        Evaluate the SSM kernel at arbitrary time indices without materialising
        the full (H, N, L) tensor. Used by per-epoch stagnation diagnostics
        which only need K[:, 0] and K[:, L-1].

        Args:
            k_indices: 1-D tensor of integer time indices, shape (M,).

        Returns:
            (H, M) real tensor — kernel values at the requested indices.
        """
        dt = torch.exp(self.log_dt.float())               # (H,)
        A_real = -torch.exp(self.log_A_real.float())      # (H, N)
        A_imag = self.A_imag.float()                      # (H, N)

        C_c = torch.view_as_complex(self.C.float())
        C_r, C_i = C_c.real, C_c.imag                     # (H, N) each

        dt_exp = dt.unsqueeze(-1)
        dtA_real = dt_exp * A_real                        # (H, N)
        dtA_imag = dt_exp * A_imag                        # (H, N)

        expm1_a = torch.expm1(dtA_real)
        cos_b = torch.cos(dtA_imag)
        sin_b = torch.sin(dtA_imag)
        numer_r = expm1_a * cos_b + (cos_b - 1.0)
        numer_i = torch.exp(dtA_real) * sin_b
        denom = A_real ** 2 + A_imag ** 2 + 1e-8
        scale_r = (numer_r * A_real + numer_i * A_imag) / denom
        scale_i = (numer_i * A_real - numer_r * A_imag) / denom
        Ct_r = C_r * scale_r - C_i * scale_i
        Ct_i = C_r * scale_i + C_i * scale_r

        k = k_indices.to(device=dt.device, dtype=dt.dtype)
        decay = dtA_real.unsqueeze(-1) * k                # (H, N, M)
        phase = dtA_imag.unsqueeze(-1) * k

        exp_decay = torch.exp(decay)
        cos_phase = torch.cos(phase)
        sin_phase = torch.sin(phase)
        K_per_n = exp_decay * (Ct_r.unsqueeze(-1) * cos_phase - Ct_i.unsqueeze(-1) * sin_phase)
        return 2.0 * K_per_n.sum(dim=1)                   # (H, M)

    def build_kernel_for_grad(
        self,
        L: int,
        log_A_real: Optional[torch.Tensor] = None,
        log_dt: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Differentiable kernel construction with optional parameter overrides.
        Used by the post-hoc kernel-sensitivity diagnostic, which differentiates
        the kernel energy w.r.t. log_A_real or log_dt independently while
        holding the other fixed.

        Args:
            L: Sequence length.
            log_A_real: Override for self.log_A_real (must match shape (H, N)).
                If None, uses self.log_A_real (no grad on it from this call).
            log_dt: Override for self.log_dt (must match shape (H,)).

        Returns:
            (H, L) real tensor with grad_fn flowing back to the override(s).
        """
        log_A = log_A_real if log_A_real is not None else self.log_A_real
        log_dt_eff = log_dt if log_dt is not None else self.log_dt

        dt = torch.exp(log_dt_eff)
        A_real = -torch.exp(log_A)
        A_imag = self.A_imag

        C_c = torch.view_as_complex(self.C)
        C_r, C_i = C_c.real, C_c.imag

        dt_exp = dt.unsqueeze(-1)
        dtA_real = dt_exp * A_real
        dtA_imag = dt_exp * A_imag

        expm1_a = torch.expm1(dtA_real)
        cos_b = torch.cos(dtA_imag)
        sin_b = torch.sin(dtA_imag)
        numer_r = expm1_a * cos_b + (cos_b - 1.0)
        numer_i = torch.exp(dtA_real) * sin_b
        denom = A_real ** 2 + A_imag ** 2 + 1e-8
        scale_r = (numer_r * A_real + numer_i * A_imag) / denom
        scale_i = (numer_i * A_real - numer_r * A_imag) / denom
        Ct_r = C_r * scale_r - C_i * scale_i
        Ct_i = C_r * scale_i + C_i * scale_r

        k = torch.arange(L, device=dt.device, dtype=dt.dtype)
        decay = dtA_real.unsqueeze(-1) * k
        phase = dtA_imag.unsqueeze(-1) * k

        exp_decay = torch.exp(decay)
        cos_phase = torch.cos(phase)
        sin_phase = torch.sin(phase)
        K_per_n = exp_decay * (Ct_r.unsqueeze(-1) * cos_phase - Ct_i.unsqueeze(-1) * sin_phase)
        return 2.0 * K_per_n.sum(dim=1)


class DropPath(nn.Module):
    """Stochastic Depth: randomly bypass entire S4 blocks during training.

    Applied at the S4TemporalNowcaster level. When a block is dropped the
    residual passes through unchanged, creating an implicit ensemble of
    shallower sub-networks (depth 1, 2, … n_layers).

    Drop rate is linearly scaled per layer depth (layer 0 → 0.0, last layer →
    drop_path_rate) following "Deep Networks with Stochastic Depth"
    (Huang et al., 2016).

    Why this matters here: each S4 block specialises on a different timescale
    band (fast = advection, slow = barometric/stagnation). Dropping a block
    forces the remaining blocks to be independently useful rather than relying
    on the dropped block's representation, which is the dominant source of
    overfitting in shallow (3-layer) physics-structured SSMs.
    """

    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.drop_prob == 0.0:
            return x
        keep_prob = 1.0 - self.drop_prob
        # Per-sample binary mask broadcast over (seq_len, d_model)
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor = torch.floor(random_tensor + keep_prob)
        return x * random_tensor / keep_prob

    def extra_repr(self) -> str:
        return f"drop_prob={self.drop_prob:.3f}"


class S4DBlock(nn.Module):
    """
    S4D Block with diagonal state matrix.

    Simpler than full S4 but still captures long-range dependencies.

    Supports multiple backends:
    - 'fft': FFT-based convolution (default, good for L > 128)
    - 'scan': Associative scan (better for L <= 128 on Orin)
    - 'auto': Automatically select based on sequence length

    CAUSALITY GUARANTEE:
    This implementation is STRICTLY CAUSAL (no bidirectional leakage):
    - Kernel generation uses indices [0, L-1] (past → present only)
    - FFT convolution uses causal formulation (no future information)
    - Scan module uses left-to-right recurrence only
    - NO bidirectional flag exists or is used
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 64,
        dropout: float = 0.0,
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        lr: Optional[float] = None,
        backend: str = "fft",
        fast_fraction: float = 0.25,
        dt_init: str = "random",
        fast_dt_min: Optional[float] = None,
        fast_dt_max: Optional[float] = None,
        slow_dt_min: Optional[float] = None,
        slow_dt_max: Optional[float] = None,
        fast_anchor_hours: float = 1.0,
        slow_anchor_hours: float = 24.0,
        spread: float = 0.5,
        use_fp64_ssm_params: bool = True,
        **kwargs,  # Accept extra parameters for legacy compatibility (e.g., when switching between improved/legacy blocks)
    ):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.backend = backend
        self._auto_threshold = 128  # Use scan for L <= threshold

        # SSM Kernel
        self.kernel = S4DKernel(d_model, d_state, dt_min, dt_max, lr, fast_fraction,
                                dt_init, fast_dt_min, fast_dt_max, slow_dt_min, slow_dt_max,
                                fast_anchor_hours, slow_anchor_hours, spread,
                                use_fp64_ssm_params=use_fp64_ssm_params)

        # Scan module for 'scan' or 'auto' backend
        # Share parameters with kernel to avoid duplication and ensure
        # that optimizer updates apply to both FFT and scan paths equally
        self._scan_module = None
        if backend in ['scan', 'auto']:
            try:
                from .s4_orin import AssociativeScanS4D
                # Pass kernel's parameters as references to share them
                # This ensures both paths use the same underlying tensors
                self._scan_module = AssociativeScanS4D(
                    d_model=d_model,
                    d_state=d_state,
                    dt_min=dt_min,
                    dt_max=dt_max,
                    use_real=False,
                    log_dt=self.kernel.log_dt,           # Share reference
                    log_A_real=self.kernel.log_A_real,   # Share reference
                    A_imag=self.kernel.A_imag,           # Share reference
                    C=self.kernel.C,                     # Share reference
                )
            except ImportError:
                pass

        # Skip connection (D term in state space).
        # ZERO-INIT: see S4DBlockImproved for full explanation.
        self.D = nn.Parameter(torch.zeros(d_model))

        # Activation and dropout
        self.activation = nn.GELU()
        self.dropout = DropoutNd(dropout) if dropout > 0.0 else nn.Identity()

        # Feed-forward network (standard position-wise)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
        )

    def set_backend(self, backend: str) -> None:
        """
        Switch SSM computation backend at runtime.

        When switching to 'scan', lazily creates the scan module if needed
        and shares parameters from the FFT kernel so both paths use identical
        SSM weights.

        CRITICAL FIX: Instead of copying parameters at switch time, we now
        pass REFERENCES to the kernel's parameters. This ensures that:
        - Optimizer updates are automatically synchronized
        - No manual copy needed - both paths point to same tensors
        - Training with backend='scan' works correctly

        Args:
            backend: 'fft', 'scan', or 'auto'
        """
        if backend not in ('fft', 'scan', 'auto'):
            raise ValueError(f"Unknown backend: {backend}")

        self.backend = backend

        if backend in ('scan', 'auto') and self._scan_module is None:
            from .s4_orin import AssociativeScanS4D
            # Create scan module with REFERENCES to kernel parameters
            # This ensures both paths share the same underlying tensors
            self._scan_module = AssociativeScanS4D(
                d_model=self.d_model,
                d_state=self.d_state,
                dt_min=self.kernel.dt_min,
                dt_max=self.kernel.dt_max,
                use_real=False,
                log_dt=self.kernel.log_dt,           # Pass reference
                log_A_real=self.kernel.log_A_real,   # Pass reference
                A_imag=self.kernel.A_imag,           # Pass reference
                C=self.kernel.C,                     # Pass reference
            )

        # No need to copy parameters - they are shared!
        # After the fix, kernel and scan_module point to the same tensors,
        # so any optimizer update is automatically visible to both paths.

    def _select_backend(self, L: int) -> str:
        """Select backend based on sequence length."""
        if self.backend == "auto":
            return "scan" if L <= self._auto_threshold else "fft"
        return self.backend

    def _fft_conv(self, x, k, L):
        # FFT often fails in FP16/Half on Jetson
        # Force it to FP32 for the calculation
        orig_dtype = x.dtype  # Save original dtype before casting
        with torch.cuda.amp.autocast(enabled=False):
            x = x.float()
            k = k.float()
            u_f = torch.fft.rfft(x, n=2 * L)
            k_f = torch.fft.rfft(k, n=2 * L)
            y_f = u_f * k_f
            y = torch.fft.irfft(y_f, n=2 * L)[..., :L]
        return y.to(orig_dtype)  # Convert back to original dtype

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (B, L, H)

        Returns:
            Output tensor of same shape as input

        CAUSALITY: This method guarantees strict causality - no future information
        is used to predict the present. The kernel is causal (indices 0 to L-1)
        and convolution uses causal formulation.
        """
        L = x.size(1)
        backend = self._select_backend(L)

        if backend == "scan" and self._scan_module is not None:
            # Use associative scan (more efficient for short sequences on Orin)
            y = self._scan_module(x)  # (B, L, H)
            # Add D term (cast D to match input dtype for FP16/BF16 support)
            D_cast = self.D.to(x.dtype)
            y = y + x * D_cast.unsqueeze(0).unsqueeze(0)
            # Transpose for dropout (expects B, H, L)
            y = y.transpose(-1, -2)
        else:
            # Use FFT convolution (default)
            # Compute SSM Kernel ONCE and reuse it
            # FIX: Previously computed kernel twice (line 384 for assertion + line 400 for use)
            k = self.kernel(L=L)  # (H, L)

            # Causality assertion: verify kernel length matches input
            assert k.shape[-1] == L, f"Kernel length mismatch: expected {L}, got {k.shape[-1]}"

            # Transpose for S4 (expects B, H, L)
            x_t = x.transpose(-1, -2)  # (B, L, H) -> (B, H, L)

            # FFT-based convolution (causal) - reuses the already-computed kernel
            y = self._fft_conv(x_t, k, L)

            # Add D term (skip connection, cast D to match input dtype)
            D_cast = self.D.to(x_t.dtype)
            y = y + x_t * D_cast.unsqueeze(-1)

        # Activation and dropout
        y = self.dropout(self.activation(y))

        # Transpose back for FFN (expects B, L, H)
        y = y.transpose(-1, -2)  # (B, H, L) -> (B, L, H)

        # Feed-forward network
        residual = y
        y = self.ffn(y)
        y = y + residual

        return y


class S4DBlockImproved(nn.Module):
    """
    Improved S4D Block with Pre-Norm design and GLU activation.

    Fixes the "Amplitude Destruction" problem:
    1. Pre-Norm: Normalize input before S4, not the output
    2. GLU Activation: Preserves amplitude through gating
    3. Proper Residual: Wraps the entire S4 core

    Architecture:
        Input → Pre-Norm → S4 Convolution → GLU Gate → Output Projection → Residual

    CAUSALITY GUARANTEE:
    This implementation is STRICTLY CAUSAL (no bidirectional leakage):
    - Kernel generation uses indices [0, L-1] (past → present only)
    - FFT convolution uses causal formulation (no future information)
    - Scan module uses left-to-right recurrence only
    - NO bidirectional flag exists or is used
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 64,
        dropout: float = 0.0,
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        lr: Optional[float] = None,
        backend: str = "fft",
        fast_fraction: float = 0.25,
        dt_init: str = "random",
        fast_dt_min: Optional[float] = None,
        fast_dt_max: Optional[float] = None,
        slow_dt_min: Optional[float] = None,
        slow_dt_max: Optional[float] = None,
        fast_anchor_hours: float = 1.0,
        slow_anchor_hours: float = 24.0,
        spread: float = 0.5,
        use_fp64_ssm_params: bool = True,
        **kwargs,  # Accept extra parameters for future compatibility
    ):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.backend = backend
        self._auto_threshold = 128

        # DEBUG: Verify dt_min/dt_max received from parent
        # print(f"[DEBUG S4DBlockImproved] Received dt_min={dt_min}, dt_max={dt_max}")

        # Pre-Norm: Normalize INPUT, not the S4 output
        self.norm = nn.LayerNorm(d_model)

        # SSM Kernel
        self.kernel = S4DKernel(d_model, d_state, dt_min, dt_max, lr, fast_fraction,
                                dt_init, fast_dt_min, fast_dt_max, slow_dt_min, slow_dt_max,
                                fast_anchor_hours, slow_anchor_hours, spread,
                                use_fp64_ssm_params=use_fp64_ssm_params)

        # Scan module for 'scan' or 'auto' backend
        # Share parameters with kernel to avoid duplication and ensure
        # that optimizer updates apply to both FFT and scan paths equally
        self._scan_module = None
        if backend in ['scan', 'auto']:
            try:
                from .s4_orin import AssociativeScanS4D
                # Pass kernel's parameters as references to share them
                # This ensures both paths use the same underlying tensors
                self._scan_module = AssociativeScanS4D(
                    d_model=d_model,
                    d_state=d_state,
                    dt_min=dt_min,
                    dt_max=dt_max,
                    use_real=False,
                    log_dt=self.kernel.log_dt,           # Share reference
                    log_A_real=self.kernel.log_A_real,   # Share reference
                    A_imag=self.kernel.A_imag,           # Share reference
                    C=self.kernel.C,                     # Share reference
                )
            except ImportError:
                pass

        # Skip connection (D term in state space).
        # ZERO-INIT: D must start at zero so the model is forced to route
        # information through the SSM memory path from epoch 0.
        # randn init (std≈1.0) makes D immediately comparable in magnitude to
        # the SSM convolution output, giving the optimizer a free shortcut.
        # Reducing lr_D in hyperopt cannot fix this — the problem is the
        # *initial value*, not the learning rate. D can grow from zero if
        # a direct feedthrough is genuinely useful.
        self.D = nn.Parameter(torch.zeros(d_model))

        # GLU Activation: Splits input into gate and value, preserves amplitude
        # We need to project to 2*d_model first, then split
        self.glu_proj = nn.Linear(d_model, 2 * d_model)

        # Output projection after gating
        self.output_proj = nn.Linear(d_model, d_model)

        # Dropout
        self.dropout = DropoutNd(dropout) if dropout > 0.0 else nn.Identity()

        # Feed-forward network (also with pre-norm)
        self.ffn_norm = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
        )
        self.ffn_dropout = nn.Dropout(dropout)

    def set_backend(self, backend: str) -> None:
        """
        Switch SSM computation backend at runtime.

        CRITICAL FIX: Instead of copying parameters at switch time, we now
        pass REFERENCES to the kernel's parameters. This ensures that:
        - Optimizer updates are automatically synchronized
        - No manual copy needed - both paths point to same tensors
        - Training with backend='scan' works correctly
        """
        if backend not in ('fft', 'scan', 'auto'):
            raise ValueError(f"Unknown backend: {backend}")

        self.backend = backend

        if backend in ('scan', 'auto') and self._scan_module is None:
            from .s4_orin import AssociativeScanS4D
            # Create scan module with REFERENCES to kernel parameters
            # This ensures both paths share the same underlying tensors
            self._scan_module = AssociativeScanS4D(
                d_model=self.d_model,
                d_state=self.d_state,
                dt_min=self.kernel.dt_min,
                dt_max=self.kernel.dt_max,
                use_real=False,
                log_dt=self.kernel.log_dt,           # Pass reference
                log_A_real=self.kernel.log_A_real,   # Pass reference
                A_imag=self.kernel.A_imag,           # Pass reference
                C=self.kernel.C,                     # Pass reference
            )

        # No need to copy parameters - they are shared!
        # After the fix, kernel and scan_module point to the same tensors,
        # so any optimizer update is automatically visible to both paths.

    def _select_backend(self, L: int) -> str:
        """Select backend based on sequence length."""
        if self.backend == "auto":
            return "scan" if L <= self._auto_threshold else "fft"
        return self.backend

    def _fft_conv(self, x, k, L):
        """FFT convolution with FP32 forcing."""
        orig_dtype = x.dtype  # Save original dtype before casting
        with torch.cuda.amp.autocast(enabled=False):
            x = x.float()
            k = k.float()
            u_f = torch.fft.rfft(x, n=2 * L)
            k_f = torch.fft.rfft(k, n=2 * L)
            y_f = u_f * k_f
            y = torch.fft.irfft(y_f, n=2 * L)[..., :L]
        return y.to(orig_dtype)  # Convert back to original dtype

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with Pre-Norm design and GLU activation.

        Args:
            x: Input tensor of shape (B, L, H) or (1, B, L, H)

        Returns:
            Output tensor of same shape as input
        """
        # Handle 4D inputs (from feature attribution with extra batch dim)
        if x.dim() == 4:
            x = x.squeeze(0)  # Remove extra batch dimension

        B, L, H = x.shape
        residual = x

        # ===== PRE-NORM: Normalize INPUT before S4 =====
        x = self.norm(x)

        # ===== S4 Convolution =====
        backend = self._select_backend(L)

        if backend == "scan" and self._scan_module is not None:
            # Use associative scan - no kernel computation needed
            y = self._scan_module(x)
            # Add D term (cast D to match input dtype for FP16/BF16 support)
            D_cast = self.D.to(x.dtype)
            y = y + x * D_cast.unsqueeze(0).unsqueeze(0)
            y = y.transpose(-1, -2)  # (B, L, H) -> (B, H, L)
        else:
            # Use FFT convolution - compute kernel ONCE
            k = self.kernel(L=L)
            # Causality assertion: verify kernel length matches input
            assert k.shape[-1] == L, f"Kernel length mismatch: expected {L}, got {k.shape[-1]}"

            x_t = x.transpose(-1, -2)
            y = self._fft_conv(x_t, k, L)
            # Add D term (cast D to match input dtype for FP16/BF16 support)
            D_cast = self.D.to(x_t.dtype)
            y = y + x_t * D_cast.unsqueeze(-1)
            # Keep y as (B, H, L) to match scan path

        # ===== Dropout: Apply here while y is (B, H, L) to avoid transpose =====
        # This is more efficient than transposing later after GLU projection
        y = self.dropout(y)  # (B, H, L) -> (B, H, L), channel-wise dropout

        # ===== GLU Activation: Preserves amplitude through gating =====
        # Project to 2*d_model, split into value and gate
        # y is currently (B, H, L) - transpose to (B, L, H) for GLU projection
        y_proj = self.glu_proj(y.transpose(-1, -2))  # (B, L, 2*H)

        # Split into value and gate
        value, gate = y_proj.chunk(2, dim=-1)
        # GLU: value * sigmoid(gate) - preserves amplitude info
        y = value * torch.sigmoid(gate)

        # Output projection
        y = self.output_proj(y)

        # ===== Residual Connection =====
        y = y + residual

        # ===== Feed-Forward Network with Pre-Norm =====
        residual = y
        y = self.ffn_norm(y)
        y = self.ffn(y)
        y = self.ffn_dropout(y)
        y = y + residual

        return y


class S4TemporalNowcaster(nn.Module):
    """
    S4-based Temporal Nowcaster for H2S forecasting.

    Architecture:
        Input Projection -> S4D Stack -> Causal Pooling -> Dual Heads
                                                       -> Regression (H2S value)
                                                       -> Classification (Low/Med/High)

    Features:
    - Long-range memory via S4 state space formulation
    - Strict causal convolution (no future data leakage)
    - Captures 24-hour stagnation buildup patterns
    """

    def __init__(
        self,
        n_features: int,
        d_model: int = 128,
        n_layers: int = 4,
        d_state: int = 64,
        dropout: float = 0.2,
        num_classes: int = 3,
        lr_s4: float = 0.0001,  # Reduced from 0.0005 to prevent A matrix instability
        backend: str = "fft",
        use_improved_block: bool = True,
        num_stations: int = 1,
        seq_len: int = 96,
        dt_min: float = 0.0001,  # NOTE: Minimum timescale ~30min due to fixed A=0.5 (S4D limitation)
        dt_max: float = 0.1,     # NEW: Maximum dt for slow timescales
        auxiliary_target_names: Optional[List[str]] = None,  # NEW: Auxiliary regression targets
        fast_fraction: float = 0.25,  # Fraction of channels in fast lane (0.0=all slow, 1.0=all fast)
        dt_init: str = "random",
        fast_dt_min: Optional[float] = None,
        fast_dt_max: Optional[float] = None,
        slow_dt_min: Optional[float] = None,
        slow_dt_max: Optional[float] = None,
        warmup_epochs: int = 5,  # NEW: State warmup - freeze A matrix for first N epochs
        drop_path_rate: float = 0.0,  # Stochastic Depth: max drop rate (linearly scaled per layer)
        fast_anchor_hours: float = 1.0,   # physics_anchored dt_init: fast lane target timescale (hours)
        slow_anchor_hours: float = 24.0,  # physics_anchored dt_init: slow lane target timescale (hours)
        spread: float = 0.5,              # physics_anchored dt_init: log-space Gaussian std
        forecast_horizon: int = 0,        # SSM-native forecast steps: pad input by h, read last position
        use_fp64_ssm_params: bool = True, # Scope A: store log_A_real & log_dt (and Adam m,v) in fp64
    ):
        """
        Args:
            n_features: Number of input features
            d_model: Model dimension (hidden size)
            n_layers: Number of S4D layers
            d_state: State size for SSM
            dropout: Dropout rate
            num_classes: Number of classification classes
            lr_s4: Learning rate for SSM parameters (A, C, dt)
            backend: SSM computation backend ('fft', 'scan', or 'auto')
            use_improved_block: Use improved S4DBlock with Pre-Norm and GLU (fixes amplitude destruction)
                              Default: True (improved block is now the default)
            num_stations: Number of stations for station embedding (default: 1 for single-site)
            seq_len: Sequence length used during training (needed for proper inference)
            dt_min: Minimum timescale step (lower = faster dynamics, 15-min level when 0.0001)
            dt_max: Maximum timescale step (higher = slower dynamics)
            auxiliary_target_names: List of auxiliary regression target names (e.g., ['CH4'])
            warmup_epochs: Number of epochs to freeze A matrix during warmup (default: 5)
            drop_path_rate: Max stochastic depth drop rate. Layer i gets rate i/(n_layers-1)*drop_path_rate.
                           Layer 0 is never dropped; the deepest layer gets the full rate (default: 0.0)
        """
        super().__init__()

        self.n_features = n_features
        self.d_model = d_model
        self.n_layers = n_layers
        self.d_state = d_state
        self.num_classes = num_classes
        self.lr_s4 = lr_s4
        self.backend = backend
        self.use_improved_block = use_improved_block
        self.num_stations = num_stations
        self.seq_len = seq_len
        self.dt_min = dt_min
        self.dt_max = dt_max
        self.fast_fraction = fast_fraction
        self.dt_init = dt_init
        self.drop_path_rate = drop_path_rate
        self.fast_anchor_hours = fast_anchor_hours
        self.slow_anchor_hours = slow_anchor_hours
        self.spread = spread
        self.forecast_horizon = forecast_horizon

        # Auxiliary targets (e.g., CH4 for co-emission learning)
        self.auxiliary_target_names = auxiliary_target_names or []

        # DEBUG: Verify dt_min/dt_max received from train_s4.py
        # print(f"[DEBUG S4TemporalNowcaster] Received dt_min={dt_min}, dt_max={dt_max}")

        # Station embedding: learns site-specific characteristics (baseline H2S, sensor bias, etc.)
        # For single-site training (num_stations=1), this is a single learnable vector
        # For multi-site training, each station gets its own embedding
        self.station_embedding = nn.Embedding(num_stations, d_model)
        # Initialize station embeddings to small values for stability
        nn.init.normal_(self.station_embedding.weight, mean=0.0, std=0.1)

        # Input projection: features -> d_model (excluding station_id if present)
        # If station_id is the last feature, we subtract 1 from n_features
        self.input_proj = nn.Linear(n_features, d_model)

        # S4D layers - select between standard and improved
        BlockClass = S4DBlockImproved if use_improved_block else S4DBlock
        self.s4_layers = nn.ModuleList([
            BlockClass(
                d_model=d_model,
                d_state=d_state,
                dropout=dropout,
                lr=lr_s4,
                backend=backend,
                dt_min=dt_min,
                dt_max=dt_max,
                fast_fraction=fast_fraction,
                dt_init=dt_init,
                fast_dt_min=fast_dt_min,
                fast_dt_max=fast_dt_max,
                slow_dt_min=slow_dt_min,
                slow_dt_max=slow_dt_max,
                fast_anchor_hours=fast_anchor_hours,
                slow_anchor_hours=slow_anchor_hours,
                spread=spread,
                use_fp64_ssm_params=use_fp64_ssm_params,
            )
            for _ in range(n_layers)
        ])
        self.use_fp64_ssm_params = use_fp64_ssm_params

        # Stochastic Depth: linearly scale drop rate from 0 → drop_path_rate across layers.
        # Layer 0 (first block) is never dropped; layer n_layers-1 gets the full rate.
        # This matches the fast-to-slow lane ordering: advection (layer 0) is always kept,
        # deeper stagnation/barometric blocks are dropped more aggressively.
        if n_layers > 1:
            _drop_rates = [i * drop_path_rate / (n_layers - 1) for i in range(n_layers)]
        else:
            _drop_rates = [0.0]
        self.drop_paths = nn.ModuleList([DropPath(r) for r in _drop_rates])
        if drop_path_rate > 0.0:
            print(f"[S4 TRAIN] Stochastic Depth: rates = {[f'{r:.3f}' for r in _drop_rates]}")

        # Dual output heads
        self.regression_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1)
        )

        self.classification_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes)
        )

        # EVT Probabilistic Head (activated only when evt_mode=True)
        # Outputs 4 parameters: [mu, sigma, sigma_gpd, xi]
        self.evt_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 4)  # mu, sigma, sigma_gpd, xi
        )

        # Auxiliary regression heads (one per target gas)
        # Simple linear heads: force the backbone to produce good representations
        # rather than letting complex aux heads compensate with their own capacity
        self.auxiliary_heads = nn.ModuleDict()
        for name in self.auxiliary_target_names:
            self.auxiliary_heads[name] = nn.Linear(d_model, 1)

        # Initialize weights
        self._initialize_weights()
        self._is_fitted = True

        # State warmup: freeze A matrix during early training
        # This allows C matrix to learn how to read from HiPPO timescales first
        self.warmup_epochs = warmup_epochs
        self._current_epoch = 0
        if warmup_epochs > 0:
            self._freeze_a_matrix()
            print(f"[S4 WARMUP] A matrix FROZEN for {warmup_epochs} epochs - C will learn to read HiPPO timescales first")

        # EVT mode configuration (activated via set_evt_mode when labeling.mode == 'evt')
        self.evt_mode = False
        self.evt_config: Dict[str, Any] = {}

    def set_evt_mode(self, enabled: bool, evt_config: Dict[str, Any]) -> None:
        """
        Activate EVT (Extreme Value Theory) mode for probabilistic density estimation.

        When enabled, the model uses:
        - A 4-parameter EVT head (mu, sigma, sigma_gpd, xi) instead of classification/regression heads
        - The spliced_evt_criterion loss instead of joint_criterion
        - predict_ews_alarms for EWS integration instead of predict_proba

        Args:
            enabled: Whether to activate EVT mode
            evt_config: EVT configuration dict with keys:
                - threshold_u: Physical boundary for Lognormal/GPD splice (default: 2.0)
                - phi: Exceedance probability at threshold_u (default: 0.10)
                - target_scale_divisor: Scales targets to prevent NLL gradient explosion (default: 10.0)
                - ews_high_threshold: Physical threshold for High alarm integration (default: 7.0)
        """
        self.evt_mode = enabled
        self.evt_config = evt_config
        if enabled:
            print(f"[EVT] Mode enabled. Config: {evt_config}")
            # Apply physics-informed priors to evt_head if provided
            priors = evt_config.get('priors', {})
            if priors and hasattr(self, 'evt_head'):
                import math
                mu_val = priors.get('mu', 0.0)
                sigma_val = priors.get('sigma', 0.5212)
                sigma_g_val = priors.get('sigma_gpd', 0.9850)
                xi_val = priors.get('xi', 0.3000)
                # Inverse softplus for sigma/sigma_gpd (softplus_inv)
                sigma_raw = math.log(math.exp(sigma_val + 1e-6) - 1 + 1e-6)
                sigma_g_raw = math.log(math.exp(sigma_g_val + 1e-6) - 1 + 1e-6)
                # Logit for xi (before sigmoid*0.99+0.01 scaling)
                xi_val_clip = min(max(xi_val, 1e-4), 1 - 1e-4)
                xi_raw = math.log(xi_val_clip / (1 - xi_val_clip) + 1e-6)
                with torch.no_grad():
                    self.evt_head[-1].bias[0] = mu_val
                    self.evt_head[-1].bias[1] = sigma_raw
                    self.evt_head[-1].bias[2] = sigma_g_raw
                    self.evt_head[-1].bias[3] = xi_raw
                print(f"[EVT] Priors applied: mu={mu_val:.4f}, sigma={sigma_val:.4f}, "
                      f"sigma_gpd={sigma_g_val:.4f}, xi={xi_val:.4f}")
                # Zero-init WEIGHTS only: forces initial output to match the Priors exactly
                # The BIAS must be left alone — it holds the physics-informed priors!
                with torch.no_grad():
                    nn.init.zeros_(self.evt_head[-1].weight)
                    # LEAVE THE BIAS ALONE — it holds our Mu/Sigma/Xi priors!

    def set_backend(self, backend: str) -> None:
        """
        Switch SSM computation backend for all S4D layers.

        Args:
            backend: 'fft', 'scan', or 'auto'
        """
        self.backend = backend
        for layer in self.s4_layers:
            layer.set_backend(backend)

    def _freeze_a_matrix(self) -> None:
        """Freeze A matrix parameters to prevent optimizer from killing memory during warmup."""
        for layer in self.s4_layers:
            if hasattr(layer.kernel, 'log_A_real'):
                layer.kernel.log_A_real.requires_grad = False
            if hasattr(layer.kernel, 'A_imag'):
                layer.kernel.A_imag.requires_grad = False

    def _unfreeze_a_matrix(self) -> None:
        """Unfreeze A matrix parameters after warmup to allow fine-tuning."""
        for layer in self.s4_layers:
            if hasattr(layer.kernel, 'log_A_real'):
                layer.kernel.log_A_real.requires_grad = True
            if hasattr(layer.kernel, 'A_imag'):
                layer.kernel.A_imag.requires_grad = True

    def advance_warmup(self, epoch: int) -> None:
        """
        Advance warmup state. Unfreezes A matrix after warmup_epochs.

        Call this at the start of each epoch.

        Args:
            epoch: Current epoch number (0-indexed)
        """
        self._current_epoch = epoch
        if epoch == self.warmup_epochs:
            self._unfreeze_a_matrix()
            print(f"[S4 WARMUP] Epoch {epoch}: A matrix UNFROZEN - fine-tuning physics now enabled")
        elif epoch < self.warmup_epochs:
            remaining = self.warmup_epochs - epoch
            print(f"[S4 WARMUP] Epoch {epoch}: A matrix frozen ({remaining} epochs remaining)")

    def _initialize_weights(self):
        """Initialize weights with layer-specific strategies.

        Strategy per layer type:
        1. GLU Projection (glu_proj): Xavier (gain=1.0) for sigmoid-like gating
        2. Output Projection (output_proj): Xavier (gain=1.0) for residual path
        3. Final Output Heads: Zero-init to prevent loss spikes at epoch 0
        4. Standard Linear: Kaiming (ReLU) for feed-forward layers
        """
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                # 1. GLU Projection (inside S4 blocks)
                # GLU = value * sigmoid(gate) - needs Xavier (gain=1.0), not ReLU gain
                if 'glu_proj' in name:
                    nn.init.xavier_normal_(module.weight, gain=1.0)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)

                # 2. Output Projection (inside S4 blocks)
                # In residual path: y = output_proj(glu_output) + residual
                # Standard Xavier: Pre-LayerNorm + GLU gate provide amplitude stability.
                # gain=0.1 (previously) caused ~200x gradient starvation of SSM parameters.
                elif 'output_proj' in name:
                    nn.init.xavier_normal_(module.weight, gain=1.0)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)

                # 3. Final Output Heads (Regression/Classification)
                # Zero-init prevents massive initial loss spikes
                elif (hasattr(self, 'regression_head') and module is self.regression_head[-1]) or \
                     (hasattr(self, 'classification_head') and module is self.classification_head[-1]):
                    nn.init.constant_(module.weight, 0.0)
                    if module.bias is not None:
                        nn.init.constant_(module.bias, 0.0)

                # 3.5. Auxiliary Output Heads (zero-init)
                elif hasattr(self, 'auxiliary_heads') and \
                     any(module is head for head in self.auxiliary_heads.values()):
                    nn.init.constant_(module.weight, 0.0)
                    if module.bias is not None:
                        nn.init.constant_(module.bias, 0.0)

                # 4. EVT Probabilistic Head (zero-init fallback)
                # NOTE: Prior initialization is done in set_evt_mode() when evt_config is available.
                # This zero-init is a fallback if no priors are provided.
                elif hasattr(self, 'evt_head') and module is self.evt_head[-1]:
                    nn.init.constant_(module.weight, 0.0)
                    if module.bias is not None:
                        nn.init.constant_(module.bias, 0.0)
                        with torch.no_grad():
                            module.bias[3] = -3.0  # xi → sigmoid(-3) ≈ 0.05

                # 5. Standard Linear Layers (Feed-forward, etc.)
                else:
                    nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)

            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(
        self,
        x: torch.Tensor,
        station_ids: Optional[torch.Tensor] = None,
        return_features: bool = False
    ) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor] | Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (Batch, Seq_Len, Input_Dim)
            station_ids: Optional station ID indices of shape (Batch,) for multi-site training
            return_features: If True, return intermediate features for analysis

        Returns:
            If return_features=False:
                (h2s_value, class_logits) - Regression and classification outputs
            If return_features=True:
                (h2s_value, class_logits, features) - With intermediate features

        Station Embedding:
            For multi-site training, station_ids allows the model to learn site-specific
            characteristics (baseline H2S levels, sensor biases, etc.). For single-site
            training, station_ids can be None and the first embedding (index 0) is used.
        """
        # Input projection
        x = self.input_proj(x)  # (B, L, H)

        # Add station embedding if station_ids provided
        if station_ids is not None:
            # Get station embeddings: (B, d_model)
            station_emb = self.station_embedding(station_ids)  # (B, H)
            # Expand to match sequence length: (B, L, H)
            station_emb = station_emb.unsqueeze(1).expand(-1, x.size(1), -1)
            # Add to projected features (broadcasted across time)
            x = x + station_emb

        # SSM-native forecasting: zero-pad by forecast_horizon steps.
        # Zero in standardised feature space = training mean = "no new information" prior.
        # Each S4 layer auto-uses L = x.size(1), so the longer input propagates
        # through the stack without any per-layer changes.
        if self.forecast_horizon > 0:
            x = F.pad(x.transpose(1, 2), (0, self.forecast_horizon)).transpose(1, 2)
            # x is now (B, L + h, H)

        # Store initial features for analysis
        features = [x.clone()]

        # S4D layers with stochastic depth.
        # Each block returns x + delta (residual already added internally).
        # We extract the delta and apply DropPath before adding back — this
        # randomly bypasses the entire block (delta → 0) per sample per step,
        # creating an implicit ensemble of shallower sub-networks.
        for layer, drop_path in zip(self.s4_layers, self.drop_paths):
            x = x + drop_path(layer(x) - x)
            features.append(x.clone())

        # Trajectory extraction: capture intermediate horizon predictions.
        # When forecast_horizon > 0, positions [-h:] contain the model's
        # prediction trajectory from h=1 (next step) to h=forecast_horizon.
        if self.forecast_horizon > 0:
            x_horizon = x[:, -self.forecast_horizon:, :]  # (B, h, H)
            h_b, h_t, h_d = x_horizon.shape
            x_horizon_flat = x_horizon.reshape(-1, h_d)  # (B*h, H)
            if not self.evt_mode:
                traj_class = self.classification_head(x_horizon_flat)  # (B*h, C)
                traj_regr = self.regression_head(x_horizon_flat)       # (B*h, 1)
                self._trajectory_class = traj_class.reshape(h_b, h_t, -1)  # (B, h, C)
                self._trajectory_regr = traj_regr.reshape(h_b, h_t)        # (B, h)
            else:
                self._trajectory_class = None
                self._trajectory_regr = None
        else:
            self._trajectory_class = None
            self._trajectory_regr = None

        # Causal pooling: use final timestep only (strict causality)
        x_final = x[:, -1, :]  # (B, H)

        # Auxiliary outputs (stored as attribute for training loop access)
        self._last_aux_outputs = {}
        for name, head in self.auxiliary_heads.items():
            self._last_aux_outputs[name] = head(x_final)  # (B, 1)

        if self.evt_mode:
            # EVT mode: output 4-parameter distribution [mu, sigma, sigma_gpd, xi]
            raw_params = self.evt_head(x_final)  # (B, 4)

            # Enforce physical bounds to prevent NaN in NLL
            mu = raw_params[:, 0]                                             # unbounded
            sigma = F.softplus(raw_params[:, 1]) + 1e-6                    # > 0
            sigma_gpd = F.softplus(raw_params[:, 2]) + 1e-6                # > 0
            # xi in (0.01, 1.0): shifted sigmoid gives ~0.056 at init with bias=-3.0
            xi = torch.sigmoid(raw_params[:, 3]) * 0.99 + 0.01

            evt_params = torch.stack([mu, sigma, sigma_gpd, xi], dim=-1)

            if return_features:
                return evt_params, torch.stack(features, dim=1)
            return evt_params
        else:
            # Standard mode: regression + classification heads
            h2s_value = self.regression_head(x_final)  # (B, 1)
            class_logits = self.classification_head(x_final)  # (B, num_classes)

            if return_features:
                return h2s_value, class_logits, torch.stack(features, dim=1)
            return h2s_value, class_logits

    def _to_tensor(self, X: np.ndarray | pd.DataFrame | torch.Tensor) -> torch.Tensor:
        """Convert input to torch tensor."""
        if isinstance(X, torch.Tensor):
            return X.float()
        elif isinstance(X, pd.DataFrame):
            return torch.tensor(X.values, dtype=torch.float32)
        else:
            return torch.tensor(X, dtype=torch.float32)

    def _create_sliding_windows(self, X: torch.Tensor) -> torch.Tensor:
        """
        Create sliding windows for inference (VECTORIZED for speed).

        Must match create_sequence_dataset() from train_s4.py exactly:
        window[i] = X[i : i + seq_len], label = y[i + seq_len - 1].

        Args:
            X: Input tensor of shape (n_samples, n_features)

        Returns:
            X_seq: Tensor of shape (n_windows, seq_len, n_features)
            indices: Tensor of end-of-window positions (for label alignment)
        """
        n_samples = X.shape[0]
        seq_len = self.seq_len

        # Need at least seq_len samples to create one window
        if n_samples < seq_len:
            raise ValueError(f"Need at least {seq_len} samples for inference, got {n_samples}")

        # Window end positions: seq_len-1, seq_len, ..., n_samples-1-horizon
        # Trims the last forecast_horizon windows so idx+horizon is always in bounds.
        # Matches create_sequence_dataset: indices = np.arange(seq_len - 1, n_samples - horizon, stride)
        indices = torch.arange(seq_len - 1, n_samples - self.forecast_horizon, device=X.device)
        n_windows = len(indices)

        # VECTORIZED: use unfold on the raw data (NO zero-padding)
        # X.T shape: (n_features, n_samples)
        X_t = X.T.unsqueeze(0)  # (1, n_features, n_samples)
        X_seq_t = X_t.unfold(2, seq_len, 1).squeeze(0)  # (n_features, n_windows_all, seq_len)
        X_seq = X_seq_t.permute(1, 2, 0)  # (n_windows_all, seq_len, n_features)
        X_seq = X_seq[:n_windows]  # trim to valid range (forecast_horizon)

        return X_seq, indices

    def predict(
        self,
        X: np.ndarray | pd.DataFrame | torch.Tensor
    ) -> np.ndarray:
        """
        Predict class labels (XGBoost-compatible interface).

        Args:
            X: Input features of shape (n_samples, n_features) or (n_samples, seq_len, n_features)

        Returns:
            Class predictions as numpy array.
            For 2-D input (n_samples, n_features): shape is (n_samples - seq_len + 1,),
            matching create_sequence_dataset() so that caller can align y_true[seq_len-1:].
            For pre-windowed 3-D input: shape is (n_samples,).
        """
        self.eval()
        with torch.no_grad():
            x_tensor = self._to_tensor(X)
            device = next(self.parameters()).device
            x_tensor = x_tensor.to(device)

            needs_windowing = x_tensor.ndim == 2
            if needs_windowing:
                n_raw = x_tensor.shape[0]
                print(f"[S4 PRED] Creating sliding windows from {n_raw} samples (seq_len={self.seq_len})...", flush=True)
                x_tensor, _ = self._create_sliding_windows(x_tensor)
                print(f"[S4 PRED] Windows created: {x_tensor.shape[0]} windows (valid positions {self.seq_len-1}..{n_raw-1})", flush=True)

            print(f"[S4 PRED] Running forward pass on {x_tensor.shape[0]} sequences...", flush=True)
            batch_size = 128
            all_preds = []
            for i in range(0, x_tensor.shape[0], batch_size):
                model_out = self.forward(x_tensor[i:i+batch_size])
                if self.evt_mode:
                    # EVT mode: return class 2 (High) for all — EVT doesn't do standard classification
                    # Caller should use predict_ews_alarms() for EVT risk estimation
                    preds = torch.full((model_out.shape[0],), 2, dtype=torch.long, device=model_out.device)
                else:
                    _, class_logits = model_out
                    preds = torch.argmax(class_logits, dim=1)
                all_preds.append(preds)
            predictions = torch.cat(all_preds)
            print(f"[S4 PRED] {len(predictions)} predictions returned (caller must use y_true[seq_len-1:] for alignment)", flush=True)

        return predictions.cpu().numpy()

    def predict_proba(
        self,
        X: np.ndarray | pd.DataFrame | torch.Tensor
    ) -> np.ndarray:
        """
        Predict class probabilities (XGBoost-compatible interface).

        Args:
            X: Input features of shape (n_samples, n_features) or (n_samples, seq_len, n_features)

        Returns:
            Class probabilities as numpy array.
            For 2-D input: shape is (n_samples - seq_len + 1, n_classes).
            Caller must align y_true[seq_len-1:] for evaluation.
        """
        self.eval()
        with torch.no_grad():
            x_tensor = self._to_tensor(X)
            device = next(self.parameters()).device
            x_tensor = x_tensor.to(device)

            if x_tensor.ndim == 2:
                x_tensor, _ = self._create_sliding_windows(x_tensor)

            batch_size = 128
            all_proba = []
            for i in range(0, x_tensor.shape[0], batch_size):
                model_out = self.forward(x_tensor[i:i+batch_size])
                if self.evt_mode:
                    # EVT mode: caller should use predict_ews_alarms() instead
                    # Return uniform [0.33, 0.33, 0.33] as placeholder
                    proba = torch.ones(model_out.shape[0], 3, device=model_out.device) / 3.0
                else:
                    _, class_logits = model_out
                    proba = F.softmax(class_logits, dim=1)
                all_proba.append(proba)
            probabilities = torch.cat(all_proba)

        return probabilities.cpu().numpy()

    def predict_trajectory(
        self,
        X: np.ndarray | pd.DataFrame | torch.Tensor,
    ) -> dict[str, np.ndarray]:
        """
        Predict class probabilities and regression values at each forecast
        horizon step.  Returns a dict with keys:
          'proba':  (n_windows, forecast_horizon, n_classes)  — softmax probabilities
          'value':  (n_windows, forecast_horizon)             — regression predictions
          'class':  (n_windows, forecast_horizon)             — argmax class labels
        Requires forecast_horizon > 0.
        """
        if self.forecast_horizon <= 0:
            raise ValueError(
                "forecast_horizon must be > 0 for trajectory prediction"
            )

        self.eval()
        with torch.no_grad():
            x_tensor = self._to_tensor(X)
            device = next(self.parameters()).device
            x_tensor = x_tensor.to(device)

            if x_tensor.ndim == 2:
                x_tensor, _ = self._create_sliding_windows(x_tensor)

            batch_size = 128
            all_traj_class = []
            all_traj_regr = []
            for i in range(0, x_tensor.shape[0], batch_size):
                self.forward(x_tensor[i:i + batch_size])
                if self._trajectory_class is not None:
                    all_traj_class.append(self._trajectory_class)
                if self._trajectory_regr is not None:
                    all_traj_regr.append(self._trajectory_regr)

            result: dict[str, np.ndarray] = {}
            if all_traj_class:
                traj_logits = torch.cat(all_traj_class)      # (N, h, C)
                traj_proba = F.softmax(traj_logits, dim=-1)   # (N, h, C)
                result["proba"] = traj_proba.cpu().numpy()
                result["class"] = traj_logits.argmax(dim=-1).cpu().numpy()
            if all_traj_regr:
                result["value"] = torch.cat(all_traj_regr).cpu().numpy()

        return result

    def predict_ews_alarms(
        self,
        X: np.ndarray | pd.DataFrame | torch.Tensor,
        threshold_u: float = 2.0,
        high_threshold: float = 7.0,
        phi: float = 0.15,
        scale_divisor: float = 10.0,
    ) -> np.ndarray:
        """
        Integrates continuous EVT distributions to output probabilities for
        [Low, Medium, High] risk bins via EWS (Early Warning System) integration.

        Args:
            X: Input features of shape (n_samples, n_features) or (n_samples, seq_len, n_features)
            threshold_u: Physical boundary between Low and Medium (matches training u)
            high_threshold: Physical boundary between Medium and High
            phi: Exceedance probability used during training
            scale_divisor: Divides raw H2S targets before NLL — thresholds are
                           scaled by the same divisor to stay in the learned space

        Returns:
            Array of shape (n_windows, 3) representing P(Low), P(Medium), P(High)
        """
        self.eval()
        with torch.no_grad():
            x_tensor = self._to_tensor(X)
            device = next(self.parameters()).device
            x_tensor = x_tensor.to(device)

            if x_tensor.ndim == 2:
                x_tensor, _ = self._create_sliding_windows(x_tensor)

            # CRITICAL FIX: Map thresholds into the model's internal scale
            u_s = threshold_u / scale_divisor
            h_s = high_threshold / scale_divisor

            batch_size = 128
            all_probs = []

            for i in range(0, x_tensor.shape[0], batch_size):
                # forward() returns [mu, sigma, sigma_gpd, xi] in EVT mode
                evt_params = self.forward(x_tensor[i:i+batch_size])
                mu, sigma, sigma_gpd, xi = evt_params.unbind(dim=-1)

                # 1. P(Low): Lognormal CDF up to u_s
                # Erf-based CDF is more numerically stable than a standard normal object
                z_low = (math.log(u_s) - mu) / (sigma * math.sqrt(2.0))
                p_low = 0.5 * (1.0 + torch.erf(z_low))
                p_low = p_low * (1.0 - phi)  # Adjust for spliced mass balance

                # 2. P(High): GPD Survival Function beyond h_s
                # Formula: P(X > x) = phi * (1 + xi * (x - u) / sigma_gpd) ^ (-1/xi)
                exceedance = h_s - u_s
                p_high = phi * torch.pow(1.0 + xi * exceedance / sigma_gpd, -1.0 / xi)

                # 3. P(Medium): Remainder
                p_med = 1.0 - p_low - p_high

                # Stack and clamp to [0, 1] to handle floating point epsilon errors
                probs = torch.stack([p_low, p_med, p_high], dim=-1).clamp(0.0, 1.0)
                all_probs.append(probs)

            return torch.cat(all_probs).cpu().numpy()

    def predict_regression(
        self,
        X: np.ndarray | pd.DataFrame | torch.Tensor,
        reg_stats: Optional[Dict] = None,
    ) -> np.ndarray:
        """
        Predict continuous H2S values.

        Args:
            X: Input features of shape (n_samples, n_features) or (n_samples, seq_len, n_features)
            reg_stats: Optional dict with 'mean' and 'std' keys for inverse-transforming
                       standardized predictions back to the original H2S scale.

        Returns:
            H2S value predictions as numpy array.
            For 2-D input: shape is (n_samples - seq_len + 1,).
            Caller must align y_true[seq_len-1:] for evaluation.
        """
        self.eval()
        with torch.no_grad():
            x_tensor = self._to_tensor(X)
            device = next(self.parameters()).device
            x_tensor = x_tensor.to(device)

            if x_tensor.ndim == 2:
                x_tensor, _ = self._create_sliding_windows(x_tensor)

            batch_size = 128
            all_reg = []
            for i in range(0, x_tensor.shape[0], batch_size):
                reg_out, _ = self.forward(x_tensor[i:i+batch_size])
                all_reg.append(reg_out.squeeze(1))
            h2s_value = torch.cat(all_reg)

        result = h2s_value.cpu().numpy()
        if reg_stats is not None:
            result = result * reg_stats["std"] + reg_stats["mean"]
        return result

    def predict_auxiliary(
        self,
        X: np.ndarray | pd.DataFrame | torch.Tensor,
        aux_stats: Optional[Dict] = None,
    ) -> Dict[str, np.ndarray]:
        """
        Predict auxiliary regression targets (e.g. CH4) and optionally inverse-transform
        from standardized space back to real-scale concentrations.

        Args:
            X: Input features of shape (n_samples, n_features) or
               (n_samples, seq_len, n_features)
            aux_stats: Optional dict {target_name: {'mean': float, 'std': float}}
                       produced during training-set standardization.  When supplied
                       the returned values are in the original unit (e.g. ppm).

        Returns:
            Dict mapping each auxiliary target name to a numpy array of shape
            (n_samples,).  Returns an empty dict when no auxiliary heads exist.
        """
        self.eval()
        if not self.auxiliary_target_names:
            return {}

        batch_results: Dict[str, List[np.ndarray]] = {n: [] for n in self.auxiliary_target_names}
        indices = None

        with torch.no_grad():
            x_tensor = self._to_tensor(X)
            device = next(self.parameters()).device
            x_tensor = x_tensor.to(device)

            needs_windowing = x_tensor.ndim == 2
            if needs_windowing:
                x_tensor, indices = self._create_sliding_windows(x_tensor)

            # Process in batches to match predict_regression memory behaviour
            infer_batch = 256
            for i in range(0, x_tensor.shape[0], infer_batch):
                self.forward(x_tensor[i : i + infer_batch])  # sets _last_aux_outputs
                for name in self.auxiliary_target_names:
                    batch_results[name].append(
                        self._last_aux_outputs[name].squeeze(-1).cpu().numpy()
                    )

        results: Dict[str, np.ndarray] = {}
        n_samples = X.shape[0] if not isinstance(X, torch.Tensor) else X.shape[0]

        for name in self.auxiliary_target_names:
            preds = np.concatenate(batch_results[name])

            if needs_windowing and indices is not None:
                full = np.full(n_samples, preds[0], dtype=preds.dtype)
                full[:self.seq_len - 1] = preds[0]
                full[indices.cpu().numpy()] = preds
                preds = full

            if aux_stats and name in aux_stats:
                preds = preds * aux_stats[name]["std"] + aux_stats[name]["mean"]

            results[name] = preds

        return results

    def save(
        self,
        path: Path | str,
        optimizer_state: Optional[Dict] = None,
        training_info: Optional[Dict] = None
    ):
        """
        Save model, optimizer state, and configuration.

        Args:
            path: Directory to save model files
            optimizer_state: Optional optimizer state dict
            training_info: Optional training metadata
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Save model state
        model_path = path / "s4_temporal_model.pt"
        torch.save(self.state_dict(), model_path)

        # Save config
        config_path = path / "s4_temporal_config.json"
        save_dict = {
            "n_features": self.n_features,
            "d_model": self.d_model,
            "n_layers": self.n_layers,
            "d_state": self.d_state,
            "num_classes": self.num_classes,
            "dropout": getattr(self, "dropout", None),
            "use_improved_block": self.use_improved_block,
            "num_stations": self.num_stations,
            "seq_len": self.seq_len,  # Save seq_len for proper inference
            "auxiliary_target_names": self.auxiliary_target_names,  # Save aux targets
            "fast_fraction": self.fast_fraction,
            "dt_min": self.dt_min,
            "dt_max": self.dt_max,
            "dt_init": self.dt_init,
            "drop_path_rate": self.drop_path_rate,
            "fast_anchor_hours": self.fast_anchor_hours,
            "slow_anchor_hours": self.slow_anchor_hours,
            "spread": self.spread,
            "evt_mode": self.evt_mode,
            "evt_config": self.evt_config,
            "forecast_horizon": self.forecast_horizon,
            "use_fp64_ssm_params": self.use_fp64_ssm_params,
        }
        config_path.write_text(json.dumps(save_dict, indent=2))

        # Save optimizer state if provided
        if optimizer_state is not None:
            optim_path = path / "s4_temporal_optimizer.pt"
            torch.save(optimizer_state, optim_path)

        # Save training info if provided
        if training_info is not None:
            info_path = path / "s4_temporal_training_info.json"
            info_path.write_text(json.dumps(training_info, indent=2, default=str))

        print(f"[SAVE] S4 Temporal model saved to {model_path}")
        print(f"[SAVE] Config saved to {config_path}")

    @classmethod
    def load(
        cls,
        path: Path | str,
        map_location: Optional[str] = None
    ) -> "S4TemporalNowcaster":
        """
        Load model from saved files.

        Args:
            path: Directory containing saved model files
            map_location: Device to map tensors to (default: 'cpu')

        Returns:
            Loaded S4TemporalNowcaster instance
        """
        path = Path(path)
        if map_location is None:
            map_location = "cpu"

        # Load config
        config_path = path / "s4_temporal_config.json"
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        config_data = json.loads(config_path.read_text())

        model_path = path / "s4_temporal_model.pt"
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        state_dict = torch.load(model_path, map_location=map_location)

        fast_fraction = config_data.get("fast_fraction")
        if fast_fraction is None:
            fast_key = "s4_layers.0.kernel._fast_lane_indices"
            slow_key = "s4_layers.0.kernel._slow_lane_indices"
            fast_idx = state_dict.get(fast_key)
            slow_idx = state_dict.get(slow_key)
            if fast_idx is not None and slow_idx is not None:
                n_fast = int(fast_idx.numel())
                n_slow = int(slow_idx.numel())
                if (n_fast + n_slow) > 0:
                    fast_fraction = n_fast / (n_fast + n_slow)

        num_stations = config_data.get("num_stations")
        if num_stations is None:
            station_weight = state_dict.get("station_embedding.weight")
            if station_weight is not None:
                num_stations = int(station_weight.shape[0])

        # Create model
        model = cls(
            n_features=config_data["n_features"],
            d_model=config_data.get("d_model", 128),
            n_layers=config_data.get("n_layers", 4),
            d_state=config_data.get("d_state", 64),
            dropout=config_data.get("dropout") or 0.2,
            num_classes=config_data.get("num_classes", 3),
            use_improved_block=config_data.get("use_improved_block", True),
            num_stations=num_stations or 1,
            seq_len=config_data.get("seq_len", 96),  # Load seq_len from config
            dt_min=config_data.get("dt_min", 0.0001),
            dt_max=config_data.get("dt_max", 0.1),
            auxiliary_target_names=config_data.get("auxiliary_target_names", []),  # Load aux targets
            fast_fraction=fast_fraction if fast_fraction is not None else 0.25,
            dt_init=config_data.get("dt_init", "random"),
            drop_path_rate=config_data.get("drop_path_rate", 0.0),
            fast_anchor_hours=config_data.get("fast_anchor_hours", 1.0),
            slow_anchor_hours=config_data.get("slow_anchor_hours", 24.0),
            spread=config_data.get("spread", 0.5),
            forecast_horizon=config_data.get("forecast_horizon", 0),
            use_fp64_ssm_params=config_data.get("use_fp64_ssm_params", True),
        )
        model.load_state_dict(state_dict)

        # Restore EVT mode if the saved model was in EVT mode
        if config_data.get("evt_mode", False):
            model.set_evt_mode(True, config_data.get("evt_config", {}))

        print(f"[LOAD] S4 Temporal model loaded from {model_path}")
        return model

    def export_onnx(
        self,
        path: Path | str,
        seq_len: int = 96,
        opset_version: int = 17,
        dynamic_batch: bool = True
    ) -> Path:
        """
        Export model to ONNX format for deployment.

        Args:
            path: Output ONNX file path
            seq_len: Sequence length for export
            opset_version: ONNX opset version
            dynamic_batch: Enable dynamic batch size

        Returns:
            Path to exported ONNX file
        """
        from .s4_export import export_s4_to_onnx
        return export_s4_to_onnx(
            self, path,
            seq_len=seq_len,
            n_features=self.n_features,
            opset_version=opset_version,
            convert_to_real=True,
            dynamic_batch=dynamic_batch
        )

    def export_tensorrt(
        self,
        onnx_path: Path | str,
        trt_path: Path | str,
        fp16: bool = True,
        max_batch_size: int = 32
    ) -> Optional[Path]:
        """
        Compile model to TensorRT engine for fast inference.

        Args:
            onnx_path: Path to ONNX model (will create if doesn't exist)
            trt_path: Output TensorRT engine path
            fp16: Use FP16 precision
            max_batch_size: Maximum batch size

        Returns:
            Path to TensorRT engine, or None if compilation fails
        """
        from .s4_export import compile_tensorrt_engine
        onnx_path = Path(onnx_path)

        # Export ONNX if needed
        if not onnx_path.exists():
            self.export_onnx(onnx_path)

        return compile_tensorrt_engine(
            onnx_path, trt_path,
            fp16=fp16,
            max_batch_size=max_batch_size
        )

    def to_real_representation(self) -> "S4TemporalNowcaster":
        """
        Convert model to use real-valued S4D kernels.

        This is required for TensorRT export since TensorRT doesn't support
        complex tensors.

        Returns:
            Self (model is modified in-place)
        """
        from .s4_orin import convert_s4d_to_real
        convert_s4d_to_real(self)
        return self


def focal_cross_entropy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    alpha: Optional[torch.Tensor] = None,
    gamma: float = 2.0,
) -> torch.Tensor:
    """
    Focal loss for classification: FL = -alpha_t * (1 - p_t)^gamma * log(p_t).

    Down-weights easy (high-confidence) examples so the model focuses on hard
    samples near class boundaries. With gamma=2.0, a sample at 90% confidence
    has its loss reduced ~100x compared to standard CE.

    Args:
        logits: Raw class logits (B, C)
        targets: Integer class labels (B,)
        alpha: Per-class weights (C,), optional
        gamma: Focusing parameter (0 = standard CE)

    Returns:
        Scalar focal loss
    """
    ce = F.cross_entropy(logits, targets, weight=alpha, reduction='none')  # (B,)
    log_probs = F.log_softmax(logits, dim=1)
    p_t = torch.exp(log_probs).gather(1, targets.unsqueeze(1)).squeeze(1)  # (B,)
    # Clamp (1-p_t) to min=1e-6 to prevent numerical underflow with high gamma values.
    # With gamma=4.69 and p_t→1, (1-p_t)^gamma → 0 in float32, producing NaN gradients.
    focal_weight = (1.0 - p_t).clamp(min=1e-6) ** gamma
    return (focal_weight * ce).mean()


def joint_criterion(
    outputs: Tuple[torch.Tensor, torch.Tensor],
    targets: Tuple[torch.Tensor, torch.Tensor],
    class_weights: Optional[torch.Tensor] = None,
    regression_weight: float = 1.0,
    classification_weight: float = 5.0,
    focal_gamma: float = 0.0,
    # NEW: Phase 1 regression parameters
    regression_loss_type: str = "mse",        # "mse" | "huber" | "huber_asym"
    huber_delta: float = 2.0,                 # Huber threshold
    asymmetric_alpha: float = 0.5,            # Under-prediction penalty weight
    return_components: bool = False,         # NEW: Return individual loss components
) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Joint loss function for regression and classification.

    Args:
        outputs: (val_pred, class_pred) tuple of predictions
        targets: (val_true, class_true) tuple of targets
        class_weights: Per-class weights for cross-entropy
        regression_weight: Weight for regression loss
        classification_weight: Weight for CE loss
        focal_gamma: Focal loss gamma (0 = standard CE, >0 = focal loss)
        regression_loss_type: Type of regression loss ("mse", "huber", "huber_asym")
        huber_delta: Huber loss threshold (ppb)
        asymmetric_alpha: Under-prediction penalty weight for huber_asym
        return_components: If True, return (reg_loss, ce_loss, total_loss)

    Returns:
        If return_components=False: Combined loss value
        If return_components=True: Tuple of (reg_loss, ce_loss, total_loss)
    """
    val_pred, class_pred = outputs
    val_true, class_true = targets

    # Skip regression computation entirely if weight is 0
    if regression_weight == 0:
        reg_loss = torch.tensor(0.0, device=class_pred.device)
        if focal_gamma > 0:
            ce_loss = focal_cross_entropy(class_pred, class_true, alpha=class_weights, gamma=focal_gamma)
        else:
            ce_loss = F.cross_entropy(class_pred, class_true, weight=class_weights)

        total_loss = classification_weight * ce_loss

        if return_components:
            return reg_loss, ce_loss, total_loss
        return total_loss

    # Fix shape: ensure both are (B,) for regression
    val_pred_sq = val_pred.squeeze()
    val_true_sq = val_true.squeeze()

    # Regression loss: MSE, Huber, or Huber with asymmetric penalty
    if regression_loss_type == "mse":
        reg_loss = F.mse_loss(val_pred_sq, val_true_sq)
    elif regression_loss_type == "huber":
        reg_loss = F.huber_loss(val_pred_sq, val_true_sq, delta=huber_delta)
    elif regression_loss_type == "huber_asym":
        huber = F.huber_loss(val_pred_sq, val_true_sq, delta=huber_delta)
        # Asymmetric penalty: extra cost when under-predicting (val_pred < val_true)
        residual = val_true_sq - val_pred_sq
        under_pred_mask = (residual > 0).float()
        asym_penalty = (under_pred_mask * residual.abs()).mean()
        reg_loss = huber + asymmetric_alpha * asym_penalty
    else:
        # Default to MSE for unknown types
        reg_loss = F.mse_loss(val_pred_sq, val_true_sq)

    if focal_gamma > 0:
        ce_loss = focal_cross_entropy(class_pred, class_true, alpha=class_weights, gamma=focal_gamma)
    else:
        ce_loss = F.cross_entropy(class_pred, class_true, weight=class_weights)

    total_loss = regression_weight * reg_loss + classification_weight * ce_loss

    if return_components:
        return reg_loss, ce_loss, total_loss
    return total_loss


def spliced_evt_criterion(
    preds: torch.Tensor,
    targets: torch.Tensor,
    threshold_u: float,
    phi: float = 0.10,
) -> torch.Tensor:
    """
    Computes the Negative Log-Likelihood for the Spliced Lognormal-GPD distribution.

    The distribution is:
    - Lognormal for values <= threshold_u (bulk/normal regime)
    - Generalized Pareto Distribution (GPD) for values > threshold_u (tail/extreme regime)

    The two components are spliced at threshold_u with mass allocated according to phi.

    Args:
        preds: Tensor of shape (B, 4) containing [mu, sigma, sigma_gpd, xi]
        targets: True continuous H2S emission values (B,) — must already be scaled
                 by target_scale_divisor before passing to this function
        threshold_u: Physical value dividing normal data from extreme spikes (scaled)
        phi: Empirical probability of exceeding threshold_u (training set fraction)

    Returns:
        Scalar NLL loss
    """
    mu, sigma, sigma_gpd, xi = preds.unbind(dim=-1)

    # Ensure targets are strictly positive for lognormal math
    safe_targets = targets.clamp(min=1e-6)
    log_targets = torch.log(safe_targets)

    # BULK LOSS (Lognormal) — for values <= threshold_u
    # Negative log-likelihood of lognormal CDF component
    bulk_loss = -math.log(1.0 - phi) + torch.log(sigma) + ((log_targets - mu) ** 2) / (2.0 * sigma ** 2)

    # TAIL LOSS (GPD) — for values > threshold_u
    # Clamp exceedance to 1e-6 to prevent NaN in inactive tensor elements
    # during torch.where routing
    exceedance = (targets - threshold_u).clamp(min=1e-6)
    tail_loss = (
        -math.log(phi)
        + torch.log(sigma_gpd)
        + (1.0 + 1.0 / xi) * torch.log(1.0 + xi * exceedance / sigma_gpd)
    )

    # PIECEWISE ROUTING: select bulk or tail loss based on threshold crossing
    is_tail = targets > threshold_u
    loss = torch.where(is_tail, tail_loss, bulk_loss)

    return loss.mean()


def auxiliary_loss(
    aux_outputs: Dict[str, torch.Tensor],
    aux_targets: Dict[str, torch.Tensor],
    aux_configs: List[Dict]
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Compute auxiliary regression loss.

    Args:
        aux_outputs: Dict of {name: predictions} tensors, shape (B, 1)
        aux_targets: Dict of {name: targets} tensors, shape (B,)
        aux_configs: List of {name, weight, loss_type} configs

    Returns:
        (total_loss, {name: loss_component})
    """
    total = torch.tensor(0.0, device=next(iter(aux_outputs.values())).device)
    components = {}

    for cfg in aux_configs:
        name = cfg['name']
        if name not in aux_outputs:
            continue

        weight = cfg.get('weight', 0.1)
        loss_type = cfg.get('loss_type', 'mse')

        pred = aux_outputs[name].squeeze(-1)  # (B,)
        true = aux_targets[name]  # (B,)

        # Handle NaN values (missing measurements)
        valid = ~torch.isnan(true)
        if not valid.any():
            continue  # Skip if all NaN

        pred_valid = pred[valid]
        true_valid = true[valid]

        if loss_type == 'huber':
            delta = cfg.get('huber_delta', 1.0)
            loss_val = F.smooth_l1_loss(pred_valid, true_valid, reduction='mean', beta=delta)
        else:  # mse
            loss_val = F.mse_loss(pred_valid, true_valid, reduction='mean')

        components[name] = loss_val.item()
        total = total + weight * loss_val

    return total, components


def configure_s4_optimizer(
    model: S4TemporalNowcaster,
    learning_rate: float = 0.001,
    weight_decay: float = 0.01,
    lr_s4: Optional[float] = None,
    lr_A_real: Optional[float] = None,
    lr_D: Optional[float] = None,
) -> torch.optim.Optimizer:
    """
    Configure optimizer with differential learning rates for S4 parameters.

    SSM parameters are split into FIVE groups:
    - log_A_real: Gets lr_A_real (20x boost to compensate for tiny gradients)
    - kernel.C: Gets learning_rate (base rate) — C controls which dynamics appear
      in the output, so it must learn at the same speed as the heads/FFN
    - D (skip connection): Gets lr_D (reduced to prevent crutch behavior)
    - Other SSM (log_dt, A_imag): Gets lr_s4
    - Non-SSM: Gets learning_rate

    This fixes two issues:
    1. The "Redundancy Trap" where log_A_real has 18x smaller gradients than
       log_dt, causing the optimizer to only learn via dt adjustments.
    2. The "C starvation" bug where C was grouped with log_dt at lr_s4 (5e-5),
       making it 20x slower than needed and causing physics interpretability plots
       to look identical before and after training (C_real_std changed only 0.1%
       across 16 epochs).
    3. The "Skip-Connection Bypass" where D (skip connection) dominates gradient
       flow and the model ignores the SSM memory. Reducing lr_D forces the model
       to learn through the A matrix instead.

    Args:
        model: S4TemporalNowcaster model
        learning_rate: Base learning rate for non-SSM parameters
        weight_decay: Weight decay for non-SSM parameters
        lr_s4: Learning rate for SSM parameters (default: use model's lr_s4)
        lr_A_real: Separate LR for A matrix (default: lr_s4 * 20)
        lr_D: Separate LR for D skip connection (default: learning_rate * 0.1)

    Returns:
        Configured optimizer with 5 parameter groups
    """
    if lr_s4 is None:
        lr_s4 = model.lr_s4 if hasattr(model, 'lr_s4') else lr_s4 or 0.001

    # Default lr_A_real to 20x lr_s4 if not specified
    if lr_A_real is None:
        lr_A_real = lr_s4 * 20.0

    # Default lr_D to 10x slower than base LR to prevent skip-connection dominance
    if lr_D is None:
        lr_D = learning_rate * 0.1

    # Separate parameters into FIVE groups with SAFE matching
    # CRITICAL: Use 'kernel.' prefix to avoid false positives
    # - 'C' in name is dangerous (could match CNN, FC layers)
    # - Solution: Check for 'kernel.' prefix to only match S4 kernel params
    params_A_real = []
    params_C = []        # C gets base learning rate (controls output representation)
    params_D = []        # D gets reduced LR to prevent crutch behavior
    params_ssm_other = []
    params_other = []

    for name, param in model.named_parameters():
        if 'kernel.log_A_real' in name:
            params_A_real.append(param)
        elif 'kernel.C' in name:
            # C controls how learned temporal dynamics appear in the output.
            # It must learn at the base rate alongside heads/FFN, not at the
            # slow lr_s4 rate which caused physics plots to look unchanged.
            params_C.append(param)
        elif name.endswith('.D') or '.D)' in name:
            # D is the skip connection - reduce LR to prevent optimizer from
            # using it as a crutch instead of learning SSM memory dynamics.
            params_D.append(param)
        elif 'kernel.' in name:
            # log_dt, A_imag go at lr_s4
            params_ssm_other.append(param)
        elif 'evt_head' in name:
            # evt_head gets its own differential LR groups below — exclude from catch-all
            pass
        else:
            params_other.append(param)

    # Create parameter groups with different learning rates.
    # `name` is consumed by SSMTelemetryTracker for per-group LR/precision diagnostics.
    param_groups = [
        {'params': params_A_real,    'lr': lr_A_real,     'weight_decay': 0.0,          'name': 'log_A_real'},  # 20x boost
        {'params': params_C,         'lr': learning_rate, 'weight_decay': 0.0,          'name': 'kernel_C'},    # Base rate
        {'params': params_D,         'lr': lr_D,          'weight_decay': 0.0,          'name': 'D_skip'},      # 10x slower (anti-crutch)
        {'params': params_ssm_other, 'lr': lr_s4,         'weight_decay': 0.0,          'name': 'ssm_other'},   # Slow SSM LR
        {'params': params_other,     'lr': learning_rate, 'weight_decay': weight_decay, 'name': 'main'},
    ]

    # EVT differential learning rate groups — prevents "backbone gravity" from pulling
    # the physical location parameter (mu) away from its physics-informed prior.
    # Bias gets 5x turbo boost to escape the backbone's vacuum.
    # Weights get 0.5x slower pace so the bias (physics) leads, backbone follows.
    evt_bias_params = [p for n, p in model.named_parameters() if 'evt_head' in n and 'bias' in n]
    evt_weight_params = [p for n, p in model.named_parameters() if 'evt_head' in n and 'weight' in n]
    if evt_bias_params:
        param_groups.append({
            'params': evt_bias_params,
            'lr': learning_rate * 5.0,
            'name': 'evt_head_bias',
            'weight_decay': 0.0,
        })
    if evt_weight_params:
        param_groups.append({
            'params': evt_weight_params,
            'lr': learning_rate * 0.5,
            'name': 'evt_head_weight',
            'weight_decay': 0.01,
        })

    return torch.optim.AdamW(param_groups)
