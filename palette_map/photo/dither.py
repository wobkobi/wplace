# palette_map/photo/dither.py
from __future__ import annotations

"""
High-quality photo dithering (Floyd-Steinberg + blue noise + cached multi-colour texture).

- Error diffusion in Lab-L using Floyd-Steinberg.
- Blue-noise threshold map for ordered micro-mix assignment (minimal tiling).
- Palette-aware scoring in Lab/LCh with local-contrast and shadow terms.
- Adaptive 1..N colours per pixel, greedy by dE2000 gain, with a small plan cache.

Hot spots to watch:
  - RGB->Lab conversion on large images
  - Local blur and gradient precompute
  - Inner pixel loop: candidate pick and adaptive micro-mix

No runtime flags required. Sensible defaults are chosen for quality and speed.

Optional fast path: overlapped block multiprocessing with shared memory and per-row progress.
"""

from typing import List, Tuple, Dict, Optional, NamedTuple, TypeAlias
from concurrent.futures import (
    ThreadPoolExecutor,
    ProcessPoolExecutor,
    wait,
    FIRST_COMPLETED,
)
from multiprocessing import shared_memory
import math
import os
import time
import numpy as np

from palette_map.colour_convert import delta_e2000_vec, rgb_to_lab_threaded
from palette_map.core_types import U8Image, U8Mask, Lab, Lch, PaletteItem
from palette_map.utils import (
    box_blur_2d,
    format_eta,
    format_seconds_compact,
    topk_indices_by_lab_distance as prefilter_topk,
    print_progress_line,
    debug_log,
)

# Tunables (quality-first; verbose comments)

# Size of the box blur in pixels for local lightness. Higher = smoother, slower.
BLUR_RADIUS = 2

# How quickly local gradients reduce texture strength. Higher = more smoothing in busy areas.
GRADIENT_SCALE = 5.0

# How many palette candidates to test per pixel after prefiltering.
PHOTO_TOPK = 24

# Limits how much lightness error gets diffused. Stops streaks and ringing.
LIGHTNESS_ERR_CLAMP = 4.0

# Treat a pixel as near neutral if its chroma is below this.
NEUTRAL_EST_CHROMA = 7.0

# Only estimate neutral a/b bias from pixels with lightness inside this range.
NEUTRAL_EST_LMIN = 20.0
NEUTRAL_EST_LMAX = 90.0

# Strength of the neutral a/b bias correction. 0 disables it.
AB_BIAS_GAIN = 0.6

# Base weight for keeping local contrast similar to the source.
W_LOCAL_BASE = 1.0

# Allow a little contrast loss before adding a penalty.
CONTRAST_MARGIN = 2.0

# How much hue mismatch matters. Scales up with source chroma.
W_HUE = 0.55

# How much chroma mismatch matters.
W_CHROMA = 0.015

# Below this lightness, treat pixels as shadow and relax matching slightly.
SHADOW_L = 40.0

# Extra hue penalty in shadows.
W_SHADOW_HUE = 0.10

# Extra penalty for picking a target that is even more chromatic in shadows.
W_SHADOW_CHROMA_UP = 0.05

# If the source chroma is below this, prefer neutral targets a bit more.
CHROMA_SRC_MIN = 6.0

# Targets with chroma at or below this are considered neutral.
NEUTRAL_C_MAX = 5.0

# If chroma is below this, treat as very close to neutral.
NEAR_NEUTRAL_C = 2.5

# Lightness cut for near black and near white.
NEAR_BLACK_L = 10.0
NEAR_WHITE_L = 92.0

# Small nudges that protect metals and near-neutrals from bad picks.
W_NEUTRAL_GUARD = 0.05

# Lightness guard to preserve local contrast.
W_LIGHTNESS_LOCK = 0.05

# Quantization steps for the tiny decision cache. Bigger steps = more hits, less precision.
Q_L = 0.5
Q_AB = 1.0
Q_L_LOCAL = 0.5
Q_W_LOCAL = 0.3  # coarser for more cache hits

# Adaptive N-colour micro mix for fine texture and closer averages.
TEXTURE_MAX_COLOURS = 4
TEXTURE_TOPK = 6
TEXTURE_MIN_DE_GAIN = 0.12
TEXTURE_LOCK_NEUTRALS = True
TEXTURE_GRAD_GATE = 0.35

# Skip mixing if best single dE is already very good.
MIX_TRIGGER_DE = 1.0

# Error diffusion kernel: Floyd-Steinberg (weights sum to 16).
KERNEL_FS: Tuple[Tuple[int, int, float], ...] = (
    (1, 0, 7 / 16),
    (-1, 1, 3 / 16),
    (0, 1, 5 / 16),
    (1, 1, 1 / 16),
)

# Blue-noise threshold tile (power-of-two sizes tile well; bigger = finer steps).
THRESH_TILE = 512
BLUE_SEED = 1337

# Cache guard: bound memory for the tiny plan cache. Clear when exceeded.
CACHE_MAX_ENTRIES = 300_000

# Cache key type: quantised (L, a, b, L_local, w_local, near_extreme_flag)
CacheKey: TypeAlias = Tuple[int, int, int, int, int, int]


# Threaded local ops


def _split_rows_with_halo(
    height: int, parts: int, halo: int
) -> List[Tuple[int, int, int, int]]:
    """Split rows into chunks with a halo so box blur stays exact across seams."""
    parts = max(1, int(parts))
    step = (height + parts - 1) // parts
    out: List[Tuple[int, int, int, int]] = []
    start = 0
    while start < height:
        end = min(start + step, height)
        start_pad = max(0, start - halo)
        end_pad = min(height, end + halo)
        out.append((start, end, start_pad, end_pad))
        start = end
    return out


def box_blur_2d_threaded(arr: np.ndarray, radius: int, workers: int) -> np.ndarray:
    """Threaded box blur that keeps exact output via overlapped halos."""
    if radius <= 0 or workers <= 1 or arr.shape[0] < 256:
        return box_blur_2d(arr, radius)
    height, width = arr.shape
    chunks = _split_rows_with_halo(height, workers, radius)
    out = np.empty((height, width), dtype=np.float32)

    def run_one(chunk):
        start, end, start_pad, end_pad = chunk
        sub = box_blur_2d(arr[start_pad:end_pad], radius)
        core = sub[(start - start_pad) : (start - start_pad) + (end - start)]
        return (start, end, core)

    with ThreadPoolExecutor(max_workers=workers) as ex:
        for start, end, core in ex.map(run_one, chunks):
            out[start:end] = core
    return out


def grad_mag_threaded(L: np.ndarray, workers: int) -> np.ndarray:
    """Threaded L1 gradient magnitude with a small blur."""
    gx = np.zeros_like(L, dtype=np.float32)
    gy = np.zeros_like(L, dtype=np.float32)
    gx[:, 1:] = np.abs(L[:, 1:] - L[:, :-1])
    gy[1:, :] = np.abs(L[1:, :] - L[:-1, :])
    grad = gx + gy
    return box_blur_2d_threaded(grad, 1, workers)


# Blue-noise threshold map


def _make_blue_noise_tile(n: int, seed: int) -> np.ndarray:
    """
    Fast blue-ish noise: high-pass filtered white noise, then rank-map to [0,1).
    Deterministic per seed. Good ordered assignment without visible tiling.
    """
    rng = np.random.default_rng(seed)
    a = rng.random((n, n), dtype=np.float32)
    # Push energy to high frequencies
    a = a - box_blur_2d(a, 1)
    a = a - box_blur_2d(a, 1)
    # Rank to uniform thresholds
    flat = a.reshape(-1)
    order = np.argsort(flat, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float32)
    ranks[order] = (np.arange(order.size, dtype=np.float32) + np.float32(0.5)) / float(
        order.size
    )
    return ranks.reshape(n, n)


_BLUENOISE = _make_blue_noise_tile(THRESH_TILE, BLUE_SEED)


def _threshold_at(x: int, y: int) -> float:
    """Blue-noise threshold lookup; modulo provides a seamless repeat."""
    return float(_BLUENOISE[y % THRESH_TILE, x % THRESH_TILE])


# Neutral bias estimate


def estimate_ab_bias(lab_img: Lab, alpha: U8Mask) -> Tuple[float, float]:
    """
    Estimate a and b neutral bias from visible near-neutral pixels to reduce
    colour cast in highlights and midtones.
    """
    L = lab_img[..., 0]
    a = lab_img[..., 1]
    b = lab_img[..., 2]
    chroma = np.hypot(a, b)
    mask = (
        (alpha != 0)
        & (chroma <= NEUTRAL_EST_CHROMA)
        & (L >= NEUTRAL_EST_LMIN)
        & (L <= NEUTRAL_EST_LMAX)
    )
    if not mask.any():
        return (0.0, 0.0)
    return (AB_BIAS_GAIN * float(a[mask].mean()), AB_BIAS_GAIN * float(b[mask].mean()))


# Cost model


def photo_cost_components(
    src_lab_vec: np.ndarray,
    src_lch_vec: np.ndarray,
    local_L: float,
    local_weight_eff: float,
    target_lch_rows: Lch,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute palette-aware cost terms for a single source colour against target LCh rows.
    Returns (cost_vector, target_chroma_vector).
    """
    src_L, src_C, src_h = (
        float(src_lch_vec[0]),
        float(src_lch_vec[1]),
        float(src_lch_vec[2]),
    )
    tgt_L, tgt_C, tgt_h = (
        target_lch_rows[:, 0],
        target_lch_rows[:, 1],
        target_lch_rows[:, 2],
    )
    cost = np.zeros_like(tgt_L, dtype=np.float32)

    # Local contrast preservation
    cost += (
        W_LOCAL_BASE * local_weight_eff * np.abs((tgt_L - local_L) - (src_L - local_L))
    )

    d_src = abs(src_L - local_L)
    d_tgt = np.abs(tgt_L - local_L)
    cost += np.where(
        d_tgt + CONTRAST_MARGIN < d_src, (d_src - d_tgt - CONTRAST_MARGIN), 0.0
    )

    hue_weight = min(1.0, src_C / 40.0)
    hue_dist = np.minimum(np.abs(tgt_h - src_h), 360.0 - np.abs(tgt_h - src_h)) / 180.0
    cost += W_HUE * hue_weight * hue_dist
    cost += W_CHROMA * np.abs(tgt_C - src_C)

    if src_L < SHADOW_L:
        shadow = (SHADOW_L - src_L) / SHADOW_L
        chroma_up = np.maximum(0.0, tgt_C - src_C)
        cost += shadow * (W_SHADOW_HUE * hue_dist + W_SHADOW_CHROMA_UP * chroma_up)

    if src_C >= CHROMA_SRC_MIN:
        cost += np.where(
            tgt_C <= NEUTRAL_C_MAX, 0.10 + 0.02 * np.maximum(0.0, src_C - tgt_C), 0.0
        )

    return cost, tgt_C


# Candidate pick


def _nearest_idx_photo(
    src_lab_vec: np.ndarray,
    src_lch_vec: np.ndarray,
    local_L: float,
    local_weight_eff: float,
    pal_lab_rows: Lab,
    pal_lch_rows: Lch,
    near_extreme_neutral: bool,
    k: int,
    *,
    pal_is_neutral: np.ndarray,
    prof: Optional[Dict[str, float]] = None,
) -> Tuple[int, np.ndarray, np.ndarray, np.ndarray]:
    """
    Pick nearest palette index using dE2000 + palette-aware components.
    Uses a fast top-k prefilter; the hot path is delta_e2000_vec over the top-k.
    """
    t0 = time.perf_counter()
    candidate_indices = prefilter_topk(src_lab_vec, pal_lab_rows, k)
    candidate_lab = pal_lab_rows[candidate_indices]
    candidate_lch = pal_lch_rows[candidate_indices]

    de_vec = delta_e2000_vec(src_lab_vec, candidate_lab)
    comp_vec, _ = photo_cost_components(
        src_lab_vec, src_lch_vec, local_L, local_weight_eff, candidate_lch
    )
    cost_vec = de_vec + comp_vec

    if src_lch_vec[1] <= NEUTRAL_EST_CHROMA:
        cost_vec += W_NEUTRAL_GUARD * np.where(
            candidate_lch[:, 1] <= NEUTRAL_C_MAX, 1.0, 0.0
        )
        cost_vec += W_LIGHTNESS_LOCK * np.abs(
            (candidate_lch[:, 0] - local_L) - (src_lch_vec[0] - local_L)
        )
    if near_extreme_neutral:
        cost_vec = np.where(
            candidate_lch[:, 1] > NEUTRAL_C_MAX, cost_vec + 1e6, cost_vec
        )

    j_rel = int(np.argmin(cost_vec))

    if prof is not None:
        prof["t_nearest"] += time.perf_counter() - t0
        prof["calls_nearest"] += 1
        prof["evals_nearest"] += int(candidate_lab.shape[0])

    return int(candidate_indices[j_rel]), candidate_indices, cost_vec, de_vec


# Texture mix helpers


class Plan(NamedTuple):
    """Execution plan per pixel: single colour or a stochastic mix."""

    mode: int  # 0 single, 1 mix
    js: Tuple[int, ...]  # palette indices
    cum: Tuple[float, ...]  # cumulative weights in [0,1]; len==len(js), last==1.0


PLAN_SINGLE = 0
PLAN_MIX = 1


class _EtaStable:
    """
    ETA that avoids oscillation and early zero.
    Blends fast/slow EWMAs and enforces a naive floor from overall pace.
    """

    def __init__(
        self,
        total_rows: int,
        win_alpha_fast: float = 0.25,
        win_alpha_slow: float = 0.08,
    ):
        self.total = max(1, int(total_rows))
        self.rows = 0
        self.elapsed = 0.0
        self.ema_fast: Optional[float] = None
        self.ema_slow: Optional[float] = None
        self.display_eta: Optional[float] = None
        self.af = win_alpha_fast
        self.aslow = win_alpha_slow

    def _update_one(self, row_dt: float) -> None:
        self.rows += 1
        self.elapsed += row_dt
        self.ema_fast = (
            row_dt
            if self.ema_fast is None
            else (1 - self.af) * self.ema_fast + self.af * row_dt
        )
        self.ema_slow = (
            row_dt
            if self.ema_slow is None
            else (1 - self.aslow) * self.ema_slow + self.aslow * row_dt
        )

    def update(self, row_dt: float) -> float:
        self._update_one(row_dt)
        return self._eta_current()

    def update_rows(self, rows_delta: int, dt: float) -> float:
        if rows_delta <= 0:
            return self._eta_current()
        per_row = dt / float(rows_delta)
        for _ in range(rows_delta):
            self._update_one(per_row)
        return self._eta_current()

    def _eta_current(self) -> float:
        rows_left = max(0, self.total - self.rows)
        row_est = max(self.ema_fast or 0.0, self.ema_slow or 0.0)
        eta_raw = rows_left * row_est
        naive = self.elapsed * (self.total / max(1, self.rows) - 1.0)
        eta = max(eta_raw, max(0.0, 0.9 * naive))
        self.display_eta = (
            eta if self.display_eta is None else 0.3 * eta + 0.7 * self.display_eta
        )
        return self.display_eta


def _make_single_plan(idx: int) -> Plan:
    return Plan(PLAN_SINGLE, (int(idx),), (1.0,))


def _make_mix_plan(indices: np.ndarray, weights: np.ndarray) -> Plan:
    cum = np.cumsum(weights.astype(np.float32))
    cum[-1] = 1.0
    return Plan(
        PLAN_MIX,
        tuple(int(j) for j in indices.tolist()),
        tuple(float(x) for x in cum.tolist()),
    )


def _mix_weights_simplex(
    target_rows_lab: np.ndarray, src_lab_vec: np.ndarray
) -> np.ndarray:
    """
    Solve min ||P w - s||_2  s.t. sum(w)=1, w>=0 for tiny N (<=4).
    Robust fallback to uniform on ill-conditioned systems.
    """
    P = target_rows_lab.astype(np.float32, copy=False)
    s = src_lab_vec.astype(np.float32, copy=False)
    N = P.shape[0]
    Q = P @ P.T
    c = P @ s
    K = np.zeros((N + 1, N + 1), dtype=np.float32)
    K[:N, :N] = Q
    K[:N, N] = 0.5
    K[N, :N] = 1.0
    rhs = np.zeros((N + 1,), dtype=np.float32)
    rhs[:N] = c
    rhs[N] = 1.0
    try:
        sol = np.linalg.solve(K, rhs)
        w = sol[:N]
    except np.linalg.LinAlgError:
        w = np.full((N,), 1.0 / float(N), dtype=np.float32)
    w = np.maximum(w, 0.0)
    ssum = float(w.sum())
    if ssum <= 0.0:
        w[:] = 0.0
        w[0] = 1.0
    else:
        w /= ssum
    return w.astype(np.float32, copy=False)


def _de_lab(src_lab_vec: np.ndarray, pal_lab_row: np.ndarray) -> float:
    return float(delta_e2000_vec(src_lab_vec, pal_lab_row[None, :])[0])


def _choose_texture_mix(
    src_lab_vec: np.ndarray,
    src_chroma: float,
    best_idx: int,
    candidate_indices: np.ndarray,
    cost_vec: np.ndarray,
    pal_lab_rows: Lab,
    pal_lch_rows: Lch,
    *,
    near_extreme: bool,
    max_colours: int,
    topk: int,
    min_gain: float,
    lock_neutrals: bool,
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """
    Greedy N-colour mix builder. Adds partners while dE gain >= threshold.
    Respects neutral-lock and avoids extreme near-neutrals.
    """
    if near_extreme:
        return None

    must_neutral = lock_neutrals and (src_chroma <= NEUTRAL_EST_CHROMA)

    # Partner pool from top-k by cost
    k_eff = min(int(topk), cost_vec.size)
    part = np.argpartition(cost_vec, k_eff - 1)[:k_eff]
    rel = part[np.argsort(cost_vec[part])]
    partners = [
        int(candidate_indices[r]) for r in rel if int(candidate_indices[r]) != best_idx
    ]
    if not partners:
        return None

    if must_neutral:
        if pal_lch_rows[best_idx, 1] > NEUTRAL_C_MAX:
            return None
        partners = [j for j in partners if pal_lch_rows[j, 1] <= NEUTRAL_C_MAX]
        if not partners:
            return None

    chosen: List[int] = [best_idx]
    best_de_single = _de_lab(src_lab_vec, pal_lab_rows[best_idx])
    prev_de = best_de_single

    while len(chosen) < max(2, int(max_colours)):
        best_gain = 0.0
        best_new_idx: Optional[int] = None
        best_de_candidate = prev_de
        best_weights: Optional[np.ndarray] = None

        for j in partners:
            if j in chosen:
                continue
            P_try = pal_lab_rows[chosen + [j]]
            w_try = _mix_weights_simplex(P_try, src_lab_vec)
            mix_lab = (w_try[:, None] * P_try).sum(axis=0)
            de_mix = _de_lab(src_lab_vec, mix_lab)
            gain = (
                (best_de_single - de_mix) if (len(chosen) == 1) else (prev_de - de_mix)
            )
            if gain > best_gain:
                best_gain = gain
                best_new_idx = j
                best_de_candidate = de_mix
                best_weights = w_try

        if best_new_idx is not None and best_gain >= float(min_gain):
            chosen.append(best_new_idx)
            prev_de = best_de_candidate
            if len(chosen) >= max_colours:
                break
        else:
            break

        if (
            best_weights is not None
            and float((best_weights > 1e-3).sum()) <= len(chosen) - 1
        ):
            break

    if len(chosen) <= 1:
        return None

    w_final = _mix_weights_simplex(pal_lab_rows[chosen], src_lab_vec)
    mix_lab = (w_final[:, None] * pal_lab_rows[chosen]).sum(axis=0)
    final_gain = best_de_single - _de_lab(src_lab_vec, mix_lab)
    if final_gain < float(min_gain):
        return None

    return np.asarray(chosen, dtype=np.int32), w_final.astype(np.float32)


# Shared memory helpers (MP path)


def _to_shm(arr: np.ndarray, name: Optional[str] = None):
    shm = shared_memory.SharedMemory(create=True, size=arr.nbytes, name=name)
    view = np.ndarray(arr.shape, dtype=arr.dtype, buffer=shm.buf)
    view[...] = arr
    return shm, arr.shape, str(arr.dtype)


def _from_shm(name: str, shape: Tuple[int, ...], dtype_str: str):
    shm = shared_memory.SharedMemory(name=name)
    arr = np.ndarray(shape, dtype=np.dtype(dtype_str), buffer=shm.buf)
    return shm, arr


# Overlapped block multiprocessing


def _split_blocks(
    height: int, block_rows: int, overlap_rows: int
) -> List[Tuple[int, int, int, int, int, int]]:
    """
    Return [(start,end,start_pad,end_pad,keep_start,keep_end)] in absolute row coords.
    Each block processes [start_pad:end_pad] but contributes the full core [start:end].
    """
    out: List[Tuple[int, int, int, int, int, int]] = []
    start = 0
    while start < height:
        end = min(height, start + block_rows)
        start_pad = max(0, start - overlap_rows)
        end_pad = min(height, end + overlap_rows)
        keep_start = start  # contiguous keeps to avoid seams
        keep_end = end
        out.append((start, end, start_pad, end_pad, keep_start, keep_end))
        start = end
    return out


def _process_block(args) -> Tuple[int, int, np.ndarray]:
    """
    Process one overlapped block. Returns (keep_start, keep_end, kept_RGB).
    Uses shared memory for inputs to avoid pickling copies.
    Updates per-row progress in shared counter.
    """
    (
        block_index,
        start,
        end,
        start_pad,
        end_pad,
        keep_start,
        keep_end,
        shm_rgb_name,
        rgb_shape,
        rgb_dtype,
        shm_alpha_name,
        alpha_shape,
        alpha_dtype,
        shm_plab_name,
        plab_shape,
        plab_dtype,
        shm_plch_name,
        plch_shape,
        plch_dtype,
        shm_prgb_name,
        prgb_shape,
        prgb_dtype,
        progress_shm_name,
        progress_len,
        a_bias,
        b_bias,
        topk_global,
        workers_threaded,
    ) = args

    shm_rgb, img_rgb = _from_shm(shm_rgb_name, tuple(rgb_shape), rgb_dtype)
    shm_alpha, alpha = _from_shm(shm_alpha_name, tuple(alpha_shape), alpha_dtype)
    shm_plab, pal_lab_rows = _from_shm(shm_plab_name, tuple(plab_shape), plab_dtype)
    shm_plch, pal_lch_rows = _from_shm(shm_plch_name, tuple(plch_shape), plch_dtype)
    shm_prgb, pal_rgb_rows = _from_shm(shm_prgb_name, tuple(prgb_shape), prgb_dtype)
    progress_shm_local = shared_memory.SharedMemory(name=progress_shm_name)
    progress_view = np.ndarray(
        (progress_len,), dtype=np.int32, buffer=progress_shm_local.buf
    )

    try:
        full_height, width, _ = img_rgb.shape
        block_height = end_pad - start_pad

        # Convert only the padded slice
        lab_img_blk = rgb_to_lab_threaded(
            img_rgb[start_pad:end_pad], max(1, workers_threaded)
        ).reshape(block_height, width, 3)
        light_src = lab_img_blk[..., 0].astype(np.float32, copy=False)

        # Local lightness and gradient in the padded slice
        if BLUR_RADIUS > 0:
            light_local = box_blur_2d(light_src, BLUR_RADIUS)
        else:
            light_local = light_src.astype(np.float32, copy=False)

        gx = np.zeros_like(light_src, dtype=np.float32)
        gy = np.zeros_like(light_src, dtype=np.float32)
        gx[:, 1:] = np.abs(light_src[:, 1:] - light_src[:, :-1])
        gy[1:, :] = np.abs(light_src[1:, :] - light_src[:-1, :])
        grad_smoothed = box_blur_2d(gx + gy, 1)

        error_L = np.zeros((block_height, width), dtype=np.float32)
        pal_is_neutral = pal_lch_rows[:, 1] <= NEUTRAL_C_MAX
        plan_cache: Dict[CacheKey, Plan] = {}

        out_block = np.zeros((block_height, width, 3), dtype=np.uint8)

        keep_start_local = keep_start - start_pad
        keep_end_local = keep_end - start_pad

        # Scan the padded rows but output only kept core rows later.
        for row_local in range(block_height):
            row_abs = start_pad + row_local
            scan_left_to_right = (row_abs % 2) == 0
            base_kernel = KERNEL_FS
            neighbors = (
                base_kernel
                if scan_left_to_right
                else tuple(
                    (-dx, dy, w) if dx != 0 else (dx, dy, w)
                    for (dx, dy, w) in base_kernel
                )
            )

            xs = range(width) if scan_left_to_right else range(width - 1, -1, -1)
            for x in xs:
                if alpha[row_abs, x] == 0:
                    continue

                local_weight = max(
                    0.2, 1.0 - (grad_smoothed[row_local, x] / GRADIENT_SCALE)
                )

                src_L = float(lab_img_blk[row_local, x, 0]) + float(
                    error_L[row_local, x]
                )
                src_a = float(lab_img_blk[row_local, x, 1]) - a_bias
                src_b = float(lab_img_blk[row_local, x, 2]) - b_bias
                src_C = float(math.hypot(src_a, src_b))
                src_h = (math.degrees(math.atan2(src_b, src_a)) + 360.0) % 360.0

                near_extreme = (src_C <= NEAR_NEUTRAL_C) and (
                    src_L <= NEAR_BLACK_L or src_L >= NEAR_WHITE_L
                )
                topk_eff = topk_global + (4 if src_C >= 24.0 else 0)

                key: CacheKey = (
                    int(round(src_L / Q_L)),
                    int(round(src_a / Q_AB)),
                    int(round(src_b / Q_AB)),
                    int(round(light_local[row_local, x] / Q_L_LOCAL)),
                    int(round(local_weight / Q_W_LOCAL)),
                    1 if near_extreme else 0,
                )

                plan_opt = plan_cache.get(key)
                if plan_opt is None:
                    src_lab_vec = np.array([src_L, src_a, src_b], dtype=np.float32)
                    src_lch_vec = np.array([src_L, src_C, src_h], dtype=np.float32)
                    best_idx, cand_idx, cost_vec, _ = _nearest_idx_photo(
                        src_lab_vec,
                        src_lch_vec,
                        float(light_local[row_local, x]),
                        W_LOCAL_BASE * local_weight,
                        pal_lab_rows,
                        pal_lch_rows,
                        near_extreme,
                        k=topk_eff,
                        pal_is_neutral=pal_is_neutral,
                        prof=None,
                    )
                    # Gate mixing when already close or texture not visible
                    de_best = float(
                        delta_e2000_vec(src_lab_vec, pal_lab_rows[best_idx][None, :])[0]
                    )
                    if (de_best <= MIX_TRIGGER_DE) or (
                        local_weight < TEXTURE_GRAD_GATE
                    ):
                        plan_new = _make_single_plan(best_idx)
                    else:
                        res = _choose_texture_mix(
                            src_lab_vec,
                            src_C,
                            best_idx,
                            cand_idx,
                            cost_vec,
                            pal_lab_rows,
                            pal_lch_rows,
                            near_extreme=near_extreme,
                            max_colours=TEXTURE_MAX_COLOURS,
                            topk=TEXTURE_TOPK,
                            min_gain=TEXTURE_MIN_DE_GAIN,
                            lock_neutrals=TEXTURE_LOCK_NEUTRALS,
                        )
                        plan_new = (
                            _make_mix_plan(*res)
                            if (res is not None)
                            else _make_single_plan(best_idx)
                        )

                    if len(plan_cache) >= CACHE_MAX_ENTRIES:
                        plan_cache.clear()
                    plan_cache[key] = plan_new
                    plan = plan_new
                else:
                    plan = plan_opt

                mode, indices, cum_weights = plan
                if mode == PLAN_SINGLE:
                    chosen_idx = indices[0]
                else:
                    threshold = _threshold_at(x, row_abs)
                    chosen = 0
                    for i, c in enumerate(cum_weights):
                        if threshold < c:
                            chosen = i
                            break
                    chosen_idx = indices[chosen]

                out_block[row_local, x, :] = pal_rgb_rows[chosen_idx]

                tgt_L = float(pal_lch_rows[chosen_idx, 0])
                err_L = max(
                    -LIGHTNESS_ERR_CLAMP, min(LIGHTNESS_ERR_CLAMP, src_L - tgt_L)
                )
                if err_L != 0.0:
                    for dx, dy, w in neighbors:
                        nx = x + dx
                        ny = row_local + dy
                        if (
                            0 <= nx < width
                            and 0 <= ny < block_height
                            and alpha[start_pad + ny, nx] != 0
                        ):
                            error_L[ny, nx] += err_L * w

            # Per-row progress update for kept rows only
            if keep_start_local <= row_local < keep_end_local:
                progress_view[block_index] = int(row_local - keep_start_local + 1)

        kept = out_block[keep_start_local:keep_end_local].copy()
        return keep_start, keep_end, kept
    finally:
        shm_rgb.close()
        shm_alpha.close()
        shm_plab.close()
        shm_plch.close()
        shm_prgb.close()
        progress_shm_local.close()


# Public entry


def _default_workers() -> int:
    n = os.cpu_count() or 4
    reserve = 1 if n <= 6 else 2 if n <= 12 else 3 if n <= 18 else 4
    return max(1, n - reserve)


def dither_photo(
    img_rgb: U8Image,
    alpha: U8Mask,
    palette: List[PaletteItem],
    pal_lab_mat: Lab,
    pal_lch_mat: Lch,
    workers: int = 1,
    **kwargs,
) -> U8Image:
    """
    Serpentine Floyd-Steinberg diffusion in Lab-L with palette-aware scoring.
    Blue-noise micro-mix assignment and cached adaptive N-colour texturing.

    Set profile=True in kwargs to print a timing and counters summary at the end.

    Optional multiprocessing (quality-safe, shared memory, smooth ETA):
      mp_blocks=True           enable overlapped block processing
      mp_block_rows=auto       block height (default scales with image size and CPUs)
      mp_overlap_rows=24       overlap rows on each seam
      mp_procs=None            defaults to available CPUs minus a small reserve
      mp_threads=1             threads per process for local ops (blur, RGB->Lab)
    """
    progress_enabled = bool(kwargs.get("progress", True))
    profile = bool(kwargs.get("profile", False))

    use_mp = bool(kwargs.get("mp_blocks", False))
    mp_processes = kwargs.get("mp_procs")
    if mp_processes is None:
        mp_processes = _default_workers()
    mp_threads = int(kwargs.get("mp_threads", 1))

    height, width, _ = img_rgb.shape
    out = np.zeros_like(img_rgb, dtype=np.uint8)

    # Adaptive top-k by image size (cap work on giant inputs)
    pixels = height * width
    if pixels <= 800_000:
        topk_global = int(kwargs.get("topk", PHOTO_TOPK))
    elif pixels <= 2_500_000:
        topk_global = min(int(kwargs.get("topk", PHOTO_TOPK)), 10)
    else:
        topk_global = min(int(kwargs.get("topk", PHOTO_TOPK)), 8)

    # Precompute palette RGB for fast writes (both paths)
    pal_rgb_mat = np.array([p.rgb for p in palette], dtype=np.uint8)

    # Progress helper that prints every percent from 0..100, starting at 0
    last_pct_printed = {"v": -1}

    def _progress(
        pct_val: float, eta_seconds: float | None, final: bool = False
    ) -> None:
        if not progress_enabled:
            return
        pct_i = max(0, min(100, int(pct_val)))
        if final or pct_i > last_pct_printed["v"]:
            print_progress_line(
                f"[photo] {pct_i:3d}% (ETA {format_eta(eta_seconds)})", final=final
            )
            last_pct_printed["v"] = pct_i

    # Single-process path
    if not use_mp:
        prof: Dict[str, float] = {
            "t_total": 0.0,
            "t_rgb2lab": 0.0,
            "t_locals": 0.0,
            "t_bias": 0.0,
            "t_loop": 0.0,
            "t_nearest": 0.0,
            "calls_nearest": 0.0,
            "evals_nearest": 0.0,
            "cache_hits": 0.0,
            "cache_misses": 0.0,
            "mix_evals": 0.0,
            "mix_accepted": 0.0,
        }
        if profile:
            debug_log(
                "[photo] config: "
                f"Workers={workers}, Blocks=off, Processes=0, ThreadsPerProcess=0, TopK={topk_global}"
            )

        t_total0 = time.perf_counter()

        # RGB -> Lab
        t0 = time.perf_counter()
        lab_img = rgb_to_lab_threaded(img_rgb, workers).reshape(height, width, 3)
        prof["t_rgb2lab"] = time.perf_counter() - t0

        light_src = lab_img[..., 0].astype(np.float32, copy=False)

        # Locals
        t0 = time.perf_counter()
        if BLUR_RADIUS > 0:
            light_local = box_blur_2d_threaded(light_src, BLUR_RADIUS, workers)
        else:
            light_local = light_src.astype(np.float32, copy=False)
        grad_smoothed = grad_mag_threaded(light_src, workers)
        prof["t_locals"] = time.perf_counter() - t0

        # Neutral bias
        t0 = time.perf_counter()
        a_bias, b_bias = estimate_ab_bias(lab_img, alpha)
        prof["t_bias"] = time.perf_counter() - t0

        error_L = np.zeros((height, width), dtype=np.float32)
        pal_is_neutral = pal_lch_mat[:, 1] <= NEUTRAL_C_MAX

        plan_cache: Dict[CacheKey, Plan] = {}

        # ETA
        last_row_t = time.perf_counter()
        rows_done = 0
        eta_est = _EtaStable(height)

        t_loop0 = time.perf_counter()
        for y in range(height):
            scan_left_to_right = (y % 2) == 0
            base_kernel = KERNEL_FS
            neighbors = (
                base_kernel
                if scan_left_to_right
                else tuple(
                    (-dx, dy, w) if dx != 0 else (dx, dy, w)
                    for (dx, dy, w) in base_kernel
                )
            )

            xs = range(width) if scan_left_to_right else range(width - 1, -1, -1)
            for x in xs:
                if alpha[y, x] == 0:
                    continue

                local_weight = max(0.2, 1.0 - (grad_smoothed[y, x] / GRADIENT_SCALE))

                src_L = float(lab_img[y, x, 0]) + float(error_L[y, x])
                src_a = float(lab_img[y, x, 1]) - a_bias
                src_b = float(lab_img[y, x, 2]) - b_bias
                src_C = float(math.hypot(src_a, src_b))
                src_h = (math.degrees(math.atan2(src_b, src_a)) + 360.0) % 360.0

                near_extreme = (src_C <= NEAR_NEUTRAL_C) and (
                    src_L <= NEAR_BLACK_L or src_L >= NEAR_WHITE_L
                )
                topk_eff = topk_global + (4 if src_C >= 24.0 else 0)

                key: CacheKey = (
                    int(round(src_L / Q_L)),
                    int(round(src_a / Q_AB)),
                    int(round(src_b / Q_AB)),
                    int(round(light_local[y, x] / Q_L_LOCAL)),
                    int(round(local_weight / Q_W_LOCAL)),
                    1 if near_extreme else 0,
                )

                plan_opt = plan_cache.get(key)
                if plan_opt is None:
                    prof["cache_misses"] += 1
                    src_lab_vec = np.array([src_L, src_a, src_b], dtype=np.float32)
                    src_lch_vec = np.array([src_L, src_C, src_h], dtype=np.float32)
                    best_idx, cand_idx, cost_vec, _ = _nearest_idx_photo(
                        src_lab_vec,
                        src_lch_vec,
                        float(light_local[y, x]),
                        W_LOCAL_BASE * local_weight,
                        pal_lab_mat,
                        pal_lch_mat,
                        near_extreme,
                        k=topk_eff,
                        pal_is_neutral=pal_is_neutral,
                        prof=prof if profile else None,
                    )
                    de_best = float(
                        delta_e2000_vec(src_lab_vec, pal_lab_mat[best_idx][None, :])[0]
                    )
                    if (de_best <= MIX_TRIGGER_DE) or (
                        local_weight < TEXTURE_GRAD_GATE
                    ):
                        plan_new = _make_single_plan(best_idx)
                    else:
                        res = _choose_texture_mix(
                            src_lab_vec,
                            src_C,
                            best_idx,
                            cand_idx,
                            cost_vec,
                            pal_lab_mat,
                            pal_lch_mat,
                            near_extreme=near_extreme,
                            max_colours=TEXTURE_MAX_COLOURS,
                            topk=TEXTURE_TOPK,
                            min_gain=TEXTURE_MIN_DE_GAIN,
                            lock_neutrals=TEXTURE_LOCK_NEUTRALS,
                        )
                        if res is not None:
                            prof["mix_evals"] += 1
                            mix_indices, mix_weights = res
                            plan_new = _make_mix_plan(mix_indices, mix_weights)
                            prof["mix_accepted"] += 1
                        else:
                            plan_new = _make_single_plan(best_idx)

                    if len(plan_cache) >= CACHE_MAX_ENTRIES:
                        plan_cache.clear()
                    plan_cache[key] = plan_new
                    plan: Plan = plan_new
                else:
                    prof["cache_hits"] += 1
                    plan = plan_opt

                mode, indices, cum_weights = plan
                if mode == PLAN_SINGLE:
                    chosen_idx = indices[0]
                else:
                    threshold = _threshold_at(x, y)
                    chosen = 0
                    for i, c in enumerate(cum_weights):
                        if threshold < c:
                            chosen = i
                            break
                    chosen_idx = indices[chosen]

                out[y, x, :] = pal_rgb_mat[chosen_idx]

                tgt_L = float(pal_lch_mat[chosen_idx, 0])
                err_L = max(
                    -LIGHTNESS_ERR_CLAMP, min(LIGHTNESS_ERR_CLAMP, src_L - tgt_L)
                )
                if err_L != 0.0:
                    for dx, dy, w in neighbors:
                        nx = x + dx
                        ny = y + dy
                        if 0 <= nx < width and 0 <= ny < height and alpha[ny, nx] != 0:
                            error_L[ny, nx] += err_L * w

            # Row timing and per-percent progress
            rows_done += 1
            now = time.perf_counter()
            row_dt = max(1e-6, now - last_row_t)
            last_row_t = now
            eta = eta_est.update(row_dt)
            _progress(100.0 * rows_done / height, eta, final=False)

        prof["t_loop"] = time.perf_counter() - t_loop0
        prof["t_total"] = time.perf_counter() - t_total0
        _progress(100.0, 0.0, final=True)

        if profile:
            total = max(1e-9, prof.get("t_total", 0.0))

            def pct_str(t: float) -> str:
                return f"{100.0 * (t / total):5.1f}%"

            def row(label: str, secs: float) -> str:
                return (
                    f"  {label:<16} {format_seconds_compact(secs):>9}  {pct_str(secs)}"
                )

            debug_log("[profile] photo timing (single-process):")
            print(row("total", prof.get("t_total", 0.0)), flush=True)
            print(
                row(
                    "rgb->lab + bias",
                    prof.get("t_rgb2lab", 0.0) + prof.get("t_bias", 0.0),
                ),
                flush=True,
            )
            print(row("locals", prof.get("t_locals", 0.0)), flush=True)
            print(row("loop", prof.get("t_loop", 0.0)), flush=True)
            print(row("nearest", prof.get("t_nearest", 0.0)), flush=True)
            print(
                f"    {'nearest calls':<22}{int(prof.get('calls_nearest',0))}",
                flush=True,
            )
            print(
                f"    {'nearest evals':<22}{int(prof.get('evals_nearest',0))} (sum of top-k)",
                flush=True,
            )
            print(f"    {'cache hits':<22}{int(prof.get('cache_hits',0))}", flush=True)
            print(
                f"    {'cache misses':<22}{int(prof.get('cache_misses',0))}", flush=True
            )
            print(f"    {'mix evals':<22}{int(prof.get('mix_evals',0))}", flush=True)
            print(
                f"    {'mix accepted':<22}{int(prof.get('mix_accepted',0))}", flush=True
            )
        return out

    # Multiprocessing path (overlapped blocks, shared memory)
    prof_mp: Dict[str, float] = {"t_total": 0.0, "t_bias_rgb2lab": 0.0}
    t_total0 = time.perf_counter()

    # Global neutral bias (consistent across blocks)
    t0 = time.perf_counter()
    lab_full_for_bias = rgb_to_lab_threaded(img_rgb, max(1, mp_threads)).reshape(
        height, width, 3
    )
    a_bias, b_bias = estimate_ab_bias(lab_full_for_bias, alpha)
    prof_mp["t_bias_rgb2lab"] = time.perf_counter() - t0
    del lab_full_for_bias

    # Shared memory for inputs
    shm_rgb, rgb_shape, rgb_dtype = _to_shm(img_rgb)
    shm_alpha, alpha_shape, alpha_dtype = _to_shm(alpha)
    shm_plab, plab_shape, plab_dtype = _to_shm(pal_lab_mat)
    shm_plch, plch_shape, plch_dtype = _to_shm(pal_lch_mat)
    shm_prgb, prgb_shape, prgb_dtype = _to_shm(np.array(pal_rgb_mat, copy=False))
    progress_shm: Optional[shared_memory.SharedMemory] = None

    try:
        # Blocks
        default_block_rows = max(128, height // max(1, (8 * int(mp_processes))))
        mp_block_rows = int(kwargs.get("mp_block_rows", default_block_rows))
        mp_overlap_rows = int(kwargs.get("mp_overlap_rows", 24))
        blocks = _split_blocks(height, mp_block_rows, mp_overlap_rows)

        # Shared progress counter: one int per block (rows completed in kept core)
        progress_shm = shared_memory.SharedMemory(create=True, size=4 * len(blocks))
        progress_view = np.ndarray(
            (len(blocks),), dtype=np.int32, buffer=progress_shm.buf
        )
        progress_view[:] = 0

        total_kept_rows = sum(
            keep_end - keep_start for (_s, _e, _sp, _ep, keep_start, keep_end) in blocks
        )
        eta_est = _EtaStable(total_kept_rows)
        last_rows_done = 0
        last_tick = time.perf_counter()

        if profile:
            debug_log(
                "[photo] config: "
                f"Workers={workers}, Blocks=on, Processes={int(mp_processes)}, "
                f"ThreadsPerProcess={int(mp_threads)}, BlockRows={int(mp_block_rows)}, "
                f"OverlapRows={int(mp_overlap_rows)}, TopK={topk_global}"
            )

        with ProcessPoolExecutor(max_workers=int(mp_processes)) as ex:
            futures = []
            for block_index, (
                start,
                end,
                start_pad,
                end_pad,
                keep_start,
                keep_end,
            ) in enumerate(blocks):
                futures.append(
                    ex.submit(
                        _process_block,
                        (
                            block_index,
                            start,
                            end,
                            start_pad,
                            end_pad,
                            keep_start,
                            keep_end,
                            shm_rgb.name,
                            rgb_shape,
                            rgb_dtype,
                            shm_alpha.name,
                            alpha_shape,
                            alpha_dtype,
                            shm_plab.name,
                            plab_shape,
                            plab_dtype,
                            shm_plch.name,
                            plch_shape,
                            plch_dtype,
                            shm_prgb.name,
                            prgb_shape,
                            prgb_dtype,
                            progress_shm.name,
                            len(blocks),
                            float(a_bias),
                            float(b_bias),
                            int(topk_global),
                            int(mp_threads),
                        ),
                    )
                )

            pending = set(futures)
            while pending:
                # Poll progress
                done, pending = wait(pending, timeout=0.2, return_when=FIRST_COMPLETED)
                # Merge any completed blocks
                for fut in done:
                    keep_start, keep_end, kept = fut.result()
                    out[keep_start:keep_end] = kept

                # Compute smooth ETA from shared counters
                rows_done = int(np.int64(progress_view.sum()))
                now = time.perf_counter()
                rows_delta = rows_done - last_rows_done
                dt = max(1e-6, now - last_tick)
                if rows_delta > 0:
                    eta = eta_est.update_rows(rows_delta, dt)
                    pct = 100.0 * rows_done / max(1, total_kept_rows)
                    _progress(pct, eta, final=False)
                    last_rows_done = rows_done
                    last_tick = now

        prof_mp["t_total"] = time.perf_counter() - t_total0
        _progress(100.0, 0.0, final=True)

        if profile:
            total = max(1e-9, prof_mp["t_total"])

            def pct_str(t: float) -> str:
                return f"{100.0 * t / total:5.1f}%"

            def row(label: str, secs: float) -> str:
                return (
                    f"  {label:<16} {format_seconds_compact(secs):>9}  {pct_str(secs)}"
                )

            debug_log("[profile] photo timing (mp blocks, shm):")
            print(row("total", prof_mp["t_total"]), flush=True)
            print(row("rgb->lab + bias", prof_mp["t_bias_rgb2lab"]), flush=True)

        return out
    finally:
        # Clean up SHM
        for shm in (shm_rgb, shm_alpha, shm_plab, shm_plch, shm_prgb):
            try:
                shm.close()
                shm.unlink()
            except FileNotFoundError:
                pass

        if progress_shm is not None:
            try:
                progress_shm.close()
            except Exception:
                pass
            try:
                progress_shm.unlink()
            except Exception:
                pass
