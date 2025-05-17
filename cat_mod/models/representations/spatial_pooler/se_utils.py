#  Copyright (c) 2023 Autonomous Non-Profit Organization "Artificial Intelligence Research
#  Institute" (AIRI); Moscow Institute of Physics and Technology (National Research University).
#  All rights reserved.
#
#  Licensed under the AGPLv3 license. See LICENSE in the project root for license information.

from __future__ import annotations

from enum import Enum, auto

import numpy as np
from numba import jit
from numpy import typing as npt
from numpy.random import Generator

from cat_mod.models.representations.spatial_pooler.sdr import RateSdr
from cat_mod.models.representations.spatial_pooler.sdr_array import SdrArray
from cat_mod.models.representations.spatial_pooler.utils import isnone


class WeightsDistribution(Enum):
    NORMAL = 1
    UNIFORM = auto()
    LOG_NORMAL = auto()


class FilterInputPolicy(Enum):
    NO = 0
    SUBTRACT_AVG = auto()
    SUBTRACT_AVG_AND_CLIP = auto()


class BackendType(Enum):
    DENSE = 1
    SPARSE = auto()


class BoostingPolicy(Enum):
    NO = 0
    ADDITIVE = auto()
    MULTIPLICATIVE = auto()


class LearningPolicy(Enum):
    # W \cdot x
    LINEAR = 1
    # W^{p-1} \cdot x
    KROTOV = auto()


class ActivationPolicy(Enum):
    POWERLAW = 1
    EXPONENTIAL = auto()


def align_matching_learning_params(
        match_p: float | None, lebesgue_p: float, learning_policy: str
) -> tuple[float, float, LearningPolicy]:
    learning_policy = LearningPolicy[learning_policy.upper()]
    # check learning rule with p-norm compatibility
    assert (
            (learning_policy == LearningPolicy.LINEAR and lebesgue_p == 1.0) or
            (learning_policy == LearningPolicy.KROTOV and lebesgue_p > 1.0)
    ), f'{learning_policy} is incompatible with p-norm {lebesgue_p}'

    # check learning rule with matching power p compatibility
    if learning_policy == LearningPolicy.LINEAR:
        # for linear learning, we allow any weights power, default is 1.0 (linear)
        match_p = isnone(match_p, 1.0)
    elif learning_policy == LearningPolicy.KROTOV:
        # for krotov learning, power is fixed to weights' norm p - 1
        induced_match_p = lebesgue_p - 1
        assert isnone(match_p, induced_match_p) == induced_match_p
        match_p = induced_match_p

    return match_p, lebesgue_p, learning_policy


def sample_weights(
        rng, w_shape, distribution, radius, lebesgue_p, inhibitory_ratio=0.5
):
    if distribution == WeightsDistribution.NORMAL:
        init_std = get_normal_std(w_shape[1], lebesgue_p, radius)
        weights = np.abs(rng.normal(loc=0., scale=init_std, size=w_shape))

    elif distribution == WeightsDistribution.UNIFORM:
        init_std = get_uniform_std(w_shape[1], lebesgue_p, radius)
        weights = rng.uniform(0., init_std, size=w_shape)

    elif distribution == WeightsDistribution.LOG_NORMAL:
        init_std = 0.3
        weights = rng.lognormal(mean=0., sigma=init_std, size=w_shape)
        rs = np.sum(weights ** lebesgue_p, -1, keepdims=True) ** (1 / lebesgue_p)
        rs /= radius
        weights /= rs

    else:
        raise ValueError(f'Unsupported distribution: {distribution}')

    # make a portion of weights negative
    if inhibitory_ratio > 0.0:
        inh_mask = rng.binomial(1, inhibitory_ratio, size=w_shape).astype(bool)
        weights[inh_mask] *= -1.0

    return weights


def get_uniform_std(n_samples, p, required_r) -> float:
    alpha = 2 / n_samples
    alpha = alpha ** (1 / p)
    return required_r * alpha


def get_normal_std(n_samples: int, p: float, required_r) -> float:
    alpha = np.pi / (2 * n_samples)
    alpha = alpha ** (1 / p)
    return required_r * alpha


def arg_top_k(x, k):
    k = min(k, x.size)
    return np.argpartition(x, -k, axis=-1)[..., -k:]


def boosting(
        relative_rate: float | npt.NDArray[float], k: float | npt.NDArray[float],
        *, min_k: float | npt.NDArray[float] = 0.0, softness: float = 3.0
) -> float:
    # relative rate: rate / R_target
    # x = -log(relative_rate)
    #   0 1 +inf  -> +inf 0 -inf
    x = -np.log(relative_rate)

    # relative_rate -> x -> B:
    #   0 -> +inf -> K^tanh(+inf) = K
    #   1 -> 0 -> K^tanh(0) = 1
    #   +inf -> -inf -> K^tanh(-inf) = 1 / K
    # higher softness just makes the sigmoid curve smoother; default value is empirically optimized
    return np.power(k + min_k + 1.0, np.tanh(x / softness))


@jit
def nb_choice(rng, p):
    """Choose a sample from N values with weights p (they could be non-normalized)."""
    # Get cumulative weights
    acc_w = np.cumsum(p)
    # Total of weights
    mx_w = acc_w[-1]
    r = mx_w * rng.random()
    # Get corresponding index
    ind = np.searchsorted(acc_w, r, side='right')
    return ind


@jit()
def nb_choice_k(
        rng: Generator, k: int, weights: npt.NDArray[np.float64] = None, n: int = None,
        replace: bool = False, cache: npt.NDArray[np.bool_] = None
):
    """
    Choose k samples from max_n values, with optional weights and replacement.

    Be careful using this function with very skewed weights w/o replacement, as it may be slow and
    end up with time limit exception. For such cases, consider using numpy choice with weights.
    The reason for this is that when probability mass is concentrated in a few values,
    the random sampling will repeatedly produce duplicates, which are then rejected
    due to the no-replacement constraint. So, the best case scenario is when the weights are
    close to uniform.
    """
    acc_w = np.cumsum(weights) if weights is not None else np.arange(0, n, 1, dtype=np.float64)
    # Total of weights
    mx_w = acc_w[-1]
    # result
    result = np.full(k, -1, dtype=np.int64)
    if not replace and cache is None and n is not None:
        cache = np.zeros(n, dtype=np.bool_)

    i, j = 0, 0
    # reasonable time limit: avg 100 tries for each required sample, growing as the
    # choice getting more dense (n / k -> 1)
    timelimit = min(10_000_000, k * 100 ** (1.0 + k / n))
    while i < k and j < timelimit:
        j += 1
        r = mx_w * rng.random()
        ind = np.searchsorted(acc_w, r, side='right')

        if not replace and cache[ind]:
            continue
        else:
            result[i] = ind
            if not replace:
                cache[ind] = True
            i += 1

    if j >= timelimit:
        # print(f'{j=}: {i=} of {k=} in {n=} | {mx_w=}')
        # with np.printoptions(precision=3):
        #     print(f'{acc_w[:20]=}')
        #     print(f'{acc_w[-20:]=}')
        raise ValueError('Infinite loop in nb_choice_k. If weights are degenerate, use numpy')

    return result


def pow_x(x, p, has_negative):
    if p == 1.0:
        return x

    if has_negative:
        return np.sign(x) * (np.abs(x) ** p)
    return x ** p


def abs_pow_x(x, p, has_negative):
    if has_negative:
        x = np.abs(x)
    if p == 1.0:
        return x
    return x ** p


def dot_match(x, w):
    return np.dot(w, x.T).T


def min_match(x, w):
    inter = np.empty_like(w)
    return np.vstack([
        np.sum(np.fmin(w, xx, out=inter), -1)
        for xx in x
    ])


@jit()
def min_match_j(x, w):
    n_batch = x.shape[0]
    n_out, n_in = w.shape
    res = np.zeros((n_batch, n_out))
    for k in range(n_batch):
        for i in range(n_out):
            t = 0
            for xx, ww in zip(x[k], w[i]):
                t += min(xx, ww)
            res[k, i] = t
    return res


def dot_match_sparse(x: SdrArray, *, w, ixs_srt_j, shifts, srt_i):
    is_batch = isinstance(x, SdrArray)
    match_func = _match_sparse_batch if is_batch else _match_sparse
    return match_func(
        _dot_match_sparse, x, w=w, ixs_srt_j=ixs_srt_j, shifts=shifts, srt_i=srt_i
    )


def min_match_sparse(x: SdrArray, *, w, ixs_srt_j, shifts, srt_i):
    is_batch = isinstance(x, SdrArray)
    match_func = _match_sparse_batch if is_batch else _match_sparse
    return match_func(
        _min_match_sparse, x, w=w, ixs_srt_j=ixs_srt_j, shifts=shifts, srt_i=srt_i
    )


def _match_sparse_batch(
        match_func, x: SdrArray, *, w, ixs_srt_j, shifts, srt_i, out=None
):
    batch_size = len(x)
    n_out = srt_i.shape[0]

    out = np.zeros((batch_size, n_out)) if out is None else out
    for i in range(batch_size):
        match_func(
            w, ixs_srt_j, shifts, x.sparse[i].sdr, x.sparse[i].values,
            out[i, :]
        )
    return out


def _match_sparse(
        match_func, x: RateSdr, *, w, ixs_srt_j, shifts, srt_i, out=None
):
    n_out = srt_i.shape[0]
    out = np.zeros(n_out) if out is None else out
    return match_func(w, ixs_srt_j, shifts, x.sdr, x.values, out)


@jit()
def _dot_match_sparse(wi_sp, jxs, shifts, sdr, rates, out):
    for i, r in zip(sdr, rates):
        for j in range(shifts[i], shifts[i + 1]):
            out[jxs[j]] += r * wi_sp[j]
    return out


@jit()
def _min_match_sparse(wi_sp, jxs, shifts, sdr, rates, out):
    for i, r in zip(sdr, rates):
        for j in range(shifts[i], shifts[i + 1]):
            out[jxs[j]] += min(r, wi_sp[j])
    return out


def norm_p(x, p, has_negative):
    return abs_pow_x(
        np.sum(abs_pow_x(x, p, has_negative), -1),
        (1 / p),
        # even if they were negative, already took abs
        False
    )


def normalize(x, p=1.0, has_negative=False):
    r = norm_p(x, p, has_negative)
    eps = 1e-30
    if x.ndim > 1:
        mask = r > eps
        res = x.copy()
        res[mask] /= np.expand_dims(r[mask], -1)
        return res

    return x / r if r > eps else x