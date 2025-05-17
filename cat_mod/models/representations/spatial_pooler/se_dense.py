#  Copyright (c) 2022 Autonomous Non-Profit Organization "Artificial Intelligence Research
#  Institute" (AIRI); Moscow Institute of Physics and Technology (National Research University).
#  All rights reserved.
#
#  Licensed under the AGPLv3 license. See LICENSE in the project root for license information.
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt
from numba import jit
from numpy.random import Generator

from cat_mod.models.representations.spatial_pooler.timer import timed
from cat_mod.models.representations.spatial_pooler.se_utils import (
    sample_weights, WeightsDistribution,
    pow_x, dot_match, norm_p, min_match, LearningPolicy, align_matching_learning_params, BackendType
)

if TYPE_CHECKING:
    from cat_mod.models.representations.spatial_pooler.se import SpatialEncoderLayer


class SpatialEncoderDenseBackend:
    """
    An implementation of dense weights keeping and calculations for the spatial encoder.
    """
    owner: SpatialEncoderLayer
    type: BackendType = BackendType.DENSE

    rng: Generator

    # connections
    weights: npt.NDArray[float]

    rf_sparsity: float
    pruned_mask: npt.NDArray[bool] | None
    rf: npt.NDArray[int] | None

    lebesgue_p: float
    radius: npt.NDArray[float]
    pos_log_radius: npt.NDArray[float]

    # potentiation
    match_p: float
    weights_pow_p: npt.NDArray[float] | None
    match_op: callable
    match_op_name: str

    # learning
    learning_policy: LearningPolicy
    learning_rate: float

    def __init__(
            self, *, owner,

            lebesgue_p: float = 1.0, init_radius: float = 10.0,
            weights_distribution: str = 'normal',
            initial_rf_sparsity: float = 1.0,

            match_p: float = 1.0, match_op: str = 'mul',

            learning_policy: str = 'linear',
    ):
        self.owner = owner
        # set it immediately so we can use pruning controller that relies on it
        self.owner.weights_backend = self

        self.rng = np.random.default_rng(self.owner.rng.integers(100_000_000))

        # ==> Weights initialization
        n_in, n_out = self.owner.ff_size, self.owner.output_size
        w_shape = (n_out, n_in)

        match_p, lebesgue_p, learning_policy = align_matching_learning_params(
            match_p, lebesgue_p, learning_policy
        )

        weights_distribution = WeightsDistribution[weights_distribution.upper()]
        self.lebesgue_p = lebesgue_p
        self.weights = sample_weights(
            self.rng, w_shape, weights_distribution, init_radius, self.lebesgue_p,
            inhibitory_ratio=0.0
        )
        self.has_inhibitory = False
        self.radius = self.get_radius()
        self.pos_log_radius = self.get_pos_log_radius()

        # initial sparsity will be set a bit later after required initializations
        self.rf_sparsity = 1.0
        # init with None, they will be initialized on demand when needed
        self.pruned_mask = None
        self.rf = None

        # ==> Pattern matching
        self.match_p = match_p
        self.weights_pow_p = None
        if self.match_p != 1.0:
            self.weights_pow_p = self.get_weight_pow_p()

        can_use_min_operator = self.match_p == 1.0 and learning_policy == LearningPolicy.LINEAR
        if match_op == 'min' and can_use_min_operator:
            self.match_op = min_match
            self.match_op_name = 'min'
        else:
            self.match_op = dot_match
            self.match_op_name = 'mul'

        # ==> Learning
        self.learning_policy = learning_policy

        # set initial sparsity and prune excess connections
        self.set_sparsify_level(initial_rf_sparsity)

    def match_input(self, x):
        w = self.weights if self.match_p == 1.0 else self.weights_pow_p
        return self.match_op(x, w)

    def update_weights(self, x, y_sdr, y_rates, u, lr):
        if y_sdr.size == 0:
            return

        rf = None if self.rf_sparsity == 1.0 else self.rf

        # TODO: negative Xs is not supported ATM
        if self.learning_policy == LearningPolicy.KROTOV:
            oja_krotov_update(self.weights, x, u, y_sdr, y_rates, lr, rf)
        else:
            willshaw_update(self.weights, x, y_sdr, y_rates, lr, rf)

        if self.match_p != 1.0:
            self.weights_pow_p[y_sdr] = self.get_weight_pow_p(y_sdr)

        self.radius[y_sdr] = self.get_radius(y_sdr)
        self.pos_log_radius[y_sdr] = self.get_pos_log_radius(y_sdr)

    def apply_pruning_step(self):
        pc = self.owner.pruning_controller

        # move to the next step to get new current sparsity
        sparsity = pc.next_newborn_stage()

        # set current sparsity and prune excess connections
        # noinspection PyNoneFunctionAssignment,PyTupleAssignmentBalance
        _, t = self.set_sparsify_level(sparsity)

        stage = pc.stage
        sparsity_pct = round(100.0 * sparsity, 1)
        t = round(t * 1000.0, 2)
        print(f'Prune #{stage}: {sparsity_pct:.1f}% | {self.rf_size} | {t} ms')

    @timed
    def set_sparsify_level(self, sparsity):
        if sparsity >= self.rf_sparsity:
            # if feedforward sparsity is tracked, then it may change and lead to RF increase
            # ==> leave RF as is
            return

        self.rf_sparsity = sparsity
        pc = self.owner.pruning_controller
        if self.pruned_mask is None:
            self.pruned_mask = np.zeros_like(self.weights, dtype=bool)

        pc.prune_receptive_field()
        # update alive connections
        self.rf = np.array([
            np.flatnonzero(~neuron_connections)
            for neuron_connections in self.pruned_mask
        ])

        # Since a portion of weights is pruned, the norm is changed. So, we either should
        # update the radius or rescale weights to keep the norm unchanged. The latter might be
        # better to avoid interfering to the norm convergence process, because a lot of parameters
        # depend on the norm, like boosting or learning rate, and the pruning itself does not
        # affect all neurons' norms equally, so the balance may be broken.
        old_radius, new_radius = self.radius, self.get_radius()
        # I keep pow weights the same — each will be updated on its next learning step.
        # So, it's a small performance optimization.
        self.weights *= np.expand_dims(old_radius / new_radius, -1)

    def get_radius(self, ixs: npt.NDArray[int] = None) -> npt.NDArray[float]:
        p = self.lebesgue_p
        w = self.weights if ixs is None else self.weights[ixs]
        return norm_p(w, p, self.has_inhibitory)

    def get_weight_pow_p(self, ixs: npt.NDArray[int] = None) -> npt.NDArray[float]:
        p = self.match_p
        w = self.weights if ixs is None else self.weights[ixs]
        return pow_x(w, p, self.has_inhibitory)

    def get_pos_log_radius(self, ixs: npt.NDArray[int] = None) -> npt.NDArray[float]:
        r = self.radius if ixs is None else self.radius[ixs]
        return np.log2(np.maximum(r, 1.0))

    def plot_weights_distr(self):
        import seaborn as sns
        from matplotlib import pyplot as plt
        w, r = self.weights, self.radius
        w = w / np.expand_dims(r, -1)
        p = self.match_p
        _w = pow_x(w, p, False).flatten()
        eps = 1.0 / 20.0 / self.owner.ff_size
        _w = _w[_w > eps]
        sns.histplot(_w)
        plt.show()

    @property
    def avg_radius(self):
        return self.radius.mean()

    @property
    def rf_size(self):
        return round(self.rf_sparsity * self.owner.ff_size)


@jit()
def willshaw_update(weights, x, y_sdr, y_rates, lr, alive_connections):
    # Willshaw learning rule, L1 normalization:
    # dw = lr * y * (x - w)

    v = y_rates * lr
    for ix, vi in zip(y_sdr, v):
        w = weights[ix]

        if alive_connections is None:
            w += vi * (x - w)
            fix_anti_hebbian_negatives(vi, w, None)
        else:
            m = alive_connections[ix]
            w[m] += vi * (x[m] - w[m])
            fix_anti_hebbian_negatives(vi, w, m)


@jit()
def oja_krotov_update(weights, x, u, y_sdr, y_rates, lr, alive_connections):
    # Oja-Krotov learning rule, L^p normalization, p >= 2:
    # dw = lr * y * (x - u * w)

    # NB: u sign persistence is not supported ATM
    # NB2: it also replaces dw normalization
    v = y_rates * lr
    alpha = _get_scale(u)
    if alpha > 1.0:
        v /= alpha

    for ix, vi in zip(y_sdr, v):
        ui = u[ix]
        w = weights[ix]

        if alive_connections is None:
            w += vi * (x - ui * w)
            fix_anti_hebbian_negatives(vi, w, None)
        else:
            m = alive_connections[ix]
            w[m] += vi * (x[m] - ui * w[m])
            fix_anti_hebbian_negatives(vi, w, m)


@jit()
def fix_anti_hebbian_negatives(vi, w, mask):
    # NB: Anti-hebbian learning may cause sign change from positive to negative (when the
    # input x is always positive). This function fixes such cases.
    # `vi` here is a value indicating hebbian (>0) or anti-hebbian (<0)
    if vi >= 0:
        return

    if mask is None:
        np.fmax(w, 0.0, w)
    else:
        w[mask] = np.fmax(w[mask], 0.0)

@jit()
def _get_scale(u):
    u_max = np.max(np.abs(u))
    return 1.0 if u_max < 1.0 else u_max ** 0.75 if u_max < 100.0 else u_max ** 0.9