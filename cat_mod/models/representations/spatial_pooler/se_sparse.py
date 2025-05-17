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

from cat_mod.models.representations.spatial_pooler.sdr import (
    RateSdr
)
from cat_mod.models.representations.spatial_pooler.timer import timed
from cat_mod.models.representations.spatial_pooler.se_utils import (
    pow_x, norm_p, dot_match_sparse, LearningPolicy, BackendType, min_match_sparse
)

if TYPE_CHECKING:
    from cat_mod.models.representations.spatial_pooler.se import SpatialEncoderLayer


class SpatialEncoderSparseBackend:
    """
    An implementation of sparse weights keeping and calculations for the spatial encoder.
    """
    owner: SpatialEncoderLayer
    type: BackendType = BackendType.SPARSE

    rng: Generator

    # connections
    # Indices naming convention:
    #   i - postsynaptic neurons,
    #   j - presynaptic neurons,
    #   k - synaptic connections

    # flatten synaptic connections, `k`-th connection stores an index of the
    # postsynaptic neuron `i`. Connections are sorted by presynaptic neurons `j`
    ixs_srt_j: npt.NDArray[int]
    # flatten synaptic connections' corresponding weights.
    weights: npt.NDArray[float]
    # defines partition of synaptic connections by the presynaptic neurons.
    # `j`-th presynaptic neuron's connections are {k \in [shifts_j[j], shifts_j[j+1])}
    shifts_j: npt.NDArray[int]
    # srt_i: 2D matrix, each row corresponds to postsynaptic neuron `i` and contains indices
    # of connections [in weights and ixs_srt_j] that define its receptive field.
    # Indices are sorted ASC (i.e. by presynaptic neuron `j`)
    # NB: there's neither explicit i -> j mapping, nor j -> i. But it's possible to reconstruct
    #   both efficiently from the given data.
    kxs_srt_ij: npt.NDArray[int]

    rf_sparsity: float

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
            self, *, dense_backend,
            # TODO: implement initialization from params
            # seed: int,
            # feedforward_sds: Sds, output_sds: Sds,

            # adapt_to_ff_sparsity,

            # lebesgue_p: float = 1.0,
            # init_radius: float = 10.0,
            # weights_distribution: WeightsDistribution = WeightsDistribution.NORMAL,
            # inhibitory_ratio: float = 0.0,

            # match_p: float = 1.0, match_op: str = 'mul',

            # learning_policy: LearningPolicy = LearningPolicy.LINEAR,
            # pruning: TConfig = None,
    ):
        self.owner = dense_backend.owner
        # set it immediately so we can use pruning controller that relies on it
        self.owner.weights_backend = self

        seed = self.owner.rng.integers(100_000_000)
        self.rng = np.random.default_rng(seed)

        # ==> Weights initialization
        self.lebesgue_p = dense_backend.lebesgue_p
        self.rf_sparsity = dense_backend.rf_sparsity
        weights, ixs_srt_j, shifts_j, kxs_srt_ij = make_sparse_weights_from_dense(
            dense_backend.weights, dense_backend.rf
        )
        self.weights = weights
        self.ixs_srt_j = ixs_srt_j
        self.shifts_j = shifts_j
        self.kxs_srt_ij = kxs_srt_ij

        self.radius = self.get_radius()
        self.pos_log_radius = self.get_pos_log_radius()

        # ==> Pattern matching
        self.match_p = dense_backend.match_p
        self.weights_pow_p = None
        if self.match_p != 1.0:
            self.weights_pow_p = self.get_weight_pow_p()

        learning_policy = dense_backend.learning_policy
        match_op = dense_backend.match_op_name
        can_use_min_operator = self.match_p == 1.0 and learning_policy == LearningPolicy.LINEAR
        if match_op == 'min' and can_use_min_operator:
            self.match_op = min_match_sparse
            self.match_op_name = 'min'
        else:
            self.match_op = dot_match_sparse
            self.match_op_name = 'mul'

        # ==> Learning
        self.learning_policy = learning_policy

    def match_input(self, x):
        w = self.weights if self.match_p == 1.0 else self.weights_pow_p
        return self.match_op(
            x, w=w, ixs_srt_j=self.ixs_srt_j, shifts=self.shifts_j, srt_i=self.kxs_srt_ij
        )

    def update_weights(self, x: RateSdr, y_sdr, y_rates, u, lr):
        if y_sdr.size == 0:
            return

        x_dense = x.to_dense(self.owner.feedforward_sds.size)

        # TODO: negative Xs is not supported ATM
        if self.learning_policy == LearningPolicy.KROTOV:
            oja_krotov_update_sparse(
                self.weights, self.shifts_j, self.kxs_srt_ij, x_dense, u, y_sdr, y_rates, lr
            )
        else:
            willshaw_update_sparse(
                self.weights, self.shifts_j, self.kxs_srt_ij, x_dense, y_sdr, y_rates, lr
            )

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

        pc.prune_receptive_field()
        weights, ixs_srt_j, shifts_j, kxs_srt_ij = remove_pruned_synapses(
            self.weights, self.ixs_srt_j, self.shifts_j, self.kxs_srt_ij,
            self.rf_size
        )
        self.weights = weights
        self.ixs_srt_j = ixs_srt_j
        self.shifts_j = shifts_j
        self.kxs_srt_ij = kxs_srt_ij

        # Since a portion of weights is pruned, the norm is changed. So, we either should
        # update the radius or rescale weights to keep the norm unchanged. The latter might be
        # better to avoid interfering to the norm convergence process, because a lot of parameters
        # depend on the norm, like boosting or learning rate, and the pruning itself does not
        # affect all neurons' norms equally, so the balance may be broken.
        old_radius, new_radius = self.radius, self.get_radius()
        # I keep pow weights the same — each will be updated on its next learning step.
        # So, it's a small performance optimization.
        self.weights *= (old_radius / new_radius)[self.ixs_srt_j]

    def get_radius(self, ixs: npt.NDArray[int] = None) -> npt.NDArray[float]:
        p = self.lebesgue_p
        if ixs is None:
            w = self.weights[self.kxs_srt_ij]
        else:
            w = self.weights[self.kxs_srt_ij[ixs]]
        return norm_p(w, p, False)

    def get_weight_pow_p(self, ixs: npt.NDArray[int] = None) -> npt.NDArray[float]:
        p = self.match_p
        if ixs is None:
            w = self.weights[self.kxs_srt_ij]
        else:
            w = self.weights[self.kxs_srt_ij[ixs]]
        return pow_x(w, p, False)

    def get_pos_log_radius(self, ixs: npt.NDArray[int] = None) -> npt.NDArray[float]:
        r = self.radius if ixs is None else self.radius[ixs]
        return np.log2(np.maximum(r, 1.0))

    def plot_weights_distr(self):
        import seaborn as sns
        from matplotlib import pyplot as plt
        w, r = self.weights, self.radius
        r = r.mean()
        w = w / r
        p = self.match_p
        _w = pow_x(w, p, False)
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


def make_sparse_weights_from_dense(dense_weights, idxs):
    n_out, n_in = dense_weights.shape
    # idxs: row — postsynaptic (i), col — presynaptic (j)
    sparse_weights = np.take_along_axis(dense_weights, idxs, -1).copy()
    w_f = sparse_weights.flatten()

    # jxs_srt_i: pre sorted by post
    # defines synaptic connections (k)
    jxs_srt_i = idxs.flatten()

    kxs_srt_j = np.argsort(jxs_srt_i, kind='stable')
    # post sorted by post
    ixs_srt_i = np.repeat(np.arange(n_out), idxs.shape[1])

    # post sorted by pre
    ixs_srt_j = ixs_srt_i[kxs_srt_j].copy()
    # weights sorted by pre
    w_f_srt_j = w_f[kxs_srt_j].copy()
    # i -> shift for i-th presynaptic connections
    shifts_j = np.pad(np.cumsum(np.bincount(jxs_srt_i[kxs_srt_j])), (1, 0))

    kxs_srt_ij = np.argsort(ixs_srt_j, kind='stable')
    kxs_srt_ij = kxs_srt_ij.reshape(sparse_weights.shape)

    return w_f_srt_j, ixs_srt_j, shifts_j, kxs_srt_ij


@jit()
def remove_pruned_synapses(weights, ixs_srt_j, shifts_j, kxs_srt_ij, rf_size):
    n_neurons = kxs_srt_ij.shape[0]
    n_synapses = n_neurons * rf_size

    i_shifts = np.arange(n_neurons) * rf_size
    _kxs_srt_ij = np.empty(n_synapses, np.int_)

    _shifts_j = np.empty_like(shifts_j)
    _shifts_j[0] = 0

    _k, j = 0, 0
    for k, i in enumerate(ixs_srt_j):
        if k >= shifts_j[j+1]:
            _shifts_j[j+1] = _k
            j += 1
        if i == -1:
            continue
        ixs_srt_j[_k] = i
        _kxs_srt_ij[i_shifts[i]] = _k
        i_shifts[i] += 1
        _k += 1

    _shifts_j[j + 1] = n_synapses

    # assert np.all(i_shifts == (np.arange(n_neurons) + 1) * rf_size)
    # assert _k == n_synapses

    weights = weights[:n_synapses].copy()
    ixs_srt_j = ixs_srt_j[:n_synapses].copy()
    _kxs_srt_ij = _kxs_srt_ij.reshape(n_neurons, rf_size)
    return weights, ixs_srt_j, _shifts_j, _kxs_srt_ij


@jit()
def willshaw_update_sparse(weights, shifts_j, kxs_srt_ij, x, y_sdr, y_rates, lr):
    # Willshaw learning rule, L1 normalization:
    # dw = lr * y * (x - w)
    w = weights

    v = y_rates * lr
    for i, vi in zip(y_sdr, v):
        j = 0

        # traverse synaptic connections `k` of the postsynaptic neuron `i`
        for k in kxs_srt_ij[i]:
            # for each connection, find the corresponding presynaptic neuron `j`
            while k >= shifts_j[j+1]:
                j += 1

            w[k] += vi * (x[j] - w[k])
            if w[k] < 0.:  # fix anti-hebbian
                w[k] = 0.0


@jit()
def oja_krotov_update_sparse(weights, shifts_j, kxs_srt_ij, x, u, y_sdr, y_rates, lr):
    # Oja-Krotov learning rule, L^p normalization, p >= 2:
    # dw = lr * y * (x - u * w)
    w = weights

    # NB: dw normalization is replaced with soft rescaling
    v = y_rates * lr
    alpha = _get_scale(u)
    if alpha > 1.0:
        v /= alpha

    for i, vi in zip(y_sdr, v):
        ui = u[i]
        j = 0

        # traverse synaptic connections `k` of the postsynaptic neuron `i`
        for k in kxs_srt_ij[i]:
            # for each connection, find the corresponding presynaptic neuron `j`
            while k >= shifts_j[j+1]:
                j += 1

            w[k] += vi * (x[j] - ui * w[k])
            if w[k] < 0.:  # fix anti-hebbian
                w[k] = 0.0

@jit()
def _get_scale(u):
    u_max = np.max(np.abs(u))
    return 1.0 if u_max < 1.0 else u_max ** 0.75 if u_max < 100.0 else u_max ** 0.9