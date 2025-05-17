#  Copyright (c) 2023 Autonomous Non-Profit Organization "Artificial Intelligence Research
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


from cat_mod.models.representations.spatial_pooler.scheduler import Scheduler
from cat_mod.models.representations.spatial_pooler.se_utils import nb_choice_k, BackendType
from cat_mod.models.representations.spatial_pooler.sp import SpNewbornPruningMode

if TYPE_CHECKING:
    from cat_mod.models.representations.spatial_pooler.se import SpatialEncoderLayer


class PruningController:
    owner: SpatialEncoderLayer

    mode: SpNewbornPruningMode
    n_stages: int
    stage: int
    scheduler: Scheduler

    initial_rf_sparsity: float
    target_rf_to_input_ratio: float
    target_max_rf_sparsity: float

    def __init__(
            self, owner,
            mode: str, cycle: float, n_stages: int,
            initial_rf_sparsity: float,
            target_max_rf_sparsity: float = None, target_rf_to_input_ratio: float = None
    ):
        self.owner = owner

        # noinspection PyTypeChecker
        self.mode = SpNewbornPruningMode[mode.upper()]
        self.n_stages = n_stages
        self.stage = 0

        schedule = int(cycle / owner.output_sds.sparsity)
        self.scheduler = Scheduler(schedule)

        self.initial_rf_sparsity = initial_rf_sparsity
        self.target_rf_to_input_ratio = target_rf_to_input_ratio
        self.target_max_rf_sparsity = target_max_rf_sparsity

    @property
    def is_newborn_phase(self):
        return self.stage < self.n_stages

    def next_newborn_stage(self) -> float:
        self.stage += 1
        return self._get_current_stage_sparsity()

    def prune_receptive_field(self):
        # sample what connections to keep for each neuron independently
        backend = self.owner.weights_backend
        if backend.type == BackendType.DENSE:
            prune(
                self.owner.rng, backend.weights, backend.weights_pow_p,
                backend.rf_size, backend.pruned_mask
            )
        else:
            prune_sparse(
                self.owner.rng, backend.weights, backend.weights_pow_p,
                backend.rf_size, backend.ixs_srt_j, backend.kxs_srt_ij
            )

    def get_target_rf_sparsity(self):
        if self.target_max_rf_sparsity is not None:
            return self.target_max_rf_sparsity

        if self.owner.adapt_to_ff_sparsity:
            ff_sparsity = self.owner.ff_avg_sparsity
        else:
            ff_sparsity = self.owner.feedforward_sds.sparsity

        return self.target_rf_to_input_ratio * ff_sparsity

    def _get_current_stage_sparsity(self):
        if self.mode == SpNewbornPruningMode.LINEAR:
            sparsity_progress_func = self._newborn_linear_progress
        elif self.mode == SpNewbornPruningMode.POWERLAW:
            sparsity_progress_func = self._newborn_powerlaw_progress
        else:
            raise ValueError(f'Pruning mode {self.mode} is not supported')
        return sparsity_progress_func(
            initial=self.initial_rf_sparsity, target=self.get_target_rf_sparsity()
        )

    def _newborn_linear_progress(self, initial, target):
        newborn_phase_progress = self.stage / self.n_stages
        # linear decay rule
        return initial + newborn_phase_progress * (target - initial)

    # noinspection PyUnusedLocal
    def _newborn_powerlaw_progress(self, initial, target):
        steps_left = self.n_stages - self.stage + 1
        current = self.owner.rf_sparsity
        # determine, which decay is needed to reach the target in the remaining steps
        # NB: recalculate each step to exclude rounding errors
        decay = np.power(target / current, 1 / steps_left)
        # exponential decay rule
        return current * decay


@jit()
def prune(
        rng: Generator, weights: npt.NDArray[float], pow_weights: npt.NDArray[float],
        k: int, pruned_mask
):
    # WARNING: works only with non-negative weights!
    n_neurons, _ = weights.shape
    w_priority = weights if pow_weights is None else pow_weights

    for row in range(n_neurons):
        pm_row = pruned_mask[row]
        w_row = weights[row]
        w_priority_row = w_priority[row]

        active_mask = ~pm_row
        prune_probs = pruning_probs_from_synaptic_weights(w_priority_row[active_mask])

        # pruned connections are marked as already selected for "select K from N" operation
        n_active = len(prune_probs)
        not_k = n_active - k
        ixs = nb_choice_k(rng, not_k, prune_probs, n_active, False)
        new_pruned_ixs = np.flatnonzero(active_mask)[ixs]
        w_row[new_pruned_ixs] = 0.0
        pm_row[new_pruned_ixs] = True
        if pow_weights is not None:
            pow_weights[row][new_pruned_ixs] = 0.0


@jit()
def prune_sparse(
        rng: Generator, weights: npt.NDArray[float], pow_weights: npt.NDArray[float],
        k: int, ixs_srt_j, kxs_srt_ij
):
    # WARNING: works only with non-negative weights!
    w_priority = weights if pow_weights is None else pow_weights
    for kxs in kxs_srt_ij:
        w_priority_row = w_priority[kxs]
        prune_probs = pruning_probs_from_synaptic_weights(w_priority_row)

        # pruned connections are marked as already selected for "select K from N" operation
        n_active = len(prune_probs)
        not_k = n_active - k
        pruned_ixs = nb_choice_k(rng, not_k, prune_probs, n_active, False)
        # mark as pruned with -1
        ixs_srt_j[kxs[pruned_ixs]] = -1


@jit()
def pruning_probs_from_synaptic_weights(weights):
    priority = weights.copy()
    # normalize relative to the mean: < 1 are weak, > 1 are strong
    priority /= priority.mean()
    # clip to avoid numerical issues: low values threshold is safe to keep enough information
    #   i.e. we keep info until the synapse is 1mln times weaker than the average
    np.clip(priority, 1e-6, 1e+6, priority)
    # linearize the scales -> [-X, +Y], where X,Y are low < 100
    priority = np.log(priority)
    # -> shift to negative [-(X+Y), 0] -> flip to positive [0, X+Y] -> add baseline probability
    #   the weakest synapses are now have the highest probability
    prune_probs = -(priority - priority.max()) + 0.1
    return prune_probs