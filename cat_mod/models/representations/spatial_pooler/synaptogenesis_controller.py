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
from cat_mod.models.representations.spatial_pooler.sds import Sds
from cat_mod.models.representations.spatial_pooler.utils import safe_divide

if TYPE_CHECKING:
    from cat_mod.models.representations.spatial_pooler.se import SpatialEncoderLayer
    from cat_mod.models.representations.spatial_pooler.se_sparse import SpatialEncoderSparseBackend


class SynaptogenesisController:
    owner: SpatialEncoderLayer
    rng: Generator
    stats_update_scheduler: Scheduler

    def __init__(
            self, *,
            owner,
            cycle: float
    ):
        self.owner = owner
        # set immediately so, the owner knows about the controller
        self.owner.synaptogenesis_controller = self

        self.rng = np.random.default_rng(self.owner.rng.integers(100_000_000))

        stats_update_schedule = int(
            cycle / self.output_sds.sparsity
        )
        self.stats_update_scheduler = Scheduler(stats_update_schedule)
        self.event_prob = np.clip(
            5.0 / cycle, 0.0, 1.0
        )
        self.synaptogenesis_score = np.zeros(self.output_sds.size, dtype=float)

        self.synaptogenesis_cnt = 0
        self.synaptogenesis_cnt_successful = 0

        # stats collection
        self.health_check_results = {}
        self.slow_health_check_results = {}

    def try_activate_synaptogenesis(self, learn, matched_input_activity):
        if not learn:
            return

        self.try_activate_synaptogenesis_step(matched_input_activity)
        if self.stats_update_scheduler.tick():
            self.recalculate_synaptogenesis_score()

    def try_activate_synaptogenesis_step(self, matched_input_activity):
        avg_winner_recognition_str = self.slow_recognition_strength_trace.get()
        winners = self.winners.sdr
        if len(winners) > 0:
            winner_recognition_str = self.potentials[winners].mean()
        else:
            winner_recognition_str = avg_winner_recognition_str

        winner_recognition_str = max(winner_recognition_str, 1e-6)
        beta = avg_winner_recognition_str / winner_recognition_str
        sng_prob = min(1.0, beta * self.synaptogenesis_event_prob)

        if self.rng.random() >= sng_prob:
            return

        # select the winner for the synaptogenesis
        ix_enabled = np.flatnonzero(self.synaptogenesis_score)
        if len(ix_enabled) == 0:
            return

        # noisy_potentials = np.abs(self.rng.normal(
        #     loc=self.potentials[ix_enabled],
        #     scale=self.synaptogenesis_score[ix_enabled]
        # ))

        pot_logits = matched_input_activity.sum(-1)
        logits = pot_logits / (1e-5 + pot_logits.max())

        # _ix_enabled = ix_enabled[
        #     logits[ix_enabled] * self.synaptogenesis_score[ix_enabled] > 0.05
        # ]
        # if len(_ix_enabled) > 0:
        #     ix_enabled = _ix_enabled

        noisy_potentials = self.rng.normal(
            loc=logits[ix_enabled]**0.5,
            scale=self.synaptogenesis_score[ix_enabled] + 1e-6
        )
        sng_winner = ix_enabled[np.argmax(noisy_potentials)]

        success = self.activate_synaptogenesis_step(sng_winner)
        self.synaptogenesis_score[sng_winner] = 0.
        self.synaptogenesis_cnt += 1
        self.synaptogenesis_cnt_successful += int(success)

    def recalculate_synaptogenesis_score(self):
        # NB: usually we work in log-space => log_ prefix is mostly omit for vars
        self.health_check()

        self.synaptogenesis_score[:] = 0.

        # TODO: optimize thresholds and power scaling;
        #   also remember cumulative effect of prob during the cycle for frequent neurons

        # Step 1: deal with absolute rates: sleepy input RF and output
        # TODO: consider increasing random potential part for them
        abs_rate_low, abs_rate_high = np.log(20), np.log(100)
        self.apply_synaptogenesis_to_metric(
            self.slow_health_check_results['ln(nrfe_in)'],
            abs_rate_low, abs_rate_high,
            prob_power=2.0,
            apply_random_synaptogenesis=True
        )

        # Step 2: deal with sleepy output
        eff_low, eff_high = np.log(1.5), np.log(20)
        self.apply_synaptogenesis_to_metric(
            self.health_check_results['ln(nrfe_out)'],
            eff_low, eff_high,
            prob_power=1.5,
        )

        if np.allclose(self.synaptogenesis_score, 0.):
            self.synaptogenesis_score[:] = np.clip(
                self.rng.normal(-0.02, 0.01, size=self.owner.output_size),
                0., 1.0,
            )

        print(
            f'+ {self.owner.output_entropy():.3f}'
            # f' | {self.recognition_strength:.1f}'
            f' | {self.health_check_results["avg(rfe_out)"]:.3f}'
            f' || {self.synaptogenesis_cnt}'
            f' | {safe_divide(self.synaptogenesis_cnt_successful, self.synaptogenesis_cnt):.2f}'
            f' | {np.sum(self.synaptogenesis_score):.2f}'
            # f' || {self.activation_threshold:.2f}'
            # f' | {self.fast_output_sdr_size_trace.get():.1f}'
            # f' | {self.slow_output_size_trace.get():.2f}'
        )

        self.synaptogenesis_cnt = 0
        self.synaptogenesis_cnt_successful = 0

    def apply_synaptogenesis_to_metric(
            self, metric: npt.NDArray[float], low: float, high: float,
            prob_power: float = 1.5,
            apply_random_synaptogenesis: bool = False
    ):
        probs = (-metric - low) / (high - low)
        np.clip(probs, 0., 1., out=probs)
        np.power(probs, prob_power, out=probs)

        neurons = np.flatnonzero(probs > 1e-3)

        self.synaptogenesis_score[neurons] = np.maximum(
            self.synaptogenesis_score[neurons], probs[neurons]
        )

    def activate_synaptogenesis_step(self, neuron):
        in_set = set(self.sparse_input)
        to_choose_from = list(in_set - set(self.rf[neuron]))
        if not to_choose_from:
            return False

        syn = self.rng.choice(to_choose_from)
        to_change_ix = np.argmin(self.weights[neuron])
        self.assign_new_synapses(neuron, to_change_ix, syn)
        return True

    def assign_new_synapses(
            self, neurons: npt.NDArray[int] | int,
            to_change_ix: npt.NDArray[int] | int,
            new_synapses: npt.NDArray[int] | int
    ):
        self.rf[neurons, to_change_ix] = new_synapses
        self.weights[neurons, to_change_ix] = 1.0 / self.rf_size
        self.weights[neurons] = normalize_weights(self.weights[neurons])

    def health_check(self):
        backend: SpatialEncoderSparseBackend = self.owner.weights_backend

        # current Input Rate
        in_rate = self.owner.fast_feedforward_trace.get()
        # Target Input Rate: average rate of each presynaptic neuron
        target_in_rate = in_rate.sum() / self.owner.ff_size

        # NB: Most of the following metrics are relative to some target metric
        # NB2: log-space is used to make the metrics more linear (easier plots and separation)

        # IP (Input Popularity): the relative frequency of each presynaptic neuron
        ip = in_rate / target_in_rate
        log_ip = np.log(ip)

        # RFE^in (Receptive Field Efficiency for matching input):
        # how well each neuron's RF tuned to the input's distribution
        rfe_in = _get_rfe_in(
            n_pre=self.owner.ff_size, n_post=self.owner.output_size,
            ixs_srt_j=backend.ixs_srt_j, weights=backend.weights,
            shifts_j=backend.shifts_j, ip=ip
        )
        avg_rfe_in = rfe_in.mean()
        # NRFE^in (Normalized RFE^in): RFE^in relative to its average
        nrfe_in = rfe_in / avg_rfe_in
        log_nrfe_in = np.log(nrfe_in)

        # current Output Rate
        # FIXME: should be fast_
        out_rate = self.owner.slow_output_trace.get()
        # Target Output Rate: average rate of each postsynaptic neuron
        # NB: for binary output = sparsity, but for rate output this does not hold
        target_out_rate = out_rate.sum() / self.owner.output_size

        # OP (Output Popularity): the relative frequency of each postsynaptic neuron
        op = out_rate / target_out_rate
        log_op = np.log(op)

        # RFE^out (Receptive Field Efficiency for activating neuron):
        # how well each neuron's RF tuning translates to the neuron's activation
        rfe_out = op / rfe_in
        avg_rfe_out = rfe_out.mean()
        # NRFE^out (Normalized RFE^out): RFE^out relative to its average
        nrfe_out = rfe_out / avg_rfe_out
        log_nrfe_out = np.log(nrfe_out)

        self.health_check_results = {
            'ln(ip)': log_ip,
            'ln(op)': log_op,
            'ln(nrfe_in)': log_nrfe_in,
            'avg(rfe_out)': avg_rfe_out,
            'ln(nrfe_out)': log_nrfe_out,
        }

        # NB: for synaptogenesis it is better to track both fast and slow pacing stats

        # FIXME: should be slow_
        in_rate = self.owner.fast_feedforward_trace.get()
        target_in_rate = in_rate.sum() / self.owner.ff_size

        # relative to target input rate
        ip = in_rate / target_in_rate

        # relative to target input rate
        rfe_in = _get_rfe_in(
            n_pre=self.owner.ff_size, n_post=self.owner.output_size,
            ixs_srt_j=backend.ixs_srt_j, weights=backend.weights,
            shifts_j=backend.shifts_j, ip=ip
        )
        avg_rfe_in = rfe_in.mean()
        nrfe_in = rfe_in / avg_rfe_in
        log_nrfe_in = np.log(nrfe_in)

        self.slow_health_check_results = {
            'ln(ip)': log_ip,
            'ln(nrfe_in)': log_nrfe_in,
        }

    def get_health_check_stats(self, ff_trace, out_trace):
        # NB: the following metrics are from the health_check method + some additional
        # helpful metrics for analysis

        in_rate = ff_trace.get()
        target_in_rate = in_rate.sum() / self.owner.ff_size

        ip = in_rate / target_in_rate
        log_ip = np.log(ip)

        rfe_in = np.sum(ip[self.rf] * self.weights, axis=1)
        avg_rfe_in = rfe_in.mean()
        nrfe_in = rfe_in / avg_rfe_in
        log_nrfe_in = np.log(nrfe_in)

        # RFP^in (Receptive Field Popularity):
        # NB: compared to RFE^in, it uses the "uniformly distributed weights" instead of
        # real weights.
        # NB2: alone, avg RFP^in shows how much neurons connect to the strong (=frequent) input.
        # Ideally, this value should NOT be very high, because this means that the neurons
        # are not selective enough, and just connect to the most frequent input.
        rfp_in = np.mean(ip[self.rf], axis=1)
        avg_rfp_in = rfp_in.mean()
        log_rfp_in = np.log(rfp_in)

        # LP (Learning Potential):
        # how well each neuron's RF weights are tuned relative to "un-tuned" uniform weights
        # NB: if below 1, then the neuron's learning performs worse than no learning.
        lp = rfe_in / rfp_in
        avg_lp = lp.mean()
        log_lp = np.log(lp)

        out_rate = out_trace.get()
        target_out_rate = out_rate.sum() / self.owner.output_size

        op = out_rate / target_out_rate
        log_op = np.log(op)

        rfe_out = op / rfe_in
        avg_rfe_out = rfe_out.mean()
        nrfe_out = rfe_out / avg_rfe_out
        log_nrfe_out = np.log(nrfe_out)
        # NB: very rough approximate of the STD[log(nrfe_out)] — enough to see dynamics and scale
        appx_std_log_nrfe_out = np.mean(np.abs(log_nrfe_out))

        # for plotting, clip values to avoid uninformative plot scales
        min_ln = np.log(1/10)
        return {
            'ln(ip)': np.maximum(log_ip, min_ln),
            'ln(op)': np.maximum(log_op, min_ln),

            'avg(rfe_in)': avg_rfe_in,
            'ln(nrfe_in)': np.maximum(log_nrfe_in, min_ln),

            'avg(rfp_in)': avg_rfp_in,
            'ln(rfp_in)': log_rfp_in,

            'avg(lp)': avg_lp,
            'ln(lp)': log_lp,

            'avg(rfe_out)': avg_rfe_out,
            'ln(nrfe_out)': np.maximum(log_nrfe_out, min_ln),
            'std(ln(nrfe_out))': appx_std_log_nrfe_out,
        }

    @property
    def feedforward_sds(self) -> Sds:
        return self.owner.feedforward_sds

    @property
    def output_sds(self) -> Sds:
        return self.owner.output_sds


@jit()
def _get_rfe_in(n_pre, n_post, ixs_srt_j, weights, shifts_j, ip):
    # calculates equivalent of `dot(ip[rf], w)` for each neuron `i`
    rfe_in = np.zeros(n_post, float)
    k_min = 0
    for j in range(n_pre):
        # [k_min:k_max] define kxs in `ixs_srt_j` and `weights` corresponding to the current `j`
        k_max = shifts_j[j+1]
        # `ixs_srt_j[k_min:k_max]`: ixs that have synapses with `j`
        # `weights[k_min:k_max]`: their corresponding synaptic weights
        rfe_in[ixs_srt_j[k_min:k_max]] += ip[j] * weights[k_min:k_max]
        k_min = k_max

    return rfe_in