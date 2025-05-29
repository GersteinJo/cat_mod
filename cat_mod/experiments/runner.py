import wandb
import numpy as np
from scipy.stats import entropy


class Runner:
    def __init__(
        self,
        dataset,
        categoriser,
        encoder=None,
        seed=None
    ):
        self.encoder = encoder
        self.categoriser = categoriser
        self.dataset = dataset
        self.seed = seed
        self._rng = np.random.default_rng(seed)

    def run(self, n_iter, logger=None):
        peck_counter = 0
        pecked_objects = wandb.Table(
            columns=list(self.dataset[0][-1].keys())
        )
        n_iter = min(n_iter, len(self.dataset))
        indices = np.arange(len(self.dataset))
        self._rng.shuffle(indices)

        total_correct_pecks = 0
        total_missed = 0

        for i in range(n_iter):
            obs, label = self.dataset[indices[i]]
            if self.encoder is not None:
                obs = self.encoder.encode(obs)

            cls, probs = self.categoriser.predict(obs[None])
            is_pecked = cls[0] == 1

            if is_pecked:
                peck_counter += 1
                self.categoriser.fit(obs[None], np.array([int(label['edible'])]))

            reward = int(is_pecked and label['edible'])
            total_correct_pecks += reward
            missed = int((not is_pecked) and label['edible'])
            total_missed += missed

            if logger:
                logger.log(
                    {
                        "peck": int(is_pecked),
                        "reward": reward,
                        "missed": missed,
                        "total_correct_pecks": total_correct_pecks,
                        "total_missed": total_missed,
                        "success_rate": total_correct_pecks / (peck_counter + 1e-24),
                        "total_pecks": peck_counter,
                        "prediction_entropy": entropy(probs[0])
                     },
                    step=i
                )
                if is_pecked:
                    pecked_objects.add_data(*list(label.values()))

        if logger:
            logger.log({'pecked_objects': pecked_objects})
            logger.finish()
