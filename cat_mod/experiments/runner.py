import wandb
import numpy as np


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

            cls = self.categoriser.predict(obs)
            is_pecked = cls[0] == 1

            if is_pecked:
                peck_counter += 1
                self.categoriser.fit(obs, label)

            correct_peck = int(is_pecked and label['edible'])
            total_correct_pecks += correct_peck
            missed = int((not is_pecked) and label['edible'])
            total_missed += missed

            if logger:
                logger.log(
                    {
                        "peck": int(is_pecked),
                        "correct_peck": correct_peck,
                        "missed": missed,
                        "total_correct_pecks": total_correct_pecks,
                        "total_missed": total_missed,
                        "success_rate": total_correct_pecks / (peck_counter + 1e-24),
                        "total_pecks": peck_counter
                     },
                    step=i
                )
                pecked_objects.add_data(label.values())

        if logger:
            logger.log({'pecked_objects': pecked_objects})
            logger.finish()
