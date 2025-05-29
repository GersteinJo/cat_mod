import numpy as np
UINT_DTYPE = np.uint32


def sample_categorical_variables(probs, rng: np.random.Generator):
    assert np.allclose(probs.sum(axis=-1), 1)

    gammas = rng.uniform(size=probs.shape[0]).reshape((-1, 1))

    dist = np.cumsum(probs, axis=-1)
    dist[:, -1] = 1.0

    ubounds = dist
    lbounds = np.zeros_like(dist)
    lbounds[:, 1:] = dist[:, :-1]

    cond = (gammas >= lbounds) & (gammas < ubounds)

    states = np.zeros_like(probs) + np.arange(probs.shape[1])

    samples = states[cond].astype(UINT_DTYPE)

    return samples