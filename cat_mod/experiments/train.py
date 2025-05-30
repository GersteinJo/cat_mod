from cat_mod.models.representations.spatial_pooler.se import SpatialEncoderLayer
from cat_mod.models.representations.spatial_pooler.dataset import MnistDataset
from cat_mod.models.representations.spatial_pooler.sds import Sds
import numpy as np
from tqdm import tqdm
import yaml
import wandb

def read_config(path):
    with open(path, 'r') as file:
        conf = yaml.load(file, yaml.Loader)
    return conf


def split_to_batches(order, batch_size):
    n_samples = len(order)
    return np.array_split(order, n_samples // batch_size)


def train(conf, logger=None):
    data = MnistDataset(**conf['dataset'])
    dataset_sds = data.sds
    encoding_sds = Sds.make(conf['se'].pop('encoding_sds'))
    encoder = SpatialEncoderLayer(feedforward_sds=dataset_sds, output_sds=encoding_sds, **conf['se'])
    rng = np.random.default_rng(seed)

    for epoch in range(conf['epochs']):
        n_samples = len(data.train)
        order = rng.permutation(n_samples)
        sdrs = data.train.sdrs
        batched_indices = split_to_batches(order, conf['batch_size'])
        for batch_ixs in tqdm(batched_indices):
            batch_sdrs = sdrs.create_slice(batch_ixs)
            encoder.compute_batch(batch_sdrs, learn=True)
            if logger:
                logger.log(
                    {
                        'entropy': encoder.output_entropy(),
                        'radius': encoder.radius.mean()
                    }
                )

    np.save(f'weights/se_cifar.npy', encoder.weights_backend.weights)


if __name__ == '__main__':
    conf_path = 'configs/se_train.yaml'
    conf = read_config(conf_path)

    seed = conf.get('seed', None)
    if seed is None:
        seed = np.random.randint(0, np.iinfo(np.int32).max)
        conf['seed'] = seed

    conf['se'] = read_config(conf['se_config'])
    conf['se']['seed'] = seed
    conf['dataset']['seed'] = seed

    log = conf.pop('log')
    if log:
        logger = wandb.init(
            project=conf.pop('project_name'),
            config=conf
        )
    else:
        logger = None

    train(conf, logger)

