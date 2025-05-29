import yaml
import wandb
import numpy as np
from torchvision import transforms
from torch.utils.data import DataLoader

from runner import Runner
from cat_mod.utils.image_dataset import ImageLabelDataset
from cat_mod.models.representations.wrappers import DIM, SE
from cat_mod.models.representations.CNNEncoder import CNNEncoder
from cat_mod.models.representations.ConvVAE import ConvVAE as VAEncoder

ENCODERS = {
    "dim": DIM,
    "cnn": CNNEncoder,
    "vae": VAEncoder,
    "se": SE
}

from cat_mod.models.cat_mod.SEP import SEP
from cat_mod.models.cat_mod.SEP_SOM import SEP_SOM

CATS = {
    "sep": SEP,
    "sep_som": SEP_SOM
}

def read_config(path):
    with open(path, 'r') as file:
        conf = yaml.load(file, yaml.Loader)
    return conf

def setup_encoder(cfg_path, seed):
    cfg = read_config(cfg_path)
    cfg['seed'] = seed
    type_ = cfg_path.split('/')[-2]
    enc = ENCODERS[type_](**cfg)
    return enc, type_, cfg

def setup_categoriser(cfg_path, seed):
    cfg = read_config(cfg_path)
    cfg['seed'] = seed
    type_ = cfg_path.split('/')[-2]
    enc = CATS[type_](**cfg)
    return enc, type_, cfg


def setup_dataset(conf):
    dataset = ImageLabelDataset(
        csv_file=conf['info'],
        img_dir=conf['dir'],
        transform=transforms.ToTensor()  # Add any other transforms
    )
    return dataset


if __name__ == '__main__':
    # read config
    conf_path = 'configs/example_config.yaml'
    conf = read_config(conf_path)
    seed = conf.get('seed', None)
    if seed is None:
        seed = np.random.randint(0, np.iinfo(np.int32).max)
        conf['seed'] = seed

    # determine encoder and categoriser
    encoder, enc_type, enc_conf = setup_encoder(conf['encoder_conf'], seed)
    conf['encoder_type'] = enc_type
    conf['encoder_conf'] = enc_conf
    categoriser, cat_type, cat_conf = setup_categoriser(conf['categoriser_conf'], seed)
    conf['categoriser_type'] = enc_type
    conf['categoriser_conf'] = enc_conf

    conf['dataset']['seed'] = seed
    dataset = setup_dataset(conf['dataset'])

    log = conf.pop('log')
    if log:
        logger = wandb.init(
            project=conf.pop('project_name'),
            name=conf.pop('run_name'),
            config=conf,
            tags=[conf['categoriser_type'], conf['encoder_type']]
        )
    else:
        logger = None

    runner = Runner(
        encoder=encoder,
        categoriser=categoriser,
        dataset=dataset,
    )
    runner.run(conf['n_iter'], logger)
