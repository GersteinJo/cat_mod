from cat_mod.models.representations.DIM import Encoder as DIMEncoder
from cat_mod.models.representations.ConvVAE import ConvVAE
from cat_mod.models.representations.CNNEncoder import CNNEncoder
from cat_mod.models.representations.spatial_pooler.se import SpatialEncoderLayer
from cat_mod.models.representations.spatial_pooler.sdr import RateSdr
from cat_mod.models.representations.spatial_pooler.sds import Sds

import torch
import numpy as np


class BaseEncoder:
    def encode(self, obs: torch.Tensor)->np.ndarray:
        raise NotImplemented


class DIM(BaseEncoder):
    def __init__(self, pretrained_weights_path=None, **kwargs):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = DIMEncoder().to(self.device)
        if pretrained_weights_path:
            self.model.load_state_dict(
                torch.load(pretrained_weights_path, map_location=self.device)
            )
            print('Pretrained weights loaded.')
        self.model.eval()

    def encode(self, obs):
        obs = obs.to(self.device)
        with torch.no_grad():
            obs = self.model(obs[None])[0][0]
        obs = obs.detach().cpu().numpy()
        return obs


class VAE(BaseEncoder):
    def __init__(self, latent_dim, pretrained_weights_path=None, **kwargs):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = ConvVAE(latent_dim=latent_dim).to(self.device)
        if pretrained_weights_path:
            self.model.load_state_dict(
                torch.load(pretrained_weights_path, map_location=self.device)
            )
            print('Pretrained weights loaded.')
        self.model.eval()

    def encode(self, obs):
        obs = obs.to(self.device)
        with torch.no_grad():
            obs = self.model(obs[None])[0][0]
        obs = obs.detach().cpu().numpy()
        return obs


class CNN(BaseEncoder):
    def __init__(self, pretrained_weights_path=None, **kwargs):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = CNNEncoder().to(self.device)
        if pretrained_weights_path:
            self.model.load_state_dict(
                torch.load(pretrained_weights_path, map_location=self.device)
            )
            print('Pretrained weights loaded.')
        self.model.eval()

    def encode(self, obs):
        obs = obs.to(self.device)
        with torch.no_grad():
            obs = self.model(obs[None])[0][0]
        obs = obs.detach().cpu().numpy()
        return obs


class SE(BaseEncoder):
    def __init__(self, pretrained_weights=None, **config):
        encoding_sds = Sds.make(config.pop('encoding_sds'))
        self.model = SpatialEncoderLayer(feedforward_sds=Sds((32*32*3, 3064)), output_sds=encoding_sds, **config)
        if pretrained_weights:
            weights = np.load(pretrained_weights)
            self.model.weights_backend.weights = weights
            print('Pretrained weights loaded.')

    def encode(self, obs: torch.Tensor) ->np.ndarray:
        dense = obs.numpy().flatten()
        threshold = 1/255.0
        sparse = np.flatnonzero(dense >= threshold)
        dense[dense < threshold] = 0.0
        rate_sdr = RateSdr(sparse, values=dense[sparse]/dense[sparse].sum())
        emb = self.model.compute(rate_sdr)
        return emb.to_dense(self.model.output_sds.size)
