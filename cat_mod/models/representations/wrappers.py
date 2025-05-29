from cat_mod.models.representations.DIM import Encoder as DIMEncoder
import torch
import numpy as np


class BaseEncoder:
    def encode(self, obs: torch.Tensor)->np.ndarray:
        raise NotImplemented


class DIM(BaseEncoder):
    def __init__(self, pretrained_weights_path=None):
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
