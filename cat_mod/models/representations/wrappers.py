from cat_mod.models.representations.DIM import Encoder as DIMEncoder
import torch


class BaseEncoder:
    def encode(self, obs):
        raise NotImplemented


class DIM(BaseEncoder):
    def __init__(self, pretrained_weights=None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = DIMEncoder().to(self.device)
        self.model.eval()

    def encode(self, obs):
        obs = obs.to(self.device)
        with torch.no_grad():
            obs = self.model(obs[None])[0]
        obs = obs.detach().cpu().numpy()
        return obs
