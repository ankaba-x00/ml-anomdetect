import torch
import torch.nn as nn
import torch.nn.functional as F

from .configs import MTVAEConfig
from .base import BaseTabularPredictor


class MTVEncoder():
    pass


class MTVDecoder():
    pass


class MTTabularVAE(BaseTabularPredictor):
    """
    Hybrid tabular variational autoencoder with:
      - cont and cat inputs
      - learned cat embeddings
      - optional denoising for cont features
      - L3/L7 regression heads and attack type classification head 
      - KL + ELBO reconstruction
      - metrics: xx (for L3/L7) and xx (for attack type)
    """
    pass