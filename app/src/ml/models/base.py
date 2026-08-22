from abc import ABC, abstractmethod
from typing import Sequence
import torch.nn as nn

from .mixins import TabularLayerActMixin, TabularLayerInitMixin

class BaseTabularEncoder(ABC, nn.Module, TabularLayerActMixin, TabularLayerInitMixin):
    """
    Base encoder class for tabular autoencoder models.

    Tasks incl. setting up embeddings, encoder layers, as well as weight, bias and embedding initialization.

    Subclasses may override make_comp_heads() and init_comp_heads() depending on comp head composition.
    """

    def __init__(
        self,
        num_cont: int,
        cat_dims: dict[str, int],
        hidden_dims: Sequence[int] = (128, 64),
        latent_dim: int = 32,
        use_embedding: bool = False,
        embedding_dim: int | None = None,
        dropout: float = 0.1,
        activation: str = "relu"
    ):
        super().__init__()
        
        self.hidden_dims = hidden_dims
        self.latent_dim = latent_dim
        self.act_name = activation
        self.activation = self._pick_act_func(activation)

        # -----------------------------
        # Categorical embeddings
        # -----------------------------
        if use_embedding:
            self.embeddings = nn.ModuleDict()
            emb_sizes = {}

            def emb_dim(card: int) -> int:
                """Determines embedding_dim from cardinality of cat features."""
                return min(max(4, card // 2), 16)

            for name, card in cat_dims.items():
                dim = embedding_dim if embedding_dim else emb_dim(card)
                self.embeddings[name] = nn.Embedding(card, dim)
                emb_sizes[name] = dim

            emb_total = sum(emb_sizes.values())
            self.input_dim = num_cont + emb_total
        else:
            self.input_dim = num_cont + sum(cat_dims.values())

        # -----------------------------
        # Encoder layers
        # -----------------------------
        self.encoder_layers = nn.ModuleList()
        prev = self.input_dim

        for h in hidden_dims:
            self.encoder_layers.append(
                nn.Sequential(
                    nn.Linear(prev, h, bias=False),
                    nn.BatchNorm1d(h),
                    self.activation,
                    nn.Dropout(dropout)
                )
            )
            prev = h
        
        self.make_comp_heads()
        
        # -----------------------------
        # Weight initialization
        # -----------------------------
        if use_embedding:
            self.init_weights(self.embeddings)
        self.init_weights(self.encoder_layers, activation)
        self.init_comp_heads()

    @abstractmethod
    def make_comp_heads(self) -> None:
        """Adds final compression heads of encoder."""
        pass
    
    @abstractmethod
    def init_comp_heads(self) -> None:
        """Initializes final compression heads of encoder."""
        pass


class BaseTabularDecoder(ABC, nn.Module, TabularLayerActMixin, TabularLayerInitMixin):
    """
    Base decoder class for tabular autoencoder models.

    Tasks incl. setting up decoder layers, as well as weight initialization.

    Subclasses may override make_recon_heads() and init_recon_heads() depending on recon head schema.
    """
    
    def __init__(
        self,
        num_cont: int,
        cat_dims: dict[str, int],
        hidden_dims: Sequence[int] = (128, 64),
        latent_dim: int = 32,
        dropout: float = 0.1,
        activation: str = "relu",
    ):
        super().__init__()

        self.num_cont = num_cont
        self.cat_dims = cat_dims
        self.hidden_dims = hidden_dims
        self.act_name = activation
        self.activation = self._pick_act_func(activation)
    
        # -----------------------------
        # Decoder layers
        # -----------------------------
        self.decoder_layers = nn.ModuleList()
        prev = latent_dim
        
        for h in reversed(hidden_dims):
            self.decoder_layers.append(
                nn.Sequential(
                    nn.Linear(prev, h, bias=False),
                    nn.BatchNorm1d(h),
                    self.activation,
                    nn.Dropout(dropout)
                )
            )
            prev = h

        self.make_recon_heads()

        # -----------------------------
        # Weight initialization
        # -----------------------------
        self.init_weights(self.decoder_layers, activation)
        self.init_recon_heads()

    @abstractmethod
    def make_recon_heads(self) -> None:
        """Adds final reconstruction heads of decoder."""
        pass
    
    @abstractmethod
    def init_recon_heads(self) -> None:
        """Initializes final reconstruction heads of decoder."""
        pass


class BaseTabularPredictor(ABC, nn.Module):
    """
    Base class for tabular autoencoder models.

    Tasks incl. setting up encoding, decoding, forward passing.

    Subclasses override encode(), decode(), forward().
    """

    def __init__(self):
        super().__init__()

    @abstractmethod
    def encode(self) -> None:
        """Return latent variables."""
        pass

    @abstractmethod
    def decode(self) -> None:
        """Return cont and cat reconstruction."""
        pass

    @abstractmethod
    def forward(self) -> None:
        """Full forward pass through autoencoder."""
        pass
