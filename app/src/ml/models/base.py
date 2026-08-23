from abc import ABC, abstractmethod
import torch.nn as nn

from .configs import AEConfig, VAEConfig, MTAEConfig, MTVAEConfig
from .mixins import TabularLayerActMixin, TabularLayerInitMixin


class BaseTabularEncoder(ABC, nn.Module, TabularLayerActMixin, TabularLayerInitMixin):
    """
    Base encoder class for tabular autoencoder models.

    Tasks incl. setting up embeddings, encoder layers, as well as weight, bias and embedding initialization.

    Subclasses may override make_comp_heads() and init_comp_heads() depending on comp head composition.
    """

    def __init__(
        self,
        config: AEConfig | VAEConfig | MTAEConfig | MTVAEConfig
    ):
        super().__init__()
        
        self.config = config
        self.activation = self.pick_act_func(config.activation_en)

        # -----------------------------
        # Categorical embeddings
        # -----------------------------
        if config.use_embedding:
            self.embeddings = nn.ModuleDict()
            emb_sizes = {}

            def emb_dim(card: int) -> int:
                """Determines embedding_dim from cardinality of cat features."""
                return min(max(4, card // 2), 16)

            for name, card in config.cat_dims.items():
                dim = embedding_dim if config.embedding_dim else emb_dim(card)
                self.embeddings[name] = nn.Embedding(card, dim)
                emb_sizes[name] = dim

            emb_total = sum(emb_sizes.values())
            self.input_dim = config.num_cont + emb_total
        else:
            self.input_dim = config.num_cont + sum(config.cat_dims.values())

        # -----------------------------
        # Encoder layers
        # -----------------------------
        self.encoder_layers = nn.ModuleList()
        prev = self.input_dim

        for h in config.hidden_dims:
            self.encoder_layers.append(
                nn.Sequential(
                    nn.Linear(prev, h, bias=False),
                    nn.BatchNorm1d(h),
                    self.activation,
                    nn.Dropout(config.dropout)
                )
            )
            prev = h
        
        self.make_comp_heads()
        
        # -----------------------------
        # Weight initialization
        # -----------------------------
        if config.use_embedding:
            self.init_weights(self.embeddings)
        self.init_weights(self.encoder_layers, config.activation_en)
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
        config: AEConfig | VAEConfig | MTAEConfig | MTVAEConfig
    ):
        super().__init__()

        self.config = config
        self.activation = self.pick_act_func(config.activation_de)
    
        # -----------------------------
        # Decoder layers
        # -----------------------------
        self.decoder_layers = nn.ModuleList()
        prev = config.latent_dim
        
        for h in reversed(config.hidden_dims):
            self.decoder_layers.append(
                nn.Sequential(
                    nn.Linear(prev, h, bias=False),
                    nn.BatchNorm1d(h),
                    self.activation,
                    nn.Dropout(config.dropout)
                )
            )
            prev = h

        self.make_recon_heads()

        # -----------------------------
        # Weight initialization
        # -----------------------------
        self.init_weights(self.decoder_layers, config.activation_de)
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

    Tasks incl. setting up encoding, decoding, forward passing and scoring.

    Subclasses override encode(), decode(), forward(), scoring().
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
        
    @abstractmethod
    def scoring(self) -> None:
        """Computes reconstruction error."""
        pass
