import torch
from abc import ABC, abstractmethod
import torch.nn as nn
from typing import Any, Generic, TypeVar

from .configs.cbase import BaseConfig, SharedEncoderConfig, SharedVEncoderConfig, SharedDecoderConfig, SharedMTDecoderConfig
from .mixins import TabularLayerActMixin, TabularLayerInitMixin


EncoderConfigT = TypeVar("EncoderConfigT", bound=BaseConfig)

class BaseTabularEncoder(ABC, Generic[EncoderConfigT], nn.Module, TabularLayerActMixin, TabularLayerInitMixin):
    """
    Base encoder class for tabular autoencoder models.

    Tasks incl. setting up embeddings, encoder layers, as well as weight, bias and embedding initialization.

    Subclasses may override make_comp_heads() and init_comp_heads() depending on comp head composition.
    """

    def __init__(
        self,
        config: EncoderConfigT
    ) -> None:
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
                dim = config.embedding_dim if config.embedding_dim else emb_dim(card)
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


SharedEncoderConfigT = TypeVar(
    "SharedEncoderConfigT", bound=SharedEncoderConfig
)

class SharedEncoder(BaseTabularEncoder[SharedEncoderConfigT]):
    def __init__(self, config: SharedEncoderConfigT) -> None:
        super().__init__(config)


SharedVEncoderConfigT = TypeVar(
    "SharedVEncoderConfigT", bound=SharedVEncoderConfig
)

class SharedVEncoder(BaseTabularEncoder[SharedVEncoderConfigT]):
    def __init__(self, config: SharedVEncoderConfigT) -> None:
        super().__init__(config)


DecoderConfigT = TypeVar("DecoderConfigT", bound=BaseConfig)

class BaseTabularDecoder(ABC, Generic[DecoderConfigT], nn.Module, TabularLayerActMixin, TabularLayerInitMixin):
    """
    Base decoder class for tabular autoencoder models.

    Tasks incl. setting up decoder layers, as well as weight initialization.

    Subclasses may override make_recon_heads() and init_recon_heads() depending on recon head schema.
    """
    
    def __init__(
        self,
        config: DecoderConfigT
    ) -> None:
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


SharedDecoderConfigT = TypeVar(
    "SharedDecoderConfigT", bound=SharedDecoderConfig
)

class SharedDecoder(BaseTabularDecoder[SharedDecoderConfigT]):
    def __init__(self, config: SharedDecoderConfigT) -> None:
        super().__init__(config)


SharedMTDecoderConfigT = TypeVar(
    "SharedMTDecoderConfigT", bound=SharedMTDecoderConfig
)

class SharedMTDecoder(BaseTabularDecoder[SharedMTDecoderConfigT]):
    def __init__(self, config: SharedMTDecoderConfigT) -> None:
        super().__init__(config)


ConfigT = TypeVar("ConfigT", bound=BaseConfig)

class TabularBase(ABC, Generic[ConfigT], nn.Module):
    """
    Base class for tabular autoencoder models.

    Tasks incl. setting up encoding, decoding, forward passing and scoring.

    Subclasses override encode(), decode(), forward(), scoring().
    """

    def __init__(
        self,
        config: ConfigT
    ) -> None:
        super().__init__()
        self.config = config

    @abstractmethod
    def encode(self, x_cont: torch.Tensor, x_cat: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor | tuple[torch.Tensor, ...]:
        """Return latent variables."""
        pass

    @abstractmethod
    def decode(self, z: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]] | tuple[torch.Tensor, dict[str, torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return cont and cat reconstruction."""
        pass

    @abstractmethod
    def forward(self, *args: Any, **kwargs: Any) -> Any:
        """Full forward pass through autoencoder."""
        
    @abstractmethod
    def scoring(self, *args: Any, **kwargs: Any) -> tuple[torch.Tensor, ...] | torch.Tensor:
        """Computes reconstruction error."""
        pass
