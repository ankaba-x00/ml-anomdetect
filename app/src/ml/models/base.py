from abc import ABC, abstractmethod
from typing import Sequence
import torch
import torch.nn as nn


class BaseTabularEncoder(ABC, nn.Module):
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
        self.activation = _pick_act_func(activation)

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
            _init_weights(self.embeddings)
        _init_weights(self.encoder_layers, activation)
        self.init_comp_heads()

    @abstractmethod
    def make_comp_heads(self) -> None:
        """Adds final compression heads of encoder."""
        pass
    
    @abstractmethod
    def init_comp_heads(self) -> None:
        """Initializes final compression heads of encoder."""
        pass


class BaseTabularDecoder(ABC, nn.Module):
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
        self.activation = _pick_act_func(activation)
    
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
        _init_weights(self.decoder_layers, activation)
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


def _pick_act_func(act_name: str) -> nn.Module:
    """Selects activation function from a predifined collection."""

    act_dict = {
        "relu": nn.ReLU(inplace=True),
        "leaky_relu": nn.LeakyReLU(0.01, inplace=True),
        "gelu": nn.GELU(),
        "elu": nn.ELU(inplace=True),
        "silu": nn.SiLU(inplace=True),
        "tanh": nn.Tanh(),
        "sigmoid": nn.Sigmoid(),
    }

    if act_name not in act_dict:
        raise ValueError(f"[ERROR] Unknown activation: {act_name}.")
    
    return act_dict[act_name]

def _pick_weight_init_func(act_name: str, m: nn.Module) -> None:
    """Selects init function for weights depending on activation function type."""

    init_dict = {
        "he_normal": lambda layer: nn.init.kaiming_normal_(layer.weight),
        "he_uniform": lambda layer: nn.init.kaiming_uniform_(layer.weight, nonlinearity="relu"),
        "xavier_normal": lambda layer: nn.init.xavier_normal_(layer.weight),
        "xavier_uniform": lambda layer: nn.init.xavier_uniform_(layer.weight, gain=0.01),
        "zeros": lambda layer: nn.init.zeros_(layer.weight)
    }

    if act_name in ["relu", "leaky_relu", "gelu", "elu", "silu"]:
        init_dict["he_uniform"](m)
    elif act_name in ["tanh", "sigmoid"]:
        init_dict["xavier_normal"](m)
    elif act_name is None:
        init_dict["zeros"](m)
    else:
        raise ValueError(f"[ERROR] Unknown activation for weight init: {act_name}.")

def _pick_bias_init_func(init_name: str, m: nn.Module) -> None:
    """Selects init function for bias."""
    
    init_dict = {
        "zeros": lambda layer: nn.init.zeros_(layer.bias),
        "orthogonal": lambda layer: nn.init.orthogonal_(layer.bias)
    }
    init_dict[init_name](m)

def _init_weights(modules: nn.ModuleList | nn.ModuleDict, act_name: str | None = None) -> None:
    """
    Initializes weights of model layers.
    Input tensor modification depends on layer connectivity and activation function in use.
    """

    # encoder_layers / decoder_layers 
    if isinstance(modules, nn.ModuleList):
        for m in modules.modules():
            if isinstance(m, nn.Linear):
                _pick_weight_init_func(act_name, m)
                # Does Pytorch automatically for BatchNorm1d, deselect if bias is set to True for other layers
                # if m.bias is not None:
                #     _pick_bias_init_func("zeros", m)
    
    # embeddings
    elif isinstance(modules, nn.ModuleDict):
        for m in modules.children():
            nn.init.normal_(m.weight, mean=0.0, std=0.01)

def _init_head(module: nn.Linear, act_name: str) -> None:
    """
    Initializes weights and biases of final compression and reconstruction heads.
    """

    _pick_weight_init_func(act_name, module)
    _pick_bias_init_func("zeros", module)