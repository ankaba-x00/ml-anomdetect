import inspect, json, torch
from dataclasses import asdict, dataclass
from typing import Sequence, TextIO


@dataclass
class AEConfig:
    """
    Represents a AE config object for TabularAE.
    """

    num_cont: int
    cat_dims: dict[str, int]
    use_embedding: bool = False
    embedding_dim: int | None = None
    depth: int = 2
    base_dim: int = 256
    hidden_dims: Sequence[int] | None = None
    latent_dim: int = 32
    activation_en: str = "relu"
    activation_de: str = "relu"
    dropout: float = 0.1
    optimizer: str = "adam"
    lr: float = 1e-5
    weight_decay: float = 1e-5
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    sgd_momentum: float = 0.0
    lr_scheduler: str = "none"
    gradient_clip: float | None = None
    batch_size: int = 256
    allow_noise_injection: bool = True
    noise_gauss_std: float = 0.1
    noise_mask_prob: float = 0.05
    num_epochs: int = 60
    warmup_epochs: int = 10
    patience: int = 10
    temperature: float = 1.0  
    cont_w: float = 1.0
    cat_w: float = 0.1
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    def __post_init__(self):
        self.hidden_dims = [max(32, int(self.base_dim / (2**i))) for i in range(self.depth)]

    def to_json(self, file: TextIO) -> str:
        return json.dump(asdict(self), file, indent=2)
    
    def get_args(self) -> set[str]:
        sig = inspect.signature(self.__init__)
    
        return set([
            name for name, param in sig.parameters.items()
            if param.default is param.empty and name != "self"
        ])

    def get_kwargs(self) -> set[str]:
        sig = inspect.signature(self.__class__)
        return set([
            name for name, param in sig.parameters.items()
            if param.default is not param.empty
        ])
    