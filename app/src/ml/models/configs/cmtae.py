import inspect, json, torch
from dataclasses import asdict, dataclass
from typing import Sequence, TextIO

from app.src.data import ATTACK_LABELS


@dataclass
class MTAEConfig:
    """
    Represents a MTAE config object for MTTabularAE.
    """

    num_cont: int
    cat_dims: dict[str, int]
    n_attack_types: int = len(ATTACK_LABELS)
    use_embedding: bool = False
    embedding_dim: int | None = None
    depth: int = 2
    base_dim: int = 256
    hidden_dims: Sequence[int] | None = None
    latent_dim: int = 32
    activation_en: str = "relu"
    activation_de: str = "relu"
    activation_de_cls: str = "tanh"
    dropout: float = 0.1
    optimizer: str = "adam"
    lr: float = 1e-3
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
    patience: int = 30
    temperature: float = 1.0  
    quantiles: Sequence[float] = (0.10, 0.25, 0.5, 0.75, 0.9)
    focal_gamma: float = 2.0
    num_epochs: int = 60
    warmup_epochs: int = 5
    stepup_epochs: int = 5
    alpha: float = 1.0
    cont_w: float = 1.0
    cat_w: float = 0.1
    lambda_l3: float = 1.0
    lambda_l7: float = 1.0
    lambda_at: float = 1.0
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