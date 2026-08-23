import inspect, json, torch
from dataclasses import asdict, dataclass
from typing import Sequence, TextIO


@dataclass
class MTVAEConfig:
    """
    Represents a MTVAE config object for MTTabularVAE.
    """

    num_cont: int
    cat_dims: dict[str, int]
    n_attack_types: int
    depth: int = 2
    base_dim: int = 256
    hidden_dims: Sequence[int] | None = None
    latent_dim: int = 32
    quantiles: Sequence[float] = (0.5, 0.9, 0.99)
    dropout: float = 0.1
    activation_en: str = "relu"
    activation_de_reg: str = "relu"
    activation_de_cls: str = "relu"
    head_hidden_dim: int = 32
    lambda_l3: float = 1.0
    lambda_l7: float = 1.0
    lambda_attack: float = 3.0 # maybe 5.0
    use_focal_loss: bool = False
    focal_gamma: float = 2.0 # 0 = standard CE, 2 = common default, 3+ = very aggressive and migth be unstable 
    lr: float = 1e-3
    weight_decay: float = 1e-5
    batch_size: int = 256
    num_epochs: int = 60
    warmup_epochs: int = 5
    patience: int = 6
    gradient_clip: float = 1.0
    use_lr_scheduler: bool = True
    cont_w: float = 1.0
    cat_w: float = 0.1
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    def __post_init__(self):
        self.hidden_dims = [max(32, int(self.base_dim / (2**i))) for i in range(self.depth)]

    def to_json(self, file: TextIO) -> str:
        return json.dump(asdict(self), file, indent=2)
    
    def get_kwargs(self) -> set[str]:
        sig = inspect.signature(self.__class__)
        return set([
            name for name, param in sig.parameters.items()
            if param.default is not param.empty
        ])