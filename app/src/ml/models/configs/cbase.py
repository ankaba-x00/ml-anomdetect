import inspect, json
from dataclasses import asdict, dataclass, field
from typing import Any, Sequence, TextIO


@dataclass
class BaseConfig:
    """
    Represents a base config object for TabularBase.
    """

    num_cont: int
    cat_dims: dict[str, int]
    depth: int
    base_dim: int
    hidden_dims: Sequence[int] = field(init=False)
    use_embedding: bool
    embedding_dim: int | None
    latent_dim: int
    activation_en: str
    activation_de: str
    dropout: float

    def __post_init__(self) -> None:
        self.hidden_dims = [max(32, int(self.base_dim / (2**i))) for i in range(self.depth)]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self, file: TextIO) -> None:
        payload = self.to_dict()
        del payload["hidden_dims"]
        json.dump(payload, file, indent=2)
    
    def get_args(self) -> set[str]:
        sig = inspect.signature(self.__class__)
    
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


@dataclass
class SharedEncoderConfig(BaseConfig):
    ...


@dataclass
class SharedVEncoderConfig(BaseConfig):
    ...


@dataclass
class SharedDecoderConfig(BaseConfig):
    temperature: float


@dataclass
class SharedMTDecoderConfig(BaseConfig):
    n_attack_types: int
    activation_de_cls: str
    quantiles: Sequence[float]
    temperature: float
