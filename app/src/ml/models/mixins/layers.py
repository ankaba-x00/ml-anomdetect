import torch.nn as nn


class TabularLayerActMixin:
    """
    Stateless Helper Mixin providing functionality for activating autoencoder layers.
    """

    @staticmethod
    def pick_act_func(act_name: str) -> nn.Module:
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


class TabularLayerInitMixin:
    """
    Stateless Helper Mixin providing functionality for initializing autoencoder layers.
    """

    @staticmethod
    def init_weights(
        modules: nn.ModuleList | nn.ModuleDict, 
        act_name: str | None = None
    ) -> None:
        """
        Initializes weights of model layers.
        Input tensor modification depends on layer connectivity and activation function in use.
        """

        # encoder_layers / decoder_layers 
        if isinstance(modules, nn.ModuleList):
            for m in modules.modules():
                if isinstance(m, nn.Linear):
                    TabularLayerInitMixin._pick_weight_init_func(act_name, m)
                    # Does Pytorch automatically for BatchNorm1d, deselect if bias is set to True for other layers
                    # if m.bias is not None:
                    #     TabularLayerInitMixin._pick_bias_init_func("zeros", m)
        
        # embeddings
        elif isinstance(modules, nn.ModuleDict):
            for m in modules.children():
                nn.init.normal_(m.weight, mean=0.0, std=0.01)

    @staticmethod
    def init_head(
        module: nn.Linear, 
        act_name: str | None
    ) -> None:
        """
        Initializes weights and biases of final compression and reconstruction heads.
        """

        TabularLayerInitMixin._pick_weight_init_func(act_name, module)
        TabularLayerInitMixin._pick_bias_init_func("zeros", module)
    
    @staticmethod
    def _pick_weight_init_func(
        act_name: str | None, 
        m: nn.Module
    ) -> None:
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

    @staticmethod
    def _pick_bias_init_func(
        init_name: str, 
        m: nn.Module
    ) -> None:
        """Selects init function for bias."""
        
        init_dict = {
            "zeros": lambda layer: nn.init.zeros_(layer.bias),
            "orthogonal": lambda layer: nn.init.orthogonal_(layer.bias)
        }
        init_dict[init_name](m)