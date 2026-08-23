import csv, optuna, yaml
from collections import OrderedDict
from dataclasses import asdict
from pathlib import Path
from typing import Any

from app.src.ml.models.configs import AEConfig, VAEConfig, MTAEConfig


class YMLReaderError(Exception):
    """Raised when loading or parsing YAML tuning configuration fails."""

    pass


class YMLReader:
    """
    Reads hyperparameter search space from YAML tuning configuration, passes tuning parameters to objective function as trial suggestion variables and generates final config object for trial run.
    """

    def __init__(self, 
        ae_type: str,
        trial: optuna.Trial | None
    ):
        self.ae_type = ae_type
        self.trial = trial
    
    def load_search_space(
        self, 
        num_cont: int, 
        cat_dims: dict[str, int]
    ) -> AEConfig | VAEConfig | MTAEConfig:
        "Generates config object for Obtuna objective."
        params = self.extract_params()

        search_space = {}
        for key, value in params["tune"].items():
            search_space[key] = self._add_param_to_objective(key, value)
        
        config = self._add_params_to_config(
            params["static"] | search_space, 
            num_cont, 
            cat_dims
        )

        return config

    def extract_params(self) -> dict[str, Any]:
        """Loads tuning configuration from YML file."""

        param_file = Path(__file__).resolve().parents[2] / "config" / "tune" / f"param_{self.ae_type}.yml"
        if not param_file.exists():
            raise YMLReaderError(f"[ERROR] YML tuning configuration not found: {param_file}")

        try:
            with open(param_file, "r") as f:
                params = yaml.load(f, Loader=yaml.SafeLoader)
            print(f"[OK] YML params of {self.ae_type} loaded")
        except AttributeError:
            raise YMLReaderError("[ERROR] YML file empty")
        except Exception as e:
            raise YMLReaderError(f"[ERROR] Loading YML tuning configuration failed: {e}")
        
        return params

    def _add_param_to_objective(
        self, 
        key: str, 
        value: dict[str, int | float] | list[int | float | str] 
    ) -> None:
        "Adds search space parameters to objective function."
        if isinstance(value, dict):
            if set(value.keys()) != set(["start", "end"]):
                raise YMLReaderError(f"[ERROR] YML tune section value error for param {key}: only dict with start/end keys allowed")
            if all(isinstance(v, int) for v in value.values()):
                return self.trial.suggest_int(key, value["start"], value["end"])
            elif all(isinstance(v, float) for v in value.values()):
                if value["start"] != 0.0 and value["end"] / value["start"] >= 100:
                    return self.trial.suggest_float(key, value["start"], value["end"], log=True)
                return self.trial.suggest_float(key, value["start"], value["end"])
            else: 
                raise YMLReaderError(f"[ERROR] YML tune section value error for param {key}: all dict values must be of same type and either int or float") 
        elif isinstance(value, list):
            if not all(isinstance(v, type(v)) for v in value):
                raise YMLReaderError(f"[ERROR] YML tune section value error for param {key}: all dict values must be of same type")
            return self.trial.suggest_categorical(key, value)
        else: 
            raise YMLReaderError(f"[ERROR] YML tune section value error for param {key}: only dict with start/end or list allowed")
    
    def _add_params_to_config(
        self, 
        params: dict[str, Any], 
        num_cont: int, 
        cat_dims: dict[str, int],
    ) -> AEConfig | VAEConfig | MTAEConfig:
        "Adds config parameters to config object."

        config_map = {
            "ae": AEConfig,
            "vae": VAEConfig,
            "mtae": MTAEConfig
        }
        
        params["num_cont"] = num_cont
        params["cat_dims"] = cat_dims
        cfg = config_map[self.ae_type](**params)
        
        keys = set(params.keys()) - cfg.get_args()
        self.print_defaults(cfg, keys)

        return cfg
    
    def print_defaults(
        self, 
        cfg: AEConfig | VAEConfig | MTAEConfig, 
        params: set[str]
    ) -> None:
        "Prints missing config parameters not incl. in the YML tuning configuration with respective default values used in the trial."
        defaults = cfg.get_kwargs() ^ params
        if defaults:
            print("[YML] Config parameters missing in YML file, using defaults for:")
            for d in defaults:
                if d != "hidden_dims":
                    print(f"      {d}: {asdict(cfg)[d]}")


class TrialSummaryWriter:
    """
    Saves trial summary incl. hyperparameter search space from YAML tuning configuration, trial configuration and outcome.
    """

    def __init__(
        self, 
        retune_no: int, 
        ae_type: str,
        sampler: str,
        pruner: str,
        best_no: int,
        best_result: float

    ):
        self.retune_no = retune_no
        self.ae_type = ae_type
        self.sampler = sampler
        self.pruner = pruner
        self.best_no = best_no
        self.best_result = best_result

    def write(self, summary_path: Path):
        "Writes trial summary."

        trial_info = self._trial_writeout()
        search_info = self._search_writeout()
        output = trial_info | search_info
        
        mode = "w" if self.retune_no == 0 else "a"
        with open(summary_path, mode=mode, newline="") as f:
            writer = csv.DictWriter(f, fieldnames=output.keys(), delimiter=";")
            if self.retune_no == 0:
                writer.writeheader()
            writer.writerow(output)
    
    def _trial_writeout(self) -> dict[str, int | float | str]:
        "Assembles trial information for trial summary."
        return {
            "phase": self.retune_no,
            "sampler": self.sampler,
            "pruner": self.pruner,
            "best_no": self.best_no,
            "best_result": self.best_result
        }
        

    def _search_writeout(self) -> OrderedDict[str, str]:
        "Assembles and prettifies search space parameters for trial summary."
        params = YMLReader(self.ae_type, None).extract_params()
        clean_params = {}

        for k, v in params["static"].items():
            clean_params[k] = str(v)
        
        for k, v in params["tune"].items():
            if isinstance(v, dict):
                clean_params[k] = f"{v['start']}-{v['end']}"
            else:
                clean_params[k] = ",".join(map(str, v))
        
        return OrderedDict(sorted(clean_params.items(), key=lambda item: item[0]))




        

