# BEST PRACTICE TUNING

## Currently implemented options for optimizer/LR scheduler and activation functions
(A) optimizers 
    - adam
    - adamw
    - sdg
(B) LR schedulers
    - none
    - onecycle
    - plateau
    - cosine
    - step
(C) activation functions
    - relu
    - leaky_relu
    - tanh
    - sigmoid
    - silu

## Tuning objective:
    1. Choose 1 optimizer per objective 
    2. Choose 1 LR scheduler working well with optimizer per objective
    3. Choose LR according to optimizer and LR scheduler pairing

## Suggested pairings for autoencoders
    (A) adam + plateau + LR = 1e-3 + weight_decay = 1e-6, 1e-1 || pruner = hyperband or median
    (B) adamw + cosine + LR = 5e-4 or 1e-3 + weight_decay = 1e-6, 1e-1 || pruner = hyperband or median
    (C) adam + none + LR = 1e-3 + weight_decay = 1e-6, 1e-1 || pruner = hyperband or median
    (D) sdg + step + LR = 1e-2 + weight_decay = 1e-6, 1e-1 || pruner = hyperband or median

    in TRIALS adjust
        LR: 1/2*LR and 2*LR first and then 1e-2 down and 1e-5 for retune

## Adding/changing optimizers and LR schedulers 
    - may require additional parameters in search space for tuning
    - if the parameter is dependent on a particular optimizer or LR_scheduler, 
      add "name_" before the param with name being name of optimizer or LR scheduler  
    - to add new parameters to search space, adjust these files:
        1. add search space params
            app/src/config/tune/param_<MODEL>.yml
        2. add params to config class of respective models
            app/src/ml/models/<model>.py
        3. add params in model read-in config
            app/src/config/train/config_<MODEL>.json