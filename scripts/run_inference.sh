#!/bin/bash

##################################
# CONFIGURATION - EDIT THIS ONLY #
##################################

### INFERENCE ARGUMENTS ###
# Target country (or "all") as specified in app/src/models/models.yml
TARGET="AT"
MODEL="vae"  # ae | vae
PREDICTION_DATE="11/15/2025"  # format MM/DD/YYYY
TRAINING=false

### TRAINING ARGUMENTS ###
THRESHOLD_METHOD="p99"  # p99 | p995 | mad
CAL_WINDOW=30 
PLOT_LATENT=true 

#####################################
# BUILD ARGUMENT LIST - DO NOT EDIT #
#####################################

### TRAINING ARGUMENTS (FULL MODE) ###
ARGS_TRAIN=""
ARGS_TRAIN+=" --full" 
ARGS_TRAIN+=" -M $THRESHOLD_METHOD"
ARGS_TRAIN+=" -CW $CAL_WINDOW"
$PLOT_LATENT && ARGS_TRAIN+=" -L"
ARGS_TRAIN+=" $MODEL"
ARGS_TRAIN+=" $TARGET"


### INFERENCE ARGUMENTS ###
ARGS_INFER=""
ARGS_INFER+=" -d $PREDICTION_DATE"
ARGS_INFER+=" $MODEL"
ARGS_INFER+=" $TARGET"

############################
# EXECUTION  - DO NOT EDIT #
############################

if [ "$TRAINING" = true ]; then
    echo "===================================="
    echo "[TRAIN] Executing:"
    echo "python -m app.src.pipelines.train_model $ARGS_TRAIN"
    eval python -m app.src.pipelines.train_model $ARGS_TRAIN
    echo "[TRAIN] Completed!"
    echo
else
    echo "[SKIPPED] Full training; assume models are ready."
    echo 
fi

echo "===================================="
echo "[INFERENCE] Executing:"
echo "python -m app.deployment.pipeline $ARGS_INFER"
eval python -m app.deployment.pipeline $ARGS_INFER
echo "[INFERENCE] Completed!"

echo
echo "===================================="
echo "[DONE] Inference workflow completed."