#!/bin/bash

##################################
# CONFIGURATION - EDIT THIS ONLY #
##################################

### INFERENCE ARGUMENTS ###
TARGET="AT"
# TARGET specifies country for which you want to use country-specific model for inference
MODEL="ae"  # ae | vae | mtae
# MODEL specifies autoencoder type
PREDICTION_DATE="11/15/2025"  # format MM/DD/YYYY
# DATE specifies date for which you want to make a prediction for
PREPARE_BUNDLE=false
# PREPARE_BUNDLE specifies whether you want to prepare inference package which is necessary before the first inference run

### PREPARATION ARGUMENTS ###
CONFIG_PATH="trained" # config | trained | tuned 
# CONFIG_PATH specifes where model config is read from: from config files see app/src/config/train, or from previous training run or after tuning
THRESHOLD_METHOD="p99"  # p99 | p995 | mad
CAL_WINDOW=30 
PLOT_LATENT=true 

#####################################
# BUILD ARGUMENT LIST - DO NOT EDIT #
#####################################

### PREPARATION ARGUMENTS (FULL MODE) ###
ARGS_PREP=""
ARGS_PREP+="--full" 
ARGS_PREP+=" -M $THRESHOLD_METHOD"
ARGS_PREP+=" -CW $CAL_WINDOW"
$PLOT_LATENT && ARGS_PREP+=" -L"
ARGS_PREP+=" $CONFIG_PATH"
ARGS_PREP+=" $MODEL"
ARGS_PREP+=" $TARGET"


### INFERENCE ARGUMENTS ###
ARGS_INFER=""
ARGS_INFER+="-d $PREDICTION_DATE"
ARGS_INFER+=" $MODEL"
ARGS_INFER+=" $TARGET"

############################
# EXECUTION  - DO NOT EDIT #
############################

if [ "$PREPARE_BUNDLE" = true ]; then
    echo "===================================="
    echo "[TRAIN] Executing:"
    echo "python -m app.src.pipelines.train_model $ARGS_PREP"
    eval python -m app.src.pipelines.train_model $ARGS_PREP
    echo "[TRAIN] Completed!"
    echo
else
    echo "[SKIPPED] Full training; assume models are ready."
    echo 
fi

echo "===================================="
echo "[INFERENCE] Executing:"
echo "python -m app.deployment.use_model $ARGS_INFER"
eval python -m app.deployment.use_model $ARGS_INFER
echo "[INFERENCE] Completed!"

echo
echo "===================================="
echo "[DONE] Inference workflow completed."