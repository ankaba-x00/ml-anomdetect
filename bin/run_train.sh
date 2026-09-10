#!/bin/bash

##################################
# CONFIGURATION - EDIT THIS ONLY #
##################################

# FOR BUILDING MULTIPLE MODELS, SELECT MULTIPLE COUNTRIES app/src/config/models.yml

### TRAINING ARGUMENTS ###
TARGET="AT" # 2-letter country code | all 
# TARGET specifies country you want to build a model for; use 'all' for all countries specified in app/src/config/models.yml
MODEL="ae"  # ae | vae | mtae
# MODEL specifies autoencoder type
CONFIG_PATH="config" # config | trained | tuned 
# CONFIG_PATH specifes where model config is read from: from config files see app/src/config/train, or from previous training run or after tuning
 
TRAIN_RATIO=75 
VAL_RATIO=15
FULL_DATA=false 
THRESHOLD_METHOD="p99"  # p99 | p995 | mad
CAL_WINDOW=30 
PLOT_LATENT_TRAIN=true 

### VALIDATING ARGUMENTS ###
PLOT_LATENT_VAL=true 

### ANALYZE TRAINING ARGUMENTS ###
SHOW_PLOTS=false

#####################################
# BUILD ARGUMENT LIST - DO NOT EDIT #
#####################################

### BUILD ARGUMENTS ###
BUILD_AE="unsuper"
BUILD_MT="super"
if [ $MODEL = "mtae" ];
then
    BUILD=$BUILD_MT
else
    BUILD=$BUILD_AE
fi

### TRAINING ARGUMENTS ###
ARGS_TRAIN=""
ARGS_TRAIN+="-tr $TRAIN_RATIO"
ARGS_TRAIN+=" -vr $VAL_RATIO"
$FULL_DATA && ARGS_TRAIN+=" --full"
ARGS_TRAIN+=" -M $THRESHOLD_METHOD"
ARGS_TRAIN+=" -CW $CAL_WINDOW"
$PLOT_LATENT_TRAIN && ARGS_TRAIN+=" -L"
ARGS_TRAIN+=" $CONFIG_PATH"
ARGS_TRAIN+=" $MODEL"
ARGS_TRAIN+=" $TARGET"


### VALIDATION ARGUMENTS ###
ARGS_VALIDATE=""
ARGS_VALIDATE+="-tr $TRAIN_RATIO"
ARGS_VALIDATE+=" -vr $VAL_RATIO"
$PLOT_LATENT_VAL && ARGS_VALIDATE+=" -L"
ARGS_VALIDATE+=" $MODEL"
ARGS_VALIDATE+=" $TARGET"


### ANALYZE TRAINING ARGUMENTS ###
ARGS_ANALYZE=""
ARGS_ANALYZE="-M $THRESHOLD_METHOD"
$SHOW_PLOTS && ARGS_ANALYZE+=" -s"
ARGS_ANALYZE+=" $MODEL"
ARGS_ANALYZE+=" $TARGET"

############################
# EXECUTION  - DO NOT EDIT #
############################

echo "===================================="
echo "[TRAIN] Executing:"
echo "python -m app.src.pipelines.train_model $ARGS_TRAIN"
eval python -m app.src.pipelines.build_features -B -S $BUILD $TARGET
eval python -m app.src.pipelines.train_model $ARGS_TRAIN
echo "[TRAIN] Completed!"
echo

if [ "$FULL_DATA" = false ] && [ "$TRAIN_RATIO" -ne 100 ]; then

    echo "===================================="
    echo "[VALIDATE] Executing:"
    echo "python -m app.src.pipelines.validate_model $ARGS_VALIDATE"
    eval python -m app.src.pipelines.validate_model $ARGS_VALIDATE
    echo "[VALIDATE] Completed!"
    echo

    echo "===================================="
    echo "[ANALYZE] Executing:"
    echo "python -m app.src.pipelines.analyze_training $ARGS_ANALYZE"
    eval python -m app.src.pipelines.analyze_training $ARGS_ANALYZE
    echo "[ANALYZE] Completed!"
    echo

    echo "===================================="
    echo "[DONE] Full training and validation workflow completed!"

else
    echo "===================================="
    echo "[DONE] Full training workflow completed!"
    echo 

    echo "[SKIPPED] Validation + Analysis skipped because:"
    echo "  FULL_DATA=$FULL_DATA  (otherwise must be false)"
    echo "  TRAIN_RATIO=$TRAIN_RATIO  (otherwise must NOT be 100)"
fi