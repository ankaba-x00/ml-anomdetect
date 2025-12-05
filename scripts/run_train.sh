#!/bin/bash

##################################
# CONFIGURATION - EDIT THIS ONLY #
##################################

### TRAINING ARGUMENTS ###
# Target country (or "all") as specified in app/src/models/models.yml
TARGET="AT" 
TRAIN_RATIO=75 
VAL_RATIO=15
FULL_DATA=false 
THRESHOLD_METHOD="p99"  # p99 | p995 | mad
CAL_WINDOW=30 
PLOT_LATENT_TRAIN=true 

### VALIDATING ARGUMENTS ###
PLOT_LATENT_VAL=true 

### ANALYSIS ARGUMENTS ###
SHOW_PLOTS=false

#####################################
# BUILD ARGUMENT LIST - DO NOT EDIT #
#####################################

### TRAINING ARGUMENTS ###
ARGS_TRAIN=""
ARGS_TRAIN+=" -tr $TRAIN_RATIO"
ARGS_TRAIN+=" -vr $VAL_RATIO"
$FULL_DATA && ARGS_TRAIN+=" --full"
ARGS_TRAIN+=" -M $THRESHOLD_METHOD"
ARGS_TRAIN+=" -CW $CAL_WINDOW"
$PLOT_LATENT_TRAIN && ARGS_TRAIN+=" -L"
ARGS_TRAIN+=" $TARGET"


### VALIDATION ARGUMENTS ###
ARGS_VALIDATE=""
ARGS_VALIDATE+=" -tr $TRAIN_RATIO"
ARGS_VALIDATE+=" -vr $VAL_RATIO"
$PLOT_LATENT_VAL && ARGS_VALIDATE+=" -L"
ARGS_VALIDATE+=" $TARGET"


### ANALYSIS ARGUMENTS (TARGET ONLY) ###
ARGS_ANALYZE="$TARGET"
$SHOW_PLOTS && ARGS_ANALYZE+=" -s"

############################
# EXECUTION  - DO NOT EDIT #
############################

echo "===================================="
echo "[TRAIN] Executing:"
echo "python -m app.src.pipelines.train_model $ARGS_TRAIN"
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
    echo "  FULL_DATA=$FULL_DATA  (must be false)"
    echo "  TRAIN_RATIO=$TRAIN_RATIO  (must NOT be 100)"
fi