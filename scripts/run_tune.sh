#!/bin/bash

##################################
# CONFIGURATION - EDIT THIS ONLY #
##################################

### TUNING ARGUMENTS ###
# Target country (or "all") as specified in app/src/models/models.yml
TARGET="AT"

N_TRIALS=40
PRUNER="median"  # median | halving | hyperband
TRAIN_RATIO=75
VAL_RATIO=15
PLOT_LATENT_TUNE=true 

MULTI_TUNE_PLOTS=false
SHOW_TUNE_PLOTS=false

### TESTING ARGUMENTS ###
THRESHOLD_METHOD="p99"  # p99 | p995 | mad
PLOT_LATENT_TEST=true

PLOT_RAW_SIGNAL_PLOTS=false
SHOW_TEST_PLOTS=false

#####################################
# BUILD ARGUMENT LIST - DO NOT EDIT #
#####################################

### TUNING ARGUMENTS ###
ARGS_TUNE=""
ARGS_TUNE+=" -n $N_TRIALS"
ARGS_TUNE+=" -p $PRUNER"
ARGS_TUNE+=" -tr $TRAIN_RATIO"
ARGS_TUNE+=" -vr $VAL_RATIO"
$PLOT_LATENT_TUNE && ARGS_TUNE+=" -L"
ARGS_TUNE+=" $TARGET"


### ANALYZE TUNING ARGUMENTS ###
ARGS_ANALYZE_TUNE=""
$SHOW_TUNE_PLOTS && ARGS_ANALYZE_TUNE+=" -s"
$MULTI_TUNE_PLOTS && ARGS_ANALYZE_TUNE+=" -M"
ARGS_ANALYZE_TUNE+=" $TARGET"


### TESTING ARGUMENTS ###
ARGS_TEST=""
ARGS_TEST+=" -M $THRESHOLD_METHOD"
ARGS_TEST+=" -tr $TRAIN_RATIO"
ARGS_TEST+=" -vr $VAL_RATIO"
$PLOT_LATENT_TEST && ARGS_TEST+=" -L"
ARGS_TEST+=" $TARGET"


### ANALYZE TEST ARGUMENTS ###
ARGS_ANALYZE_TEST=""
$SHOW_TEST_PLOTS && ARGS_ANALYZE_TEST+=" -s"
$PLOT_RAW_SIGNAL_PLOTS && ARGS_ANALYZE_TEST+=" -R"
ARGS_ANALYZE_TEST+=" -M $THRESHOLD_METHOD"
ARGS_ANALYZE_TEST+=" $TARGET"

############################
# EXECUTION  - DO NOT EDIT #
############################

echo "===================================="
echo "[TUNE] Executing:"
echo "python -m app.src.pipelines.tune_model $ARGS_TUNE"
eval python -m app.src.pipelines.tune_model $ARGS_TUNE
echo "[TUNE] Completed!"
echo

echo "===================================="
echo "[ANALYZE] Executing:"
echo "python -m app.src.pipelines.analyze_tuning $ARGS_ANALYZE_TUNE"
eval python -m app.src.pipelines.analyze_tuning $ARGS_ANALYZE_TUNE
echo "[ANALYZE] Completed!"
echo

echo "===================================="
echo "[TEST] Executing:"
echo "python -m app.src.pipelines.test_model $ARGS_TEST"
eval python -m app.src.pipelines.test_model $ARGS_TEST
echo "[TEST] Completed!"
echo

echo "===================================="
echo "[ANALYZE] Executing:"
echo "python -m app.src.pipelines.analyze_testing $ARGS_ANALYZE_TEST"
eval python -m app.src.pipelines.analyze_testing $ARGS_ANALYZE_TEST
echo "[ANALYZE] Completed!"
echo

echo "===================================="
echo "[DONE] Full tuning and testing workflow completed!"
