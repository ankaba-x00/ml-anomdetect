#!/bin/bash

##################################
# CONFIGURATION - EDIT THIS ONLY #
##################################

### DATE RANGE ###
START_DATE="08/01/2025"  # format MM/DD/YYYY
END_DATE="08/31/2025"  # format MM/DD/YYYY

### TOGGLE DATASSET GROUPS ###
FETCH_ALL=false
FETCH_TIME=false
FETCH_NOTIME=false

### OR ###
 
### TOGGLE INDIVIDUAL DATASETS ###
# HTTP requests
FETCH_HTTPREQ=false
FETCH_HTTPREQ_TIME=false
FETCH_HTTPREQ_AUTOMATED=false

# Netflow traffic
FETCH_TRAFFIC=false
FETCH_TRAFFIC_TIME=false

# Bots / AI bots
FETCH_AIBOTS_TIME=false
FETCH_BOTS_TIME=false

# Internet quality and Anomalies
FETCH_IQ_TIME=false
FETCH_ANOMALIES=false

# Layer 3 attacks
FETCH_L3_ORIGIN=false
FETCH_L3_TARGET=false
FETCH_L3_ORIGIN_TIME=false
FETCH_L3_TARGET_TIME=false

# Layer 7 attacks
FETCH_L7_ORIGIN=false
FETCH_L7_TARGET=false
FETCH_L7_TIME=false

#####################################
# BUILD ARGUMENT LIST - DO NOT EDIT #
#####################################

ARGS=""

# Dates
ARGS+=" -S \"$START_DATE\""
ARGS+=" -E \"$END_DATE\""

# Global groups
$FETCH_ALL && ARGS+=" -A"
$FETCH_TIME && ARGS+=" -T"
$FETCH_NOTIME && ARGS+=" -N"

# Individual groups
$FETCH_HTTPREQ && ARGS+=" -hr"
$FETCH_HTTPREQ_TIME && ARGS+=" -hrt"
$FETCH_HTTPREQ_AUTOMATED && ARGS+=" -hra"

$FETCH_TRAFFIC && ARGS+=" -t"
$FETCH_TRAFFIC_TIME && ARGS+=" -tt"

$FETCH_AIBOTS_TIME && ARGS+=" -at"
$FETCH_BOTS_TIME && ARGS+=" -bt"

$FETCH_IQ_TIME && ARGS+=" -iqt"
$FETCH_ANOMALIES && ARGS+=" -a"

$FETCH_L3_ORIGIN && ARGS+=" -l3or"
$FETCH_L3_TARGET && ARGS+=" -l3ta"
$FETCH_L3_ORIGIN_TIME && ARGS+=" -l3ort"
$FETCH_L3_TARGET_TIME && ARGS+=" -l3tat"

$FETCH_L7_ORIGIN && ARGS+=" -l7or"
$FETCH_L7_TARGET && ARGS+=" -l7ta"
$FETCH_L7_TIME && ARGS+=" -l7t"

############################
# EXECUTION  - DO NOT EDIT #
############################

echo "[FETCH] Executing:"
echo "python -m app.src.data.fetch $ARGS"
eval python -m app.src.data.fetch $ARGS
echo "[FETCH] Completed!"
echo 
echo "[DONE] Full fetch workflow completed!"