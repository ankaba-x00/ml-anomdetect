# ANOMALY DETECTION MODULE 

A modular pipeline for collecting, preprocessing, and analyzing global traffic data for anomaly detection.

## Project features
1. Fetch historical Cloudflare traffic, attack, bot and anomaly data.
2. Normalize data into machine-learning-ready feature tensors.
3. Apply different anomaly-detection model pipelines e.g.
    - isolated forest
    - autoencoder
    - LSTM/Transformer-based forecasting residuals
    - etc.
4. Predict gloabal or country-level anomalies in current internet traffic.

## Module usage
run the following commands in $PROJECT_ROOT:

### Run complete module: 
                        __add__later__

### Run stages
1. Run fetching stage
                        python -m src.data.fetch
2. Run feature-engineering stage
                        __add__later__
3. Run model training stage
                        __add__later__
4. Run inference stage
                        __add__later__