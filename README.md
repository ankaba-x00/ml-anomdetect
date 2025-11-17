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
4. Perfom hyperparameter tuning on model of choice
5. Predict gloabal or country-level anomalies in current internet traffic.

## Module usage
run the following commands in $PROJECT_ROOT:

### Run module: 
1. Run fetching stage
                        python -m scripts.run_fetch
2. Run feature-engineering stage
                        python -m scripts.run_prep
3. Run model training stage
                        python -m scripts.run_train
4. Run model tuning stage
                        python -m scripts.run_tune
5. Run inference stage
                        python -m scripts.run_inference