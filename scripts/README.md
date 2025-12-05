# MODULE USAGE

## This package has multiple sequential stages
1. You need to fetch datasets
2. You need to build and train models
3. You need to tune models
4. You can use models now to make predictions

## Step-by-step instructions

### PREREQUISITES

1. Clone this project 
    ```
    git clone https://github.com/ankaba-x00/ml-anomdetect.git
    cd ml-anomdetect
    ```
2. Install Python (see https://www.python.org/downloads/)
3. Create a local environment
    ```
    python -m venv .venv
    source .venv/bin/activate
    ```
4. Install requirements
    ```
    pip install -r requirements.txt
    ```
5. (Optionally) Install Docker to run the GUI for model use (see https://www.docker.com/get-started/)
6. Make sure that all run files are executables
    ```( cd scripts && find . -maxdepth 1 -type f ! -name "README.md" -exec chmod +x {} \; )```

### 1. FETCH DATASETS

1. Get access to Cloudflare API by creating a token (see https://developers.cloudflare.com/api/)
2. Store your token in .env file in $PROJECT_ROOT which is inside ml-anomdetect folder 
    ```echo "API_TOKEN=YOUR_TOKEN_COPY" > .env```
3. Open ./scripts/run_fetch.sh and edit fetch configuration parameters to your liking.
    3.1. EXAMPLE1: To run anomaly predictions with 1 year of training data, you only need timeseries data, ergo changing the following parameters is enough
    ```
    START_DATE="12/05/2024"
    END_DATE="12/05/2025"
    FETCH_TIME=true
    ``` 
    3.2. EXAMPLE2 To fetch complete dataset with 3 years of training data, the following parameters are enough
    ```
    START_DATE="12/05/2022"
    END_DATE="12/05/2025"
    FETCH_ALL=true
    ```
    3.3. EXAMPLE3 To fetch a particular dataset within a date range, the following parameter suffice
    ```
    START_DATE="12/05/2025"
    END_DATE="12/07/2025"
    FETCH_TRAFFIC=true
    ```
    3.4. ADVICE: To speed up a multi-year fetches, you can split up the fetch in multiple pulls and then merge the datasets later (see below). To avoid CloudFlare API rate limit, it is highly recommended to start these multi-pull fetches sequentially on different days. 
    3.5. WARNING: Each fetch produces a raw dataset file with the day of your pull as file name ending. Pulling the same dataset on the same date, will overwrite the previous pull. Adjust naming convention to your usage in ./app/src/data/fetch.py directly. 
4. Run run_fetch.sh via
    ```./scripts/run_fetch.sh```
5. Preprocess data depending on your usage.
    5.1. One pull per dataset file
        5.1.1 EXAMPLE1: If you downloaded all datasets, run
        ```python -m app.src.data.preprocess all```
        5.1.2 EXAMPLE2: If you did not fetch all datasets, you can specify the file key instead of all. You get a list of all file keys via
        ```python -m app.src.data.preprocess -k all```
        5.1.3. ADVICE: If you have multiple datasets you need to preprocess, remove individual fetch keys from the file ./app/src/data/preprocess directly to automate the process for your dataset bundle
    5.2. Multiple pulls per dataset file
        5.2.1 EXAMPLE1: If you downloaded all datasets and have 3 pulls per dataset file, you need to check the pull directions. If the first downloaded file is fetching data before the second, the pulls are consecutively, ergo merge direction = 0
        ```python -m app.src.data.merge_preprocess -N 3 0 all``` 
        5.2.2 EXAMPLE2: If the first downloaded file is fetching data after the second, the pulls are non-consecutively, ergo merge direction = 1
        ```python -m app.src.data.merge_preprocess -N 3 1 all```
        5.2.3. ADVICE: If you have multiple datasets you need to preprocess, remove individual fetch keys from the file ./app/src/data/merge_preprocess directly to automate the process for your dataset bundle
    5.3. You can run an automated analysis on your raw datasets via
    ```python -m app.src.exploration.exploration```

### 2. BUILD AND TRAIN MODELS
1. Specify all countries you want to build as models in ./app/src/models/models.yml by adding their respective 2-letter country codes. For a list of 2-letter country codes, see regions dictionary in ./app/src/exploration/core/params.py
2. Open ./scripts/run_train.sh and edit training and validation configuration parameters to your liking.
3. Run run_train.sh to train models as specified
    ```./scripts/run_train.sh```

### 3. TUNE MODELS
1. Open ./scripts/run_tune.sh and edit tuning and testing configuration parameters to your liking.
2. Run run_tune.sh to tune and test models as specified
    ```./scripts/run_train.sh```

### 4. USE MODELS
#### CLI
1. Open ./scripts/run_inference.sh and edit inference configuration parameters to your liking.
2. Make sure that you produce models trained on full dataset for inference. Hence, to prepare inference and test it for teh first time, run inference on all datasets, with the following configuration
    ```
    TARGET="all"
    PREDICTION_DATE="11/15/2025"  # make sure prediction date is AFTER your training date range
    TRAINING=true  # !! very important
    THRESHOLD_METHOD="p995"  # best threshold obtained in tuning
    CAL_WINDOW=30  # recommended  
    ```
3. WARNING: Do not set TRAINING=true for all inference runs. This is only necessary for preparing the final models ready for inference! All subsequent runs should set TRAINING=false
4. Run run_inference.sh to either prepare models for inference or perform inference as specified
    ```./scripts/run_inference.sh```

#### GUI
1. To run inference in GUI, make sure all models are prepared for inference (see above CLI steps 1, 2 and 4)
2. Build Docker images and bring up all services defined in docker-compose.yml via
    ```docker-compose up --build ```
3. Visit the web GUI in a browser of your choosing by visiting http://0.0.0.0:8000/ and have fun :-) 