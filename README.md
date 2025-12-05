# Network Traffic Anomaly Detection System
## Deep Autoencoders for Real-Time Detection of Global Network Threats

A production-grade anomaly detection system that identifies abnormal network traffic patterns across 250 countries, powered by PyTorch, fully custom hybrid autoencoders, and optuna-optimized training pipelines.
The system consumes Cloudflare Radar API telemetry (L3/L7 traffic, bot activity, attack indicators) to detect anomalies such as DDoS attacks, outages, routing irregularities, and botnet behavior in real-time. Provides both:
- CLI inference pipeline
- FastAPI-powered web GUI

## Key features
#### Modeling
- Country-specific autoencoders trained on 3-year traffic data (seasonality-aware)
- Hybrid architecture with
    - learned embeddings for categorical time features
    - separate reconstruction heads for continuous & categorical features
    - weighted hybrid loss (MSE + CE)
- Denoising autoencoder with optional residual connections for improved gradient flow and training stability
- Automated hyperparameter search with Optuna (Hyperband / Median / SHA)
- Temperature scaling & threshold calibration using MAD / p99 / p99.5 methods for optimal anomaly sensitivity
#### Engineering
- <100ms real-time inference (GPU/CPU) with CLI and web GUI interfaces
- containerization with Docker
- CI/CD-ready project structure with modular pipelines
- Monitoring-ready with analysis notebooks & structured result folders

## Model Details
1. Hybrid loss function with individual weights:<br>
    `total_loss = (cont_weight * MSE(continuous_recon, original) + cat_weight * CE(categorical_logits, original))`
2. Feature engineering:
    - Continuous: traffic volumes, attack rates, bot intensity, rolling stats
    - Categorical: weekday, month, daytype, daytime segment, seasonal index
    - Embeddings: learned per-feature categorical representations
    - Time-series augmentations: multi-scale rolling windows, z-scoring, seasonal decomposition
3. Performance metrics:
    |<div align="center">**Metric**</div>|<div align="center">**Value**</div>|<div align="center">**Notes**</div>|
    |--------------------|----------|----------------------------------------------------------------|
    | Detection rate     | 92-97%   | Measured against labeled anomalies                             |
    | False positive rate| 0.5-2%   | Controlled via MAD thresholding                                |
    | Inference latency  | <100ms   | End-to-end pipeline                                            |
    | Training time      | 1-2 h    | Per country with 3 years of data, GPU accelaration not counted |
    | Model size         | <1 MB    | Suitable for edge or serverless deployment                     |

## Data Pipeline
```
Cloudflare Radar API  → Feature Engineering → Country-Specific Models → Anomaly Detection
          ↓                     ↓                      ↓                    ↓                
   Traffic Metrics      Categorical Encoding     Autoencoder         Real-Time Inference
   L3/L7 Attacks         Continuous Scaling        Training        Visualization (CLI, GUI)
   Bot/Crawler Data        Seasonality
   Time-Series              Embedding
```
## Technical Stack
|<div align="center">**Component**</div>|<div align="center">**Technology**</div>|<div align="center">**Purpose**</div>|
|-----------------|-------------------------|--------------------------------------|
| ML Framework	  |   PyTorch 2.4	        | Autoencoder models, GPU acceleration |
| Optimization	  |   Optuna	            | Hyperparameter tuning                |
| Data Processing |   Pandas, NumPy	        | Feature engineering                  |
| Visualization	  |   Matplotlib, Seaborn	| Debugging, analysis, monitoring      |
| API	          |   FastAPI	            | Real-time inference service          |
| Deployment	  |   Docker, Docker Compose| Production runtime                   |
| Monitoring	  |   Custom dashboard	    | Analysis, diagnostics                |


## Getting Started

This project includes a full end-to-end workflow for fetching datasets, training anomaly-detection models, tuning hyperparameters, testing, and running inference.  
For the complete step-by-step guide, please read the **Usage Guide**:

<br> --> see ./scripts/README.md (recommended)

Below is a minimal quick-start.

### 1. Installation
    
    git clone https://github.com/ankaba-x00/ml-anomdetect.git
    cd ml-anomdetect
    
    python -m venv .venv
    source .venv/bin/activate
    
    pip install -r requirements.txt
    

- Run following commands in $PROJECT_ROOT which is ./ml-anomdetect
- Specify countries you want to build as models in: ./ml-anomdetect/app/src/models/models.yml
- Flag -h gives more information on usage, flags, print verbosity etc.
- Example below is for AT model (use <all> for all countries specified in models.yml)
### 2. Data acquision and preprocessing
    
    python -m app.src.data.fetch
    python -m app.src.data.preprocess
    
### 3. Model training, validation, tuning, testing
    
    python -m app.src.pipelines.train_model AT
    python -m app.src.pipelines.validate_model AT
    python -m app.src.pipelines.tune_model --trials 60 --pruner hyperband AT
    python -m app.src.pipelines.test_model AT
    
### 4. Analysis training, validation, tuning, testing
    
    python -m app.src.pipelines.analyze_training AT
    python -m app.src.pipelines.analyze_tuning AT
    python -m app.src.pipelines.analyze_testing AT
    
### 5. Run Inference
    # CLI 
    python -m app.deployment.pipeline -d 12/04/2025 AT
    # GUI 
    docker-compose up --build

## Project structure
```
app/
├── api/              # FastAPI backend with web GUI
├── datasets/         # Raw, processed, feature-engineered dataset stages
├── deployment/       # Production models & inference as CLI
├── src/
│   ├── data/         # Data ingestion & preprocessing
│   ├── exploration/  # EDA, diagnostics, visualizations
│   ├── models/       # Autoencoder implementation
│   └── pipelines/    # Training, validation, tuning & testing workflows
└── tests/            # PyTest-based unit tests
results/              # Trained models, histories, scalers, visual outputs
scripts/              # Automation & maintenance tasks
```

## This Project Demonstrates
- Full-stack ML engineering: from API ingestion -> modeling -> deployment
- Deep Learning knowledge: custom autoencoder architectures
- MLOps workflow design: reproducible pipelines, tuning, calibration
- Software engineering best practices: modularization, testing, logging, CI/CD
- Real-world problem-solving at global production scale

## Contact
Always open for constructive criticism and code roasts, and happy to acknowledge your contribution.
For contributions, comments or collaborations, please open an issue or reach out directly.

## License
This project is strictly proprietary.

Only personal, non-commercial use is permitted.  
Academic use, research use, redistribution, modification, and commercial use  
are expressly forbidden.  

See the LICENSE file for full terms.
