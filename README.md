# Anomaly Detection Module
## Autoencoders for real-time detection of global network threats

A production-grade anomaly detection module that identifies abnormal network traffic patterns across 250 countries, powered by PyTorch, fully custom hybrid autoencoders, trained on 3-year country-specific traffic data (seasonality-aware), with automated Optuna tuning pipeline and modular package structure. The system consumes Cloudflare Radar API telemetry (L3/L7 traffic, bot activity, attack indicators etc.) to detect anomalies such as DDoS attacks, outages, routing irregularities, and botnet behavior in real-time. Provides both:
- CLI inference pipeline
- FastAPI-powered GUI web dashboard

## Key features
#### Modeling
- 3 different autoencoders for both unsupervised and supervised learning
    - Denoising autoencoder with symmetrical encoder and decoder for anomaly detection
    - Variational autoencoder with reparametrized latent space for anomaly detection
    - Multi-task autoencoder with asymmetrical encoder and decoder with sepearte regression and classification heads for anomaly detection and prediction of L3/l7 intensities and attack type
- Hybrid architecture with
    - customizable encoder and decoder structure
    - learned embeddings or encoding for categorical time features
    - separate reconstruction heads for continuous and categorical features
    - weighted hybrid loss (huber loss or MSE + CE) for reconstruction, incl. KL-divergence for variational models, focal and quantile loss for multi-task model 
    - optional noise injection with residual connections for improved gradient flow and training stability
    - different warmup routines for improved latent regularization 
- Automated hyperparameter tuning with Optuna
- Temperature scaling and threshold calibration for optimal anomaly sensitivity
- A custom labelling schema for supervised attack type classification to differentiate 8 traffic patterns

#### Engineering
- <100ms Real-time inference (GPU/CPU) with CLI and web GUI interfaces
- Containerization with Docker
- CI/CD-ready project structure with modular pipelines
- Monitoring-ready with analysis pipelines & structured result folders

## Model details
1. Hybrid loss functions with individual weights:<br>
    - Reconstruction:<br>
    `total_loss = (𝝺_cont * HUBER(cont_recon, original) + 𝝺_cat * CE(cat_logits, original))`
    - Variational reconstruction:<br>
    `total_loss = (𝝺_cont * HUBER(cont_recon, original) + 𝝺_cat * CE(cat_logits, original)) + β D_KL(mu, logvar)`
    - MT Prediction:<br>
    `total_loss = (𝝺_cont * HUBER(cont_recon, original) + 𝝺_cat * CE(cat_logits, original)) + 𝛼 * [𝝺_l3 * QUANT_LOSS_l3(l3_pred, original) + 𝝺_l7 * QUANT_LOSS_l7(l7_pred, original) + 𝝺_at * FOCAL_LOSS(at_logits, original)]`
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

## Data pipeline
```
Cloudflare Radar API  → Feature Engineering → Country-Specific Models → Anomaly Detection
          ↓                     ↓                      ↓                    ↓                
   Traffic Metrics      Categorical Encoding     Autoencoder         Real-Time Inference
   L3/L7 Attacks         Continuous Scaling        Training        Visualization (CLI, GUI)
   Bot/Crawler Data        Seasonality
   Time-Series              Embedding
```
## Technical stack
|<div align="center">**Component**</div>|<div align="center">**Technology**</div>|<div align="center">**Purpose**</div>|
|-----------------|-------------------------|--------------------------------------|
| ML Framework	  |   PyTorch 2.4	        | Autoencoder models, GPU acceleration |
| Optimization	  |   Optuna	            | Hyperparameter tuning                |
| Data Processing |   Pandas, NumPy	        | Feature engineering                  |
| Visualization	  |   Matplotlib, Seaborn	| Debugging, analysis, monitoring      |
| API	          |   FastAPI	            | Real-time inference service          |
| Deployment	  |   Docker, Docker Compose| Production runtime                   |
| Monitoring	  |   Custom dashboard	    | Analysis, diagnostics                |


## Getting started

This project includes a full end-to-end workflow for fetching datasets, training models, tuning hyperparameters, testing, and running inference for anomaly detection.  
For the complete step-by-step guide, please read the **Usage Guide**:
<br> --> see ./bin/README_usage.md (**highly recommended**)
Below is a minimal quick-start.

### 1. Installation
    
    git clone https://github.com/ankaba-x00/ml-anomdetect.git
    cd ml-anomdetect
    
    python -m venv .venv
    source .venv/bin/activate
    
    pip install -r requirements.txt
    

- Run following commands in $PROJECT_ROOT which is ./ml-anomdetect
- Specify countries you want to build as models in: ./ml-anomdetect/app/src/config/models.yml
- Flag -h gives more information on usage, flags, print verbosity etc.
- Example below is for AT model (use <all> for all countries specified in models.yml)

### 2. Data acquision and preprocessing
    
    python -m app.src.data.fetch
    python -m app.src.data.preprocess
    
### 3. Multi-task model training, validation, tuning, testing
    
    python -m app.src.pipelines.build_features -B -S super AT
    python -m app.src.pipelines.train_model config mtae AT
    python -m app.src.pipelines.validate_model mtae AT
    python -m app.src.pipelines.tune_model --ntrials 60 --pruner hyperband mtae AT
    python -m app.src.pipelines.test_model mtae AT
    
### 4. Analysis training, validation, tuning, testing
    
    python -m app.src.pipelines.analyze_labels mtae AT
    python -m app.src.pipelines.analyze_training mtae AT
    python -m app.src.pipelines.analyze_tuning mtae AT
    python -m app.src.pipelines.analyze_testing mtae AT
    
### 5. Run inference
    python -m app.src.pipelines.train_model -F config mtae AT
    # CLI 
    python -m app.deployment.run_inference -d 01/01/2026 AT
    # GUI 
    docker-compose up --build

## Project structure
```
app/
├── api/              # FastAPI backend with web GUI
├── datasets/         # Raw, processed, feature-engineered dataset stages
├── deployment/       # Production models & inference as CLI
├── src/
│   ├── config/       # Set countries, model configuration & tuning search space
│   ├── data/         # Data ingestion & preprocessing
│   ├── exploration/  # Data EDA, diagnostics & visualizations
│   ├── ml/           # All DL helper incl. models, training, tuning, & analysis 
│   └── pipelines/    # Training, validation, tuning, testing & analysis workflows
└── tests/            # PyTest-based unit tests
results/              # All workflow artifacts incl. models, scalars, summaries & plots
bin/                  # Automation & maintenance tasks
```

## This project demonstrates
- Full-stack ML engineering: from API ingestion -> modeling -> deployment
- Deep Learning knowledge: custom autoencoder architectures
- MLOps workflow design: reproducible pipelines, tuning, calibration
- Software engineering best practices: modularization, testing, logging, CI/CD
- Real-world problem-solving at global production scale

## Contact
Always open for constructive criticism and code roasts, and happy to acknowledge your contribution.
For contributions, comments or collaborations, please open an issue or reach out directly.

## License
This project is open-source under the MIT License.  
You may use, modify, and distribute this software freely, subject to the terms stated in the LICENSE file.
