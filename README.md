# AnamolyDetection
EVT Bayesian Framework Anamoly Detection
AnamolyDetection/
├── data/
│   ├── raw/              # Raw OHLCV parquet files
│   ├── processed/        # Log-returns with regime labels
│   └── residuals/        # TimesFM residuals
├── src/
│   ├── data_pipeline.py  # Download, clean, compute log-returns
│   ├── timesfm_runner.py # Rolling-window TimesFM inference
│   ├── stationarity.py   # ADF, KPSS, Ljung-Box, ARCH-LM tests
│   ├── gpd_mle.py        # scipy MLE GPD fitting
│   ├── gpd_bayesian.py   # PyMC Bayesian GPD with 3 prior specs
│   ├── anomaly_scorer.py # Posterior predictive tail probability
│   ├── baselines.py      # Isolation Forest, One-Class SVM, pure TSLM
│   └── evaluation.py     # Precision, Recall, F1, PR-AUC
├── notebooks/            # Exploratory analysis
├── configs/              # Hyperparameter YAML files
├── results/              # Tables, figures, saved posteriors
└── tests/                # Unit tests for pipeline
