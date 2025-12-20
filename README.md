# E-Commerce Customer Churn Prediction System

> A production-ready machine learning system for predicting customer churn in e-commerce platforms, featuring a complete ML pipeline, REST API, and interactive dashboard.

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red)
![MLflow](https://img.shields.io/badge/MLflow-2.8+-orange)
![License](https://img.shields.io/badge/License-MIT-green)

## Project Overview

This project implements an end-to-end machine learning solution for predicting customer churn in e-commerce. It goes beyond a simple Jupyter notebook by providing:

- **Production ML Pipeline**: Modular, maintainable code with proper software engineering practices
- **Experiment Tracking**: MLflow integration for tracking experiments, parameters, and metrics
- **REST API**: FastAPI backend for real-time and batch predictions
- **Interactive Dashboard**: Streamlit-based UI for business users
- **Model Explainability**: SHAP-based feature importance and prediction explanations
- **Database Layer**: SQLAlchemy-based storage for predictions and feedback

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     CHURN PREDICTION SYSTEM                             │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌──────────────┐     ┌──────────────┐     ┌──────────────┐           │
│  │   Streamlit  │────▶│   FastAPI    │────▶│   ML Models  │           │
│  │  Dashboard   │     │   Backend    │     │   (XGBoost)  │           │
│  └──────────────┘     └──────────────┘     └──────────────┘           │
│         │                    │                    │                    │
│         │                    ▼                    │                    │
│         │             ┌──────────────┐           │                    │
│         │             │   Database   │           │                    │
│         │             │   (SQLite)   │           │                    │
│         │             └──────────────┘           │                    │
│         │                                        │                    │
│         ▼                                        ▼                    │
│  ┌──────────────┐                         ┌──────────────┐           │
│  │   MLflow     │                         │    SHAP      │           │
│  │   Tracking   │                         │  Explainer   │           │
│  └──────────────┘                         └──────────────┘           │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

## Features

### Machine Learning
- Multiple algorithms: Logistic Regression, Random Forest, XGBoost, LightGBM, CatBoost
- Ensemble methods (Voting, Stacking)
- Hyperparameter tuning with Optuna
- Class imbalance handling
- Cross-validation

### Feature Engineering
- Tenure-based features
- Engagement metrics
- Value-based features
- Interaction features
- Ratio features

### Model Explainability
- SHAP summary plots
- Feature importance rankings
- Individual prediction explanations
- Risk factor identification

### Production Features
- RESTful API with FastAPI
- Interactive Streamlit dashboard
- SQLite database for predictions
- Docker support
- MLflow experiment tracking

## Project Structure

```
churn-prediction/
├── config/
│   ├── __init__.py          # Configuration loader
│   └── config.yaml          # Main configuration file
├── data/
│   ├── raw/                 # Raw data files
│   ├── processed/           # Processed data files
│   └── external/            # External data sources
├── models/
│   ├── saved/               # Trained model files
│   └── mlflow/              # MLflow tracking data
├── notebooks/
│   ├── 01_eda.ipynb         # Exploratory Data Analysis
│   └── 02_training.ipynb    # Model Training & Evaluation
├── reports/
│   └── figures/             # Generated visualizations
├── scripts/
│   ├── train.py             # Training script
│   ├── run_api.py           # API server script
│   └── run_dashboard.py     # Dashboard script
├── src/
│   ├── api/
│   │   ├── main.py          # FastAPI application
│   │   ├── schemas.py       # Pydantic models
│   │   └── database.py      # Database operations
│   ├── dashboard/
│   │   └── app.py           # Streamlit dashboard
│   ├── data/
│   │   ├── data_loader.py   # Data loading utilities
│   │   └── preprocessor.py  # Data preprocessing
│   ├── features/
│   │   └── feature_engineer.py  # Feature engineering
│   ├── models/
│   │   ├── trainer.py       # Model training with MLflow
│   │   ├── evaluator.py     # Model evaluation
│   │   └── explainer.py     # SHAP explanations
│   └── utils/
│       └── helpers.py       # Utility functions
├── tests/                   # Unit tests
├── Dockerfile               # Docker configuration
├── docker-compose.yml       # Docker Compose setup
├── requirements.txt         # Python dependencies
└── README.md
```

## Quick Start

### Prerequisites
- Python 3.10+
- pip or conda

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/churn-prediction.git
cd churn-prediction
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Add your dataset**
Place your e-commerce dataset in `data/raw/` as `ecommerce_customer_churn.csv`

### Running the Project

#### Option 1: Jupyter Notebooks (Development)
```bash
jupyter notebook notebooks/
```
- Run `01_eda.ipynb` for exploratory data analysis
- Run `02_training.ipynb` for model training

#### Option 2: Command Line Training
```bash
# Train all models
python scripts/train.py --model all --ensemble --shap

# Train specific model with tuning
python scripts/train.py --model xgboost --tune --tune-trials 50
```

#### Option 3: Run the Full System
```bash
# Terminal 1: Start API server
python scripts/run_api.py --port 8000 --reload

# Terminal 2: Start Dashboard
python scripts/run_dashboard.py --port 8501
```

Then open:
- Dashboard: http://localhost:8501
- API Docs: http://localhost:8000/docs

#### Option 4: Docker
```bash
docker-compose up --build
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | API information |
| `/health` | GET | Health check |
| `/predict` | POST | Single prediction |
| `/predict/batch` | POST | Batch predictions |
| `/predictions/history` | GET | Prediction history |
| `/predictions/statistics` | GET | Aggregate statistics |
| `/model/info` | GET | Model information |
| `/model/reload` | POST | Reload model |
| `/feedback/{id}` | POST | Add prediction feedback |

### Example API Request
```python
import requests

customer = {
    "Tenure": 12,
    "WarehouseToHome": 15.5,
    "HourSpendOnApp": 3.0,
    "NumberOfDeviceRegistered": 4,
    "NumberOfAddress": 2,
    "OrderAmountHikeFromlastYear": 15.0,
    "CouponUsed": 2,
    "OrderCount": 5,
    "DaySinceLastOrder": 7,
    "CashbackAmount": 150.0,
    "PreferredLoginDevice": "Mobile Phone",
    "CityTier": 1,
    "PreferredPaymentMode": "Debit Card",
    "Gender": "Male",
    "PreferedOrderCat": "Laptop & Accessory",
    "SatisfactionScore": 3,
    "MaritalStatus": "Single",
    "Complain": 0
}

response = requests.post("http://localhost:8000/predict", json=customer)
print(response.json())
```

## Model Performance

| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
|-------|----------|-----------|--------|----------|---------|
| Logistic Regression | 0.85 | 0.72 | 0.68 | 0.70 | 0.88 |
| Random Forest | 0.91 | 0.85 | 0.78 | 0.81 | 0.94 |
| XGBoost | 0.93 | 0.88 | 0.82 | 0.85 | 0.96 |
| LightGBM | 0.92 | 0.87 | 0.80 | 0.83 | 0.95 |
| Ensemble | 0.94 | 0.89 | 0.84 | 0.86 | 0.97 |

*Note: These are example metrics. Actual performance depends on your dataset.*

## Technologies Used

### Core
- **Python 3.10+**: Programming language
- **Pandas/NumPy**: Data manipulation
- **Scikit-learn**: ML algorithms and preprocessing
- **XGBoost/LightGBM/CatBoost**: Gradient boosting models

### MLOps
- **MLflow**: Experiment tracking and model registry
- **Optuna**: Hyperparameter optimization
- **SHAP**: Model explainability

### Web
- **FastAPI**: REST API framework
- **Streamlit**: Interactive dashboard
- **SQLAlchemy**: Database ORM
- **Pydantic**: Data validation

### Visualization
- **Plotly**: Interactive charts
- **Matplotlib/Seaborn**: Static plots

### Deployment
- **Docker**: Containerization
- **Docker Compose**: Multi-container orchestration

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Author

**Muhammad Abdullah**
- GitHub: [@abdullahxdev](https://github.com/abdullahxdev)
- LinkedIn: [Muhammad Abdullah](https://linkedin.com/in/mabdullahxdev)
- Email: abdullahisdev@gmail.com

## Acknowledgments

- E-commerce dataset providers
- Scikit-learn, XGBoost, and MLflow communities
- FastAPI and Streamlit teams

---

*Built with dedication for learning ML engineering best practices.*
