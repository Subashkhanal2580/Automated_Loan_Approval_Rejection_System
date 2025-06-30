# Automated Loan Approval/Rejection System

## Overview
This project implements an end-to-end machine learning pipeline for processing and analyzing financial loan data, with a focus on automated data cleaning, feature selection, and model training. The pipeline uses advanced techniques including vector embeddings, automated data cleaning, class balancing, and AutoML for loan approval decisions.

## Architecture
![Pipeline Architecture]
```
Input Data → Data Cleaning → Class Balancing → Feature Selection → AutoML Training → Model Evaluation
```

## Components

### 1. Data Cleaning Agent (`data_cleaning/data_cleaning.py`)
- Automated data cleaning using metadata-driven approach
- Integrates with Pinecone for metadata storage and retrieval
- Features:
  - Missing value detection and handling
  - Outlier detection using IQR method
  - Duplicate detection and removal
  - Automated data type inference and conversion
  - One-hot encoding for categorical variables

### 2. Class Balancing Agent (`class_imbalance_handler_Agent/class_balancer_agent.py`)
- Handles class imbalance in loan approval/rejection data
- Implements a two-stage balancing approach:
  - Stage 1: Undersampling using Tomek Links
  - Stage 2: Oversampling using SMOTE
- Configurable parameters:
  - `HUGE_IMBALANCE_THRESHOLD`: 4.0
  - `TARGET_RATIO`: 1.5

### 3. Feature Selection Agent (`feature_selection/feature_selection_agent.py`)
- Statistical feature selection using:
  - F-scores for numerical features
  - Chi-square tests for categorical features
- Metadata-driven feature type inference
- Integration with Pinecone for feature metadata storage

### 4. AutoML Predictor (`AutoML/predictor_agent.py`)
- Implements H2O AutoML for model training
- Features:
  - Automated model selection
  - Hyperparameter optimization
  - Cross-validation
  - Model evaluation and interpretation

### 5. Metadata Management (`scripts/`)
- `upload_to_pinecone.py`: Manages descriptive metadata
- `upload_statistical_metadata.py`: Handles statistical metadata

## Directory Structure
```
data_cleaning_agent/
├── AutoML/
│   └── predictor_agent.py
├── balanced_data/
│   └── application_train.csv
├── class_imbalance_handler_Agent/
│   └── class_balancer_agent.py
├── cleaned_csv/
│   └── cleaned_application_train.csv
├── data/
│   └── metadata.csv
├── data_cleaning/
│   └── data_cleaning.py
├── feature_selection/
│   └── feature_selection_agent.py
├── fintech_data/
│   ├── application_test.csv
│   └── application_train.csv
├── logs/
│   ├── cleaning_logs/
│   └── class_balancing_logs/
└── scripts/
    ├── upload_statistical_metadata.py
    └── upload_to_pinecone.py
```

## Setup and Installation

### Prerequisites
- Python 3.8+
- H2O AutoML
- Pinecone account
- Ollama
- Required Python packages (see requirements.txt)

### Environment Variables
Create a `.env` file with:
```env
PINECONE_API_KEY="your_pinecone_api_key"
OLLAMA_MODEL_NAME="nomic-embed-text"
OLLAMA_CHAT_MODEL="llama3.2:1b"
GEMINI_API_KEY="your_gemini_api_key"
GEMINI_API_KEY_1="your_backup_gemini_api_key"
```

### Installation
1. Clone the repository:
```bash
git clone <repository_url>
cd data_cleaning_agent
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Initialize Pinecone indexes:
```bash
python scripts/upload_to_pinecone.py
python scripts/upload_statistical_metadata.py
```

## Usage

### 1. Data Cleaning
```bash
cd data_cleaning
python data_cleaning.py
```

### 2. Class Balancing
```bash
cd class_imbalance_handler_Agent
python class_balancer_agent.py
```

### 3. Feature Selection
```bash
cd feature_selection
python feature_selection_agent.py
```

### 4. Model Training
```bash
cd AutoML
python predictor_agent.py
```

## Data Flow
1. Raw loan application data is loaded from `fintech_data/`
2. Cleaned data is saved to `cleaned_csv/`
3. Balanced data is stored in `balanced_data/`
4. Feature selection results are saved in respective directories
5. Final model and predictions are stored in AutoML output directory

## Logging
- All operations are logged in the `logs/` directory
- Separate log files for cleaning and class balancing operations
- Timestamp-based log file naming for traceability

## Vector Database Integration
- Uses Pinecone for metadata storage
- Two separate indexes:
  - `fintech-app-traintestfinal-metadata-index`: Descriptive metadata
  - `fintech-statisticalfinal-metadata-index`: Statistical metadata

## Error Handling
- Comprehensive error handling throughout the pipeline
- Detailed logging of errors and exceptions
- Graceful fallbacks for missing data or failed operations

## Model Features
- Handles both numerical and categorical loan features
- Automated feature importance analysis
- Model interpretability through SHAP values
- Cross-validation for robust performance estimation

## Performance Monitoring
- Automated logging of model performance metrics
- Class-wise precision and recall monitoring
- ROC-AUC and PR-AUC curve analysis
- Confusion matrix generation

## Contributing
1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## Acknowledgments
- H2O.ai for AutoML functionality
- Pinecone for vector database services
- Ollama for embedding generation
- Google for Gemini API integration