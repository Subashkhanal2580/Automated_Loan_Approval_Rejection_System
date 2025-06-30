import h2o
from h2o.automl import H2OAutoML
import os
from langchain.agents import AgentExecutor, create_react_agent
from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.tools import Tool
from typing import List, Dict, Any
import pandas as pd
import logging
import uuid

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize Google API keys for Gemini models (replace with your actual keys)
GEMINI_API_KEYS = [
    os.getenv("GEMINI_API_KEY"),
    os.getenv("GEMINI_API_KEY_1")
]

# Validate API keys
for i, key in enumerate(GEMINI_API_KEYS):
    if key == f"your-{i+1}-gemini-api-key":
        logger.warning(f"API key {i+1} is not set. Please provide a valid key.")

# Initialize H2O environment
def initialize_h2o() -> bool:
    try:
        h2o.init()
        logger.info("H2O environment initialized successfully")
        return True
    except Exception as e:
        logger.error(f"Failed to initialize H2O environment: {e}")
        return False

# Load CSV files from directory
def load_data(directory: str) -> List[h2o.H2OFrame]:
    try:
        total_data = []
        for filename in os.listdir(directory):
            if filename.endswith('.csv'):
                filepath = os.path.join(directory, filename)
                df = h2o.import_file(filepath)
                total_data.append(df)
                logger.info(f"Loaded: {filename}")
        return total_data
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        return []

# Preprocess data
def preprocess_data(df: h2o.H2OFrame, target: str) -> tuple:
    try:
        if df is None or len(df) == 0:
            raise ValueError("Input DataFrame is empty or None")
        features = [col for col in df.columns if col != target]
        df[target] = df[target].asfactor()
        logger.info(f"Column {target} converted to categorical")
        train, test = df.split_frame(ratios=[0.8], seed=42)
        logger.info(f"Data split: train={train.nrows}, test={test.nrows}")
        return features, train, test
    except Exception as e:
        logger.error(f"Data preprocessing failed: {e}")
        return None, None, None

# Train H2O AutoML model
def train_automl(features: List[str], target: str, train: h2o.H2OFrame, sort_metric: str = "AUC") -> H2OAutoML:
    try:
        if not features or train is None:
            raise ValueError("Invalid features or training data")
        valid_metrics = ["AUC", "logloss", "mean_per_class_error", "rmse", "mse", "precision", "recall"]
        if sort_metric not in valid_metrics:
            logger.warning(f"Invalid sort_metric '{sort_metric}'. Defaulting to AUC.")
            sort_metric = "AUC"
        
        auto_ml = H2OAutoML(
            max_models=5,
            seed=42,
            balance_classes=False,
            max_runtime_secs=600,
            exclude_algos=['DeepLearning', 'StackedEnsemble'],
            nfolds=4,
            sort_metric=sort_metric,
            verbosity='info'
        )
        auto_ml.train(x=features, y=target, training_frame=train)
        logger.info(f"AutoML training completed with sort_metric={sort_metric}")
        return auto_ml
    except Exception as e:
        logger.error(f"AutoML training failed: {e}")
        return None

# Evaluate model
def evaluate_model(auto_ml: H2OAutoML, test: h2o.H2OFrame) -> Dict[str, Any]:
    try:
        if auto_ml is None or test is None:
            raise ValueError("Invalid AutoML model or test data")
        predictions = auto_ml.leader.predict(test)
        performance = auto_ml.leader.model_performance(test_data=test)
        logger.info("Model evaluation completed")
        
        # Extract key metrics
        metrics = {
            "auc": performance.auc(),
            "precision": performance.precision()[0][0] if performance.precision() else None,
            "recall": performance.recall()[0][0] if performance.recall() else None,
            "f1": performance.F1()[0][0] if performance.F1() else None,
            "logloss": performance.logloss(),
            "aucpr": performance.aucpr()
        }
        
        return {
            "leaderboard": auto_ml.leaderboard.as_data_frame().to_dict(),
            "predictions": predictions.as_data_frame().to_dict(),
            "performance": str(performance),
            "metrics": metrics
        }
    except Exception as e:
        logger.error(f"Model evaluation failed: {e}")
        return {}

# Extract best model and confusion matrix
def get_best_model_info(auto_ml: H2OAutoML) -> Dict[str, Any]:
    try:
        if auto_ml is None:
            raise ValueError("Invalid AutoML model")
        leaderboard = auto_ml.leaderboard.as_data_frame()
        best_model_name = leaderboard['model_id'][0] if not leaderboard.empty else None
        if best_model_name:
            best_model = h2o.get_model(best_model_name)
            performance = best_model.model_performance()
            confusion_matrix = performance.confusion_matrix().table.as_data_frame().to_dict()
            logger.info(f"Best model: {best_model_name}")
            return {
                "best_model_name": best_model_name,
                "confusion_matrix": confusion_matrix
            }
        else:
            logger.warning("No models found in leaderboard")
            return {}
    except Exception as e:
        logger.error(f"Failed to extract best model info: {e}")
        return {}

# Explain model
def explain_model(auto_ml: H2OAutoML, test: h2o.H2OFrame) -> Dict[str, Any]:
    try:
        if auto_ml is None or test is None:
            raise ValueError("Invalid AutoML model or test data")
        best_model = h2o.get_model(auto_ml.leaderboard.as_data_frame()['model_id'][0])
        
        # Feature importance (if available)
        feature_importance = {}
        if hasattr(best_model, 'varimp'):
            varimp = best_model.varimp(use_pandas=True)
            if varimp is not None:
                feature_importance = varimp.to_dict()

        # SHAP explanations
        explanation = h2o.explain(best_model, test, render=False)
        shap_summary = explanation.get('shap_summary', None)
        shap_dict = shap_summary.data().as_data_frame().to_dict() if shap_summary else {}

        logger.info("Model explainability completed")
        return {
            "feature_importance": feature_importance,
            "shap_summary": shap_dict
        }
    except Exception as e:
        logger.error(f"Model explainability failed: {e}")
        return {}

# Fallback wrapper for Gemini models
def get_gemini_model(api_key: str, model_name: str = "gemini-1.5-pro"):
    try:
        model = ChatGoogleGenerativeAI(model=model_name, google_api_key=api_key)
        logger.info(f"Successfully initialized Gemini model with key ending in {api_key[-4:]}")
        return model
    except Exception as e:
        logger.error(f"Failed to initialize Gemini model with key ending in {api_key[-4:]}: {e}")
        return None

# Create LangChain tools
tools = [
    Tool(name="initialize_h2o", func=initialize_h2o, description="Initialize H2O environment"),
    Tool(name="load_data", func=load_data, description="Load CSV files from a directory"),
    Tool(name="preprocess_data", func=preprocess_data, description="Preprocess data including feature selection and train-test split"),
    Tool(name="train_automl", func=train_automl, description="Train H2O AutoML model"),
    Tool(name="evaluate_model", func=evaluate_model, description="Evaluate the trained model and return performance metrics"),
    Tool(name="get_best_model_info", func=get_best_model_info, description="Extract the name of the best performing model and its confusion matrix"),
    Tool(name="explain_model", func=explain_model, description="Generate feature importance and SHAP explanations for the best model")
]

# Create prompt template with explicit tool names
tool_names = ", ".join([tool.name for tool in tools])
prompt_template = PromptTemplate(
    input_variables=["input", "agent_scratchpad", "tool_names", "tools"],
    template="""You are an AI assistant managing an H2O AutoML workflow. 
    Your task is to automate the process of initializing H2O, loading data, preprocessing, training, evaluating the model, extracting the best model information, and generating model explanations.
    The data directory is './balanced_data' and the target column is 'TARGET'.
    
    Available tools: {tool_names}
    Tools details: {tools}
    
    Execute the tools in this order:
    1. initialize_h2o
    2. load_data
    3. preprocess_data
    4. train_automl
    5. evaluate_model
    6. get_best_model_info
    7. explain_model
    
    If any step fails, log the error and continue with the next step if possible.
    Return the final evaluation results including leaderboard, predictions, performance metrics, best model name, confusion matrix, feature importance, and SHAP explanations.
    
    Input: {input}
    Agent Scratchpad: {agent_scratchpad}
    """
)

# Main function to run the workflow
def run_automl_workflow(data_dir: str = "./balanced_data", target: str = "TARGET", sort_metric: str = "AUC"):
    results = {}
    current_model_index = 0
    loaded_data = None
    features = None
    train = None
    test = None
    auto_ml = None

    while current_model_index < len(GEMINI_API_KEYS):
        llm = get_gemini_model(GEMINI_API_KEYS[current_model_index])
        if llm is None:
            logger.warning(f"Failed to initialize Gemini model {current_model_index + 1}. Trying next model...")
            current_model_index += 1
            continue

        try:
            # Create LangChain agent
            agent = create_react_agent(
                llm=llm,
                tools=tools,
                prompt=prompt_template.partial(
                    tool_names=tool_names,
                    tools=str([tool.description for tool in tools])
                )
            )
            agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

            # Run individual steps to maintain state
            if not initialize_h2o():
                raise Exception("H2O initialization failed")

            loaded_data = load_data(data_dir)
            if not loaded_data:
                raise Exception("Data loading failed")
            df = loaded_data[0]  # Assuming single CSV for simplicity

            features, train, test = preprocess_data(df, target)
            if features is None or train is None or test is None:
                raise Exception("Data preprocessing failed")

            auto_ml = train_automl(features, target, train, sort_metric=sort_metric)
            if auto_ml is None:
                raise Exception("AutoML training failed")

            results = evaluate_model(auto_ml, test)
            if not results:
                raise Exception("Model evaluation failed")

            best_model_info = get_best_model_info(auto_ml)
            results.update(best_model_info)

            explain_results = explain_model(auto_ml, test)
            results.update(explain_results)

            logger.info("Workflow completed successfully")
            break
        except Exception as e:
            logger.error(f"Workflow failed with Gemini model {current_model_index + 1}: {e}")
            current_model_index += 1
            if current_model_index == len(GEMINI_API_KEYS):
                logger.error("All Gemini models failed. Workflow aborted.")
                return {}

    return results

if __name__ == "__main__":
    # Generate unique artifact ID
    artifact_id = str(uuid.uuid4())
    
    # Prompt user for sort metric
    sort_metric = input("Enter sort metric (e.g., AUC, precision, recall, logloss, mean_per_class_error, rmse, mse): ")
    
    # Run the workflow
    results = run_automl_workflow(sort_metric=sort_metric)
    
    # Print results
    if results:
        print("AutoML Workflow Results:")
        print(f"Leaderboard:\n{results.get('leaderboard', 'No leaderboard available')}")
        print(f"Performance:\n{results.get('performance', 'No performance metrics available')}")
        print(f"Metrics:\n{results.get('metrics', 'No metrics available')}")
        print(f"Best Model Name:\n{results.get('best_model_name', 'No best model identified')}")
        print(f"Confusion Matrix:\n{results.get('confusion_matrix', 'No confusion matrix available')}")
        print(f"Feature Importance:\n{results.get('feature_importance', 'No feature importance available')}")
        print(f"SHAP Summary:\n{results.get('shap_summary', 'No SHAP summary available')}")
    else:
        print("Workflow failed to produce results")