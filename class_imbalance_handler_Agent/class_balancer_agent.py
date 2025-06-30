import os
import pandas as pd
from collections import Counter
from imblearn.under_sampling import TomekLinks
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from autogen import AssistantAgent, UserProxyAgent
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()
gemini_api_key = os.getenv("GEMINI_API_KEY")

# --- Configuration ---
# Directory for input CSVs
INPUT_DIR = 'cleaned_csv'
# Directory for output balanced CSVs
OUTPUT_DIR = 'balanced_data'
# The name of the target variable column
TARGET_COLUMN = 'TARGET'
# Threshold to trigger Tomek Links. Applied if (majority/minority) > this value.
HUGE_IMBALANCE_THRESHOLD = 4.0
# The desired final ratio after balancing.
TARGET_RATIO = 1.5

# --- Agent Configuration ---
config_list = [
    {
        "model": "gemini-1.5-flash",
        "api_key": gemini_api_key,
        "api_type": "google"
    }
]

# ==================================
# AGENT DEFINITIONS WITH REFACTORED SYSTEM MESSAGES
# ==================================

# --- Agent 1: DataLoader ---
# Responsibility: Loads, cleans, and prepares the initial dataset.
data_loader_agent = AssistantAgent(
    name="DataLoader",
    llm_config={"config_list": config_list},
    system_message=(
        "You are the starting point of the data processing pipeline. "
        f"Your role is to load the first available CSV file from the '{INPUT_DIR}' directory. "
        "You must ensure the file and the target column ('TARGET') exist. "
        "You will then impute missing values across all feature columns using the 'most_frequent' strategy "
        "and apply label encoding to any columns with object data types. "
        "Finally, you will separate the data into a feature matrix (X) and a target vector (y)."
    )
)

# --- Agent 2: ImbalanceDetector ---
# Responsibility: Measures the class imbalance, providing the data for decision-making.
imbalance_detector_agent = AssistantAgent(
    name="ImbalanceDetector",
    llm_config={"config_list": config_list},
    system_message=(
        "You are a specialist in data diagnostics. Your critical function is to analyze the target vector (y). "
        "You must accurately compute the count for each class and calculate the imbalance ratio, "
        "defined as (majority class count / minority class count). "
        "The workflow's direction depends entirely on the precise metrics you provide."
    )
)

# --- Agent 3: DataBalancer ---
# Responsibility: Conditionally applies undersampling and oversampling techniques.
data_balancer_agent = AssistantAgent(
    name="DataBalancer",
    llm_config={"config_list": config_list},
    system_message=(
        "You are an expert data balancing agent that executes a conditional, two-stage process. "
        f"**Stage 1 (Undersampling):** You will first be informed if a 'huge' imbalance (ratio > {HUGE_IMBALANCE_THRESHOLD}) exists. "
        "If it does, you MUST apply the Tomek Links algorithm to remove noisy and borderline majority class instances. "
        "If not, you will stand by and pass the data to the next stage without modification. "
        f"**Stage 2 (Oversampling):** After the optional undersampling stage, you will check the new imbalance ratio. "
        f"If the ratio is still greater than the target of {TARGET_RATIO}, you MUST apply SMOTE to oversample the minority class. "
        "Your goal is to engineer a final dataset where the imbalance ratio is as close to 1.5 as possible. "
        "You will then output the final, balanced X and y."
    )
)

# --- Agent 4: DataSaver ---
# Responsibility: Saves the final, processed dataset to disk.
data_saver_agent = AssistantAgent(
    name="DataSaver",
    llm_config={"config_list": config_list},
    system_message=(
        "You are the final agent in the workflow. Your sole responsibility is to receive the balanced feature matrix (X) and target vector (y). "
        "You will combine them back into a single pandas DataFrame, preserving the original column names. "
        f"You must then save this DataFrame as a CSV file to the '{OUTPUT_DIR}' directory, "
        "prefixing the original filename with 'balanced_' to signify completion."
    )
)

# ==================================
# WORKFLOW FUNCTIONS
# ==================================

def load_and_prepare_data(directory: str) -> tuple[pd.DataFrame, pd.Series, list, str]:
    """Loads, imputes, and encodes the dataset."""
    if not os.path.exists(directory):
        raise FileNotFoundError(f"Input directory '{directory}' does not exist.")
    
    files = [f for f in os.listdir(directory) if f.endswith('.csv')]
    if not files:
        raise FileNotFoundError(f"No CSV files found in '{directory}' directory.")
    
    csv_file_name = files[0]
    file_path = os.path.join(directory, csv_file_name)
    df = pd.read_csv(file_path)

    if TARGET_COLUMN not in df.columns:
        raise KeyError(f"'{TARGET_COLUMN}' column not found in {csv_file_name}.")
        
    X = df.drop(TARGET_COLUMN, axis=1)
    y = df[TARGET_COLUMN]
    original_cols = X.columns.tolist()

    # Encode categorical features
    for col in X.select_dtypes(include='object').columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))

    # Impute missing values
    imputer = SimpleImputer(strategy='most_frequent')
    X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=original_cols)
    
    print(f"Successfully loaded and prepared data from '{csv_file_name}'. Shape: {df.shape}")
    return X_imputed, y, original_cols, csv_file_name

def detect_imbalance(y: pd.Series) -> tuple[Counter, float]:
    """Computes class distribution and imbalance ratio."""
    counts = Counter(y)
    if len(counts) < 2:
        return counts, 1.0 # Data is perfectly balanced or has only one class
    
    majority_count = max(counts.values())
    minority_count = min(counts.values())
    
    ratio = majority_count / minority_count if minority_count > 0 else float('inf')
    return counts, ratio

def apply_tomek_links(X: pd.DataFrame, y: pd.Series) -> tuple[pd.DataFrame, pd.Series]:
    """Applies Tomek Links to undersample the majority class."""
    print("Applying Tomek Links for undersampling...")
    tl = TomekLinks(sampling_strategy='majority')
    X_resampled, y_resampled = tl.fit_resample(X, y)
    print(f"Removed {len(X) - len(X_resampled)} samples with Tomek Links.")
    return X_resampled, y_resampled

def apply_smote(X: pd.DataFrame, y: pd.Series, target_ratio: float) -> tuple[pd.DataFrame, pd.Series]:
    """Applies SMOTE to oversample the minority class to a target ratio."""
    print(f"Applying SMOTE to achieve a target ratio of ~{target_ratio}...")
    counts = Counter(y)
    minority_class_label = min(counts, key=counts.get)
    majority_count = max(counts.values())
    
    # Calculate the desired number of minority samples
    desired_minority_samples = int(majority_count / target_ratio)
    
    if counts[minority_class_label] >= desired_minority_samples:
        print("Minority class already has sufficient samples. Skipping SMOTE.")
        return X,y

    sampling_strategy = {minority_class_label: desired_minority_samples}
    smote = SMOTE(sampling_strategy=sampling_strategy, random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)
    
    print(f"Generated {desired_minority_samples - counts[minority_class_label]} new minority samples with SMOTE.")
    return X_resampled, y_resampled

def save_balanced_data(X_balanced: pd.DataFrame, y_balanced: pd.Series, columns: list, original_filename: str):
    """Saves the final DataFrame to a new CSV file."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Reconstruct the DataFrame
    balanced_df = pd.DataFrame(X_balanced, columns=columns)
    balanced_df[TARGET_COLUMN] = y_balanced.values
    
    output_filename = f"balanced_{original_filename}"
    output_path = os.path.join(OUTPUT_DIR, output_filename)
    
    balanced_df.to_csv(output_path, index=False)
    print(f"Successfully saved balanced dataset to '{output_path}'.")


# ==================================
# MAIN WORKFLOW EXECUTION
# ==================================
def run_balancing_workflow():
    """Orchestrates the entire conditional data balancing process."""
    print("--- Starting Disciplined Class Balancing Workflow ---")
    
    try:
        # 1. DataLoader Agent's Task
        X, y, columns, filename = load_and_prepare_data(INPUT_DIR)
        
        # 2. ImbalanceDetector Agent's Task (Initial Check)
        initial_counts, initial_ratio = detect_imbalance(y)
        print(f"\nInitial Class Distribution: {initial_counts}")
        print(f"Initial Imbalance Ratio: {initial_ratio:.2f}")

        X_processed, y_processed = X, y

        # 3. DataBalancer Agent's Task (Stage 1: Conditional Tomek Links)
        if initial_ratio > HUGE_IMBALANCE_THRESHOLD:
            print(f"\nImbalance ratio ({initial_ratio:.2f}) > threshold ({HUGE_IMBALANCE_THRESHOLD}).")
            X_processed, y_processed = apply_tomek_links(X, y)
            
            # Re-detect imbalance after Tomek
            post_tomek_counts, post_tomek_ratio = detect_imbalance(y_processed)
            print(f"Post-Tomek Class Distribution: {post_tomek_counts}")
            print(f"Post-Tomek Imbalance Ratio: {post_tomek_ratio:.2f}")
        else:
            print(f"\nInitial imbalance ratio ({initial_ratio:.2f}) is within the threshold. Skipping Tomek Links.")

        # 4. DataBalancer Agent's Task (Stage 2: Conditional SMOTE)
        current_counts, current_ratio = detect_imbalance(y_processed)
        if current_ratio > TARGET_RATIO:
            print(f"\nCurrent ratio ({current_ratio:.2f}) > target ({TARGET_RATIO}).")
            X_final, y_final = apply_smote(X_processed, y_processed, TARGET_RATIO)
        else:
            print(f"\nCurrent ratio ({current_ratio:.2f}) is within the target. No SMOTE needed.")
            X_final, y_final = X_processed, y_processed
            
        # 5. ImbalanceDetector Agent's Task (Final Check)
        final_counts, final_ratio = detect_imbalance(y_final)
        print("\n--- Final Results ---")
        print(f"Final Class Distribution: {final_counts}")
        print(f"Final Imbalance Ratio: {final_ratio:.2f}")
        print(f"Final dataset shape: {X_final.shape}")

        # 6. DataSaver Agent's Task
        save_balanced_data(X_final, y_final, columns, filename)

    except (FileNotFoundError, KeyError) as e:
        print(f"\nERROR: A critical error occurred. {e}")
        print("Workflow terminated.")

    print("\n--- Workflow Complete ---")

if __name__ == "__main__":
    run_balancing_workflow()
