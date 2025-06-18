import pandas as pd
from pinecone import Pinecone, ServerlessSpec
from langchain_ollama import OllamaLLM, OllamaEmbeddings
import os
from dotenv import load_dotenv
import numpy as np
from datetime import datetime 

# Load environment variables from .env file
load_dotenv()

# --- Configurationx ---
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
OLLAMA_EMBEDDING_MODEL = "nomic-embed-text"
OLLAMA_CHAT_MODEL = os.getenv("OLLAMA_CHAT_MODEL")

FINTECH_DATA_DIR = "fintech_data"
CLEANED_CSV_DIR = "cleaned_csv"
LOG_FILE_DIR = "logs" # New directory for logs

# Create necessary directories if they don't exist
os.makedirs(CLEANED_CSV_DIR, exist_ok=True)
os.makedirs(LOG_FILE_DIR, exist_ok=True)

# Ensure API key and Ollama models are set
if not PINECONE_API_KEY:
    raise ValueError("PINECONE_API_KEY environment variable not set. Please set it in your .env file.")
if not OLLAMA_CHAT_MODEL:
    raise ValueError("OLLAMA_CHAT_MODEL environment variable not set. Please set it (e.g., 'llama3.2:1b') in your .env file.")

# Initialize Pinecone Client
pc = Pinecone(api_key=PINECONE_API_KEY)

# --- Initialize Ollama Models ---
print(f"Initializing OllamaEmbeddings with model: {OLLAMA_EMBEDDING_MODEL}")
ollama_embeddings = OllamaEmbeddings(model=OLLAMA_EMBEDDING_MODEL)

print(f"Initializing OllamaLLM with model: {OLLAMA_CHAT_MODEL}")
ollama_llm = OllamaLLM(model=OLLAMA_CHAT_MODEL)

# --- Define Embedding Dimension ---
embedding_dimension = 768
print(f"Using embedding dimension: {embedding_dimension} (from {OLLAMA_EMBEDDING_MODEL})")

# Helper function to get or create a Serverless index (unchanged)
def get_or_create_pinecone_serverless_index(index_name: str, dimension: int, metric: str = 'cosine', cloud: str = 'aws', region: str = 'us-west-2'):
    """
    Connects to an existing Pinecone Serverless index or creates it if it doesn't exist.
    Requires cloud and region for ServerlessSpec.
    """
    if index_name not in pc.list_indexes():
        print(f"Creating new Pinecone Serverless index: {index_name} in {cloud}-{region}...")
        try:
            pc.create_index(
                name=index_name,
                dimension=dimension,
                metric=metric,
                spec=ServerlessSpec(cloud=cloud, region=region)
            )
            print(f"Index '{index_name}' created successfully with ServerlessSpec.")
        except Exception as create_e:
            raise RuntimeError(f"Failed to create Serverless Pinecone index '{index_name}': {create_e}")
    else:
        print(f"Connecting to existing Pinecone Serverless index: {index_name}...")
    return pc.Index(index_name)

# --- Initialize Indexes ---
index_1_name = "fintech-app-traintestfinal-metadata-index"
index_2_name = "fintech-statisticalfinal-metadata-index"

print(f"Connecting to Pinecone index: {index_1_name}...")
index_1 = pc.Index(index_1_name)
print(f"Connecting to Pinecone index: {index_2_name}...")
index_2 = pc.Index(index_2_name)

print(f"Indexes '{index_1_name}' and '{index_2_name}' are initialized.")

# --- Dictionary to hold loaded and modified DataFrames ---
loaded_dfs = {}

# --- List to store cleaning log messages ---
cleaning_log = []

# --- New Helper Function to get comprehensive metadata ---
def _get_comprehensive_metadata(all_results):
    """
    Aggregates metadata for the most relevant column from all search results.
    Assumes that metadata for the same logical column (Row, Table) might be split
    across different entries in Pinecone (e.g., descriptive vs. statistical).
    """
    if not all_results:
        return None, None, None, None

    # Start with the metadata of the absolute best match
    top_match = all_results[0]
    column_id = top_match.id # Keep the ID of the top match as the primary ID
    base_metadata = top_match.metadata.copy() # Make a copy to modify

    # Extract primary identifiers from the top match
    column_name = base_metadata.get("Row")
    table_name = base_metadata.get("Table")

    if not column_name or not table_name:
        # print(f"Debug: Column name ('{column_name}') or table name ('{table_name}') not found in top match metadata. Cannot proceed with merging.")
        return None, None, None, None

    # Iterate through all results to gather all related metadata for the same logical column
    for i, match in enumerate(all_results):
        match_column_name = match.metadata.get("Row")
        match_table_name = match.metadata.get("Table")

        # If this match refers to the same logical column (same Row and Table), merge its metadata
        if match_column_name == column_name and match_table_name == table_name:
            # print(f"Debug: Merging metadata from match {i} (ID: {match.id}) with score {match.score:.4f}")
            for key, value in match.metadata.items():
                # Prioritize specific statistical keys or add new keys if not present
                if key not in base_metadata or key in ["is_null", "null_count", "is_duplicate", "duplicate_count", "is_outlier", "outlier_count", "typeofdata"]:
                    base_metadata[key] = value

        
    
    # Normalize 'datatype' key: use 'typeofdata' if available, otherwise 'datatype'
    # This ensures consistency for downstream checks (e.g., "identifier", "numerical")
    final_datatype = base_metadata.get("typeofdata") or base_metadata.get("datatype")
    if final_datatype:
        base_metadata['datatype'] = final_datatype # Ensure 'datatype' key holds the most comprehensive type

    # print(f"Debug: Final merged metadata for '{column_name}' in '{table_name}': {base_metadata}")
    # print(f"--- End Debug: _get_comprehensive_metadata ---\n")

    return column_id, column_name, table_name, base_metadata

# --- Helper Function to determine cleaning plan ---
def _determine_cleaning_plan(df: pd.DataFrame, column_name: str, column_metadata: dict):
    """
    Analyzes the DataFrame column and its metadata to propose a cleaning plan.
    """
    cleaning_actions = []
    action_descriptions = []
    reasons_for_no_action = [] # To provide more specific feedback if no actions are taken

    # Debug prints (can be removed in final version)
    # print(f"\n--- Debug: _determine_cleaning_plan for '{column_name}' ---")
    # print(f"DataFrame dType (inferred by Pandas): {df[column_name].dtype}")
    # print(f"Column Metadata provided (merged): {column_metadata}")

    # Determine column type based on Pandas inferred dtype
    is_numeric = pd.api.types.is_numeric_dtype(df[column_name])
    is_categorical = pd.api.types.is_object_dtype(df[column_name]) or pd.api.types.is_categorical_dtype(df[column_name])
    
    num_rows = len(df)
    num_nulls_calculated = df[column_name].isnull().sum()
    # print(f"Calculated nulls from DataFrame: {num_nulls_calculated}")

    # Report null count (prefer calculated, fallback to metadata if useful)
    original_nulls_for_report = column_metadata.get("null_count")
    source_of_null_count = "from metadata"
    # Fallback to calculated if metadata is missing or inconsistent with direct calculation
    if original_nulls_for_report is None or original_nulls_for_report != num_nulls_calculated: 
        original_nulls_for_report = num_nulls_calculated
        source_of_null_count = "calculated from CSV"


    # 1. Null Value Handling
    if num_nulls_calculated > 0:
        null_percentage = (num_nulls_calculated / num_rows) * 100
        action_desc = f"Detected {num_nulls_calculated} missing values ({null_percentage:.2f}%). "

        if is_numeric:
            if null_percentage < 5: # Threshold for dropping (can be customized)
                cleaning_actions.append("drop_nulls")
                action_desc += "Proposed action: Drop rows with missing values (low percentage)."
            else: # Higher percentage, impute
                cleaning_actions.append("impute_mean") # Simplifed to mean imputation by default for numerical
                action_desc += "Proposed action: Impute missing values with mean (higher percentage)."
        elif is_categorical:
            cleaning_actions.append("impute_mode_or_category")
            action_desc += "Proposed action: Impute missing values with mode or a 'Missing' category."
        action_descriptions.append(action_desc)
    else:
        reasons_for_no_action.append("No significant missing values detected.")


    # 2. Outlier Handling (for Numeric columns only)
    outliers_detected_calculated = 0 # Initialize
    if is_numeric:
        Q1 = df[column_name].quantile(0.25)
        Q3 = df[column_name].quantile(0.75)
        IQR = Q3 - Q1
        
        # If IQR is 0 (all values are the same), there are no statistical outliers
        if IQR == 0:
            reasons_for_no_action.append("Outlier check skipped: IQR is zero (all values are identical).")
        else:
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            # Count outliers using the calculated bounds from the current DataFrame
            outliers_detected_calculated = df[(df[column_name] < lower_bound) | (df[column_name] > upper_bound)].shape[0]
            # print(f"Calculated outliers (IQR method): {outliers_detected_calculated}")

            if outliers_detected_calculated > 0:
                cleaning_actions.append("handle_outliers_iqr_capping")
                action_descriptions.append(f"Detected {outliers_detected_calculated} outliers (IQR method). Proposed action: Cap outliers at {lower_bound:.2f} and {upper_bound:.2f}.")
            else:
                reasons_for_no_action.append("No significant outliers detected by IQR method.")
    else:
        reasons_for_no_action.append("Outlier check skipped: column is not numeric.")


    # 3. Duplicate Handling (Conditional on 'identifier' typeofdata from merged metadata)
    column_type_from_metadata = column_metadata.get("datatype") # This 'datatype' now holds the merged 'typeofdata' or original 'datatype'
    # print(f"Metadata 'datatype' for duplicate check (merged): {column_type_from_metadata}")
    num_duplicates_calculated = df[column_name].duplicated().sum()
    # print(f"Calculated duplicates from DataFrame: {num_duplicates_calculated}")

    if column_type_from_metadata == "identifier":
        if num_duplicates_calculated > 0:
            cleaning_actions.append("remove_duplicates_identifier")
            action_descriptions.append(f"Detected {num_duplicates_calculated} duplicates. Proposed action: Remove duplicate records as '{column_name}' is an identifier.")
        else:
            reasons_for_no_action.append("No duplicates found for identifier column (or none remaining).")
    else:
        # If is_duplicate from metadata is true but not identifier, inform the user
        if column_metadata.get("is_duplicate", False) and column_metadata.get("duplicate_count", 0) > 0:
            action_descriptions.append(f"Note: Duplicates detected ({column_metadata.get('duplicate_count', 'N/A')}) but '{column_name}' is not an 'identifier'. Ignoring duplicacy as per rule.")
        else:
            reasons_for_no_action.append("Duplicate handling skipped: column is not an identifier, or no duplicates detected.")


    # 4. Categorical to Numerical (One-Hot Encoding)
    if is_categorical and column_type_from_metadata != "identifier": # Do not encode identifier columns
        unique_values = df[column_name].nunique()
        # print(f"Categorical unique values: {unique_values}")
        # Avoid OHE for columns with too few (0-1) or too many unique values (high cardinality)
        if 2 < unique_values <= 50: # Example threshold for unique values for OHE
            cleaning_actions.append("one_hot_encode")
            action_descriptions.append(f"Detected categorical column with {unique_values} unique values. Proposed action: Apply One-Hot Encoding.")
        elif unique_values > 50:
            action_descriptions.append(f"Note: Categorical column '{column_name}' has too many unique values ({unique_values}) for standard One-Hot Encoding. Consider other encoding methods or feature engineering.")
            reasons_for_no_action.append("One-Hot Encoding skipped: high cardinality.")
        else: # unique_values is 0, 1 or NaN for some reason
            reasons_for_no_action.append("One-Hot Encoding skipped: categorical column has 0 or 1 unique values (not suitable for OHE).")
    else:
        if is_categorical: # It is categorical but not for OHE (e.g., identifier or high cardinality already noted)
            reasons_for_no_action.append("One-Hot Encoding skipped: column is categorical but not a candidate (e.g., identifier).")
        else:
            reasons_for_no_action.append("One-Hot Encoding skipped: column is not categorical.")

    if not cleaning_actions:
        action_descriptions.append("No specific cleaning actions required for this column based on analysis:")
        for reason in set(reasons_for_no_action): # Use set to avoid duplicate reasons
            action_descriptions.append(f"  - {reason}")

    return cleaning_actions, action_descriptions, original_nulls_for_report, source_of_null_count


# --- Main interactive cleaning loop ---
print("\n--- Data Cleaning Agent Started ---")

FINTECH_DATA_DIR = "fintech_data"  # directory path

# List all CSV files
csv_files = [f for f in os.listdir(FINTECH_DATA_DIR) if f.endswith(".csv")]

# Loop through each CSV file
for csv_file in csv_files:
    file_path = os.path.join(FINTECH_DATA_DIR, csv_file)
    
    print(f"\n📥 Loading file: {csv_file}")
    
    try:
        # Use na_values to handle common missing value representations
        missing_value_indicators = ['NA', 'N/A', 'NaN', 'nan', 'NULL', 'null', '', ' ']
        df = pd.read_csv(file_path, na_values=missing_value_indicators)
        loaded_dfs[csv_file] = df # Store the initial loaded DataFrame
    except Exception as e:
        print(f"Error loading '{csv_file}': {e}. Skipping this file.")
        cleaning_log.append(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] ERROR: Failed to load '{csv_file}': {e}")
        continue

    # Use the filename as the identifier
    table_name = csv_file

    print(f"Loaded {len(df)} rows and {len(df.columns)} columns from {csv_file}")
    current_df_in_memory = loaded_dfs[table_name]
    # Iterate through each column in the currently loaded DataFrame
    for column in loaded_dfs[table_name].columns: # Iterate over columns of the *current* DataFrame
        user_query = column.strip() # user_query is now the column name automatically
        print(f"\n--- Processing column: '{user_query}' in file: '{table_name}' ---")

        # No 'exit' check here as it's fully automated now
        if not user_query: # Should not happen if iterating through actual columns
            print("Warning: Empty column name encountered. Skipping.")
            cleaning_log.append(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] WARNING: Empty column name encountered in '{table_name}'. Skipping.")
            continue

        # --- Step 2: Embed User Input (which is now the column name) ---
        query_embedding = ollama_embeddings.embed_query(user_query)

        # --- Step 3: Pinecone Search ---
        print("Searching Pinecone for relevant columns...")
        # Increased top_k to gather more metadata if spread across results
        pinecone_filter = {"Table": {"$eq": table_name}}
        results_1 = index_1.query(
            vector=query_embedding,
            top_k=5, 
            include_metadata=True,
            filter=pinecone_filter
        )
        results_2 = index_2.query(
            vector=query_embedding,
            top_k=5, 
            include_metadata=True,
            filter=pinecone_filter
        )

        all_results = sorted(results_1.matches + results_2.matches, key=lambda x: x.score, reverse=True)

        if not all_results:
            print(f"No relevant columns found in Pinecone for '{user_query}'. Skipping this column.")
            cleaning_log.append(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {table_name} - {user_query}: No relevant columns found in Pinecone. Skipping.")
            continue

        # --- Step 4: Identify Column & Retrieve Comprehensive Metadata ---
        column_id, column_name, table_name_from_pinecone, column_metadata = _get_comprehensive_metadata(all_results)

        # Ensure the column found in Pinecone actually matches the current column being processed
        if not column_name or not table_name_from_pinecone or column_name != user_query: # <--- MODIFIED CONDITION
            print(f"Error: Pinecone did not return expected metadata for column '{user_query}' in '{table_name}'. Skipping.")
            cleaning_log.append(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {table_name} - {user_query}: Pinecone returned unexpected metadata or none. Skipping.")
            continue
        
        if column_name not in current_df_in_memory.columns: # <--- ADDED CHECK
            print(f"Error: Column '{column_name}' not found in {table_name} after loading/previous modifications. Skipping.")
            cleaning_log.append(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {table_name} - {column_name}: Column not found in DataFrame after modifications. Skipping.")
            continue

        print(f"\n--- Most Relevant Column Found: '{column_name}' from '{table_name}' ---")
        print("\nFull Merged Metadata for this column:")
        for key, value in column_metadata.items():
            print(f"  {key}: {value}")

        # The DataFrame for the current table should already be in `loaded_dfs`
        # and its path checked at the file loading stage.
        df_current = loaded_dfs[table_name] 

        if column_name not in df_current.columns:
            print(f"Error: Column '{column_name}' not found in {table_name} after loading. Skipping.")
            cleaning_log.append(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {table_name} - {column_name}: Column not found in DataFrame. Skipping.")
            continue
        
        # --- Step 5 & 6: Determine Cleaning Strategy based on Comprehensive Metadata & DataFrame Analysis ---
        cleaning_actions, action_descriptions, original_nulls_report, source_of_null_count = \
            _determine_cleaning_plan(df_current, column_name, column_metadata)


        print(f"\nProposed Cleaning Plan for '{column_name}':")
        print(f"- Data type (Pandas inferred): {df_current[column_name].dtype}")
        print(f"- Data type (from metadata): {column_metadata.get('datatype', 'N/A')}")
        print(f"- Original missing values ({source_of_null_count}): {original_nulls_report}")
        
        for desc in action_descriptions:
            print(f"- {desc}")

        # --- Automated Approval (User Approval removed) ---
        print(f"\nAutomatically proceeding with cleaning for column '{column_name}'...")
        log_prefix = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {table_name} - {column_name}: "

        # --- Step 7: Execute Cleaning ---
        if cleaning_actions:
            print(f"\nExecuting cleaning actions for '{column_name}'...")
            try:
                for action in cleaning_actions:
                    if action == "impute_mean":
                        if pd.api.types.is_numeric_dtype(df_current[column_name]):
                            num_nulls_before = df_current[column_name].isnull().sum()
                            mean_value = df_current[column_name].mean()
                            df_current[column_name].fillna(mean_value, inplace=True)
                            print(f"  - Action: Imputed missing values with mean ({mean_value:.2f}).")
                            cleaning_log.append(f"{log_prefix}Imputed {num_nulls_before} missing values with mean ({mean_value:.2f}).")
                        else:
                            print(f"  - Action: Cannot impute mean; '{column_name}' is not a numeric column. Skipping imputation.")
                            cleaning_log.append(f"{log_prefix}Skipped mean imputation as '{column_name}' is not numeric.")
                        
                    elif action == "drop_nulls":
                        initial_rows = len(df_current)
                        num_nulls_before = df_current[column_name].isnull().sum()
                        df_current.dropna(subset=[column_name], inplace=True)
                        rows_dropped = initial_rows - len(df_current)
                        print(f"  - Action: Dropped rows with null values ({rows_dropped} rows).")
                        cleaning_log.append(f"{log_prefix}Dropped {rows_dropped} rows due to {num_nulls_before} missing values.")
                        loaded_dfs[table_name] = df_current # Update df in loaded_dfs if rows were dropped (changes DF shape)

                    elif action == "impute_mode_or_category":
                        if pd.api.types.is_categorical_dtype(df_current[column_name]) or pd.api.types.is_object_dtype(df_current[column_name]):
                            num_nulls_before = df_current[column_name].isnull().sum()
                            if not df_current[column_name].mode().empty:
                                mode_value = df_current[column_name].mode()[0]
                                df_current[column_name].fillna(mode_value, inplace=True)
                                print(f"  - Action: Imputed missing categorical values with mode ('{mode_value}').")
                                cleaning_log.append(f"{log_prefix}Imputed {num_nulls_before} missing categorical values with mode ('{mode_value}').")
                            else:
                                print(f"  - Action: Mode could not be determined for '{column_name}'. Skipping imputation.")
                                cleaning_log.append(f"{log_prefix}Mode could not be determined for '{column_name}'. Skipping imputation.")
                        else:
                            print(f"  - Action: Skipping mode imputation for non-categorical column '{column_name}'.")
                            cleaning_log.append(f"{log_prefix}Skipped mode imputation as '{column_name}' is not categorical.")

                    elif action == "handle_outliers_iqr_capping":
                        if pd.api.types.is_numeric_dtype(df_current[column_name]):
                            Q1 = df_current[column_name].quantile(0.25)
                            Q3 = df_current[column_name].quantile(0.75)
                            IQR = Q3 - Q1
                            
                            if IQR == 0:
                                print(f"  - Action: IQR is zero for '{column_name}'. Skipping outlier capping as all values are identical.")
                                cleaning_log.append(f"{log_prefix}Skipped outlier capping: IQR is zero for '{column_name}'.")
                                continue

                            lower_bound = Q1 - 1.5 * IQR
                            upper_bound = Q3 + 1.5 * IQR

                            outliers_before = df_current[(df_current[column_name] < lower_bound) | (df_current[column_name] > upper_bound)].shape[0]
                            if outliers_before > 0:
                                df_current[column_name] = np.where(df_current[column_name] < lower_bound, lower_bound, df_current[column_name])
                                df_current[column_name] = np.where(df_current[column_name] > upper_bound, upper_bound, df_current[column_name])
                                print(f"  - Action: Capped {outliers_before} outliers in '{column_name}' using IQR method (values capped between {lower_bound:.2f} and {upper_bound:.2f}).")
                                cleaning_log.append(f"{log_prefix}Capped {outliers_before} outliers using IQR method (values capped between {lower_bound:.2f} and {upper_bound:.2f}).")
                            else:
                                print(f"  - Action: No outliers detected by IQR method for '{column_name}'.")
                                cleaning_log.append(f"{log_prefix}No outliers detected by IQR method.")
                        else:
                            print(f"  - Action: Outlier handling skipped for '{column_name}' as it is not a numeric column.")
                            cleaning_log.append(f"{log_prefix}Skipped outlier handling as '{column_name}' is not numeric.")

                    elif action == "remove_duplicates_identifier":
                        num_duplicates_before = df_current[column_name].duplicated().sum()
                        if num_duplicates_before > 0:
                            initial_rows = len(df_current)
                            df_current.drop_duplicates(subset=[column_name], inplace=True)
                            rows_dropped = initial_rows - len(df_current)
                            print(f"  - Action: Removed {rows_dropped} duplicate rows based on '{column_name}' (identifier).")
                            cleaning_log.append(f"{log_prefix}Removed {rows_dropped} duplicate rows based on identifier column '{column_name}'.")
                            loaded_dfs[table_name] = df_current # Update df in loaded_dfs if rows were dropped
                        else:
                            print(f"  - Action: No duplicates found for '{column_name}'. Skipping removal.")
                            cleaning_log.append(f"{log_prefix}No duplicates found for identifier column.")
                        
                    elif action == "one_hot_encode":
                        if pd.api.types.is_categorical_dtype(df_current[column_name]) or pd.api.types.is_object_dtype(df_current[column_name]):
                            original_cols = df_current.columns.tolist()
                            df_encoded = pd.get_dummies(df_current, columns=[column_name], prefix=column_name, dummy_na=False)
                            
                            # IMPORTANT: Update the DataFrame in loaded_dfs to include new columns
                            loaded_dfs[table_name] = df_encoded 
                            df_current = df_encoded # Update current df_current reference to the new wide DataFrame

                            new_cols = [col for col in df_current.columns if col not in original_cols and col.startswith(f"{column_name}_")]
                            print(f"  - Action: Applied One-Hot Encoding to '{column_name}'. Created new columns: {', '.join(new_cols[:3])}{'...' if len(new_cols) > 3 else ''}.")
                            cleaning_log.append(f"{log_prefix}Applied One-Hot Encoding to categorical column '{column_name}'. Created {len(new_cols)} new columns.")
                        else:
                            print(f"  - Action: Skipping One-Hot Encoding for non-categorical column '{column_name}'.")
                            cleaning_log.append(f"{log_prefix}Skipped One-Hot Encoding as '{column_name}' is not categorical.")

                print(f"Current missing values after all actions in '{column_name}': {loaded_dfs[table_name][column_name].isnull().sum()}")

            except Exception as e:
                print(f"An error occurred during cleaning '{column_name}': {e}. The DataFrame for '{table_name}' might be in an inconsistent state.")
                cleaning_log.append(f"{log_prefix}ERROR: An error occurred during cleaning: {e}")
        else:
            print(f"No specific cleaning actions were applied for '{column_name}' based on the determined strategy.")
            cleaning_log.append(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {table_name} - {column_name}: No specific cleaning actions applied (no significant issues detected).")


# --- Cleaning session finished. Save all cleaned DataFrames and logs ---
print("\n--- Cleaning session finished. Saving cleaned files and logs... ---")
if loaded_dfs:
    for table_name, df_cleaned in loaded_dfs.items():
        cleaned_file_name = f"cleaned_{table_name}"
        save_path = os.path.join(CLEANED_CSV_DIR, cleaned_file_name)
        try:
            # .to_csv() by default overwrites if file exists
            df_cleaned.to_csv(save_path, index=False)
            print(f"Saved cleaned '{table_name}' to '{save_path}' (overwritten if exists).")
        except Exception as e:
            print(f"Error saving cleaned '{table_name}' to '{save_path}': {e}")
else:
    print("No DataFrames were loaded or modified during this session.")

# Save the cleaning log to a file
log_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_filename = os.path.join(LOG_FILE_DIR, f"cleaning_log_{log_timestamp}.txt")
try:
    with open(log_filename, "w") as f:
        for entry in cleaning_log:
            f.write(entry + "\n")
    print(f"Cleaning log saved to '{log_filename}'")
except Exception as e:
    print(f"Error saving cleaning log to '{log_filename}': {e}")

print("\n--- Cleaning Log ---")
if cleaning_log:
    for entry in cleaning_log:
        print(entry)
else:
    print("No cleaning actions were logged during this session.")

print("\n--- Data Cleaning Agent Exited ---")