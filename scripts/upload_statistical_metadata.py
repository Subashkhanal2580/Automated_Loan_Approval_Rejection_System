import pandas as pd
import ollama
from pinecone import Pinecone, ServerlessSpec
import os
from dotenv import load_dotenv
import time
import numpy as np # For numerical operations, especially NaN checks

# Load environment variables
load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
OLLAMA_MODEL_NAME = os.getenv("OLLAMA_MODEL_NAME")

# Pinecone Index details for STATISTICAL metadata
STAT_INDEX_NAME = "fintech-statistical-metadata-index" # Using the same name
EMBEDDING_DIMENSION = 768 # Still using nomic-embed-text for embedding text description of stats

# Define the path to your fintech data directory
FINTECH_DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "fintech_data")

# Global dictionary to store loaded dataframes for efficiency
loaded_fintech_dfs = {}

def get_embedding(text):
    """Generates an embedding for a given text using the local Ollama model."""
    try:
        response = ollama.embeddings(model=OLLAMA_MODEL_NAME, prompt=text)
        return response['embedding']
    except Exception as e:
        print(f"Error generating embedding with Ollama for text: '{text[:50]}...': {e}")
        return None

def load_fintech_df(table_name):
    """
    Loads a fintech DataFrame into the global cache if not already loaded.
    """
    if table_name not in loaded_fintech_dfs:
        file_path = os.path.join(FINTECH_DATA_DIR, table_name)
        if not os.path.exists(file_path):
            print(f"Warning: Fintech data file '{table_name}' not found at '{file_path}'")
            loaded_fintech_dfs[table_name] = None # Mark as not found
            return None
        try:
            df = pd.read_csv(file_path)
            loaded_fintech_dfs[table_name] = df
            print(f"Loaded '{table_name}' for statistical analysis.")
            return df
        except Exception as e:
            print(f"Error loading '{table_name}' for statistical analysis: {e}")
            loaded_fintech_dfs[table_name] = None # Mark as error
            return None
    return loaded_fintech_dfs[table_name] # Return from cache

def get_high_level_column_type(column_series):
    """
    Infers a high-level data type for a Pandas Series.
    """
    if pd.api.types.is_numeric_dtype(column_series):
        # Check for potential identifiers that are numeric
        if column_series.nunique() == len(column_series.dropna()):
            return 'identifier' # Numeric, all unique (e.g., SK_ID_CURR)
        return 'numerical'
    elif pd.api.types.is_datetime64_any_dtype(column_series):
        return 'datetime'
    elif pd.api.types.is_bool_dtype(column_series):
        return 'boolean'
    elif pd.api.types.is_object_dtype(column_series) or pd.api.types.is_string_dtype(column_series):
        # Heuristic for categorical vs. text vs. identifier for string/object columns
        num_unique = column_series.nunique()
        total_rows = len(column_series)
        
        if total_rows == 0: # Handle empty series
            return 'unknown'
        
        unique_ratio = num_unique / total_rows

        # Common threshold for categorical (e.g., less than 5-10% unique values)
        CATEGORICAL_THRESHOLD_RATIO = 0.05 
        
        if num_unique == total_rows:
            return 'identifier' # String column where all values are unique (e.g., UUIDs)
        elif num_unique <= 2 and total_rows > 1: # Binary like 'M/F', 'Yes/No'
             return 'boolean_like_categorical'
        elif unique_ratio < CATEGORICAL_THRESHOLD_RATIO and num_unique > 1:
            return 'categorical'
        else:
            return 'text' # Default for free-form strings
    return 'other' # Fallback for unexpected types

def calculate_column_statistics(df, column_name):
    """
    Calculates statistical metadata for a given column.
    Includes nulls, duplicates, IQR-based outlier detection, and high-level type.
    """
    stats = {
        "null_count": 0,
        "is_null": False,
        "duplicate_count": 0,
        "is_duplicate": False,
        "outlier_count": 0,
        "is_outlier": False,
        "typeofdata": "unknown" # New key
    }

    if df is None or column_name not in df.columns:
        return stats # Return default stats if DataFrame or column is invalid

    column_series = df[column_name]

    # 1. High-level data type
    stats["typeofdata"] = get_high_level_column_type(column_series)

    # 2. Nulls
    stats["null_count"] = int(column_series.isnull().sum()) # Cast to int
    stats["is_null"] = stats["null_count"] > 0

    # 3. Duplicates (counts occurrences beyond the first)
    # This applies to all types, as value repetition is a valid stat
    stats["duplicate_count"] = int(column_series.duplicated(keep='first').sum()) # Cast to int
    stats["is_duplicate"] = stats["duplicate_count"] > 0
    
    # 4. Outliers (using IQR method for numeric columns)
    if stats["typeofdata"] == 'numerical' or stats["typeofdata"] == 'identifier': # Only for numerical/identifier types
        Q1 = column_series.quantile(0.25)
        Q3 = column_series.quantile(0.75)
        IQR = Q3 - Q1
        
        # Avoid division by zero or large bounds if IQR is 0 (e.g., all values are same)
        if IQR == 0:
            stats["outlier_count"] = 0
            stats["is_outlier"] = False
        else:
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            # Count outliers (excluding NaNs from outlier check)
            outliers = column_series[(column_series < lower_bound) | (column_series > upper_bound)]
            stats["outlier_count"] = int(len(outliers.dropna())) # Cast to int
            stats["is_outlier"] = stats["outlier_count"] > 0
    
    return stats

def upload_statistical_metadata_to_pinecone(metadata_csv_path):
    """
    Reads metadata from a CSV, calculates statistical metadata for each column,
    generates embeddings of the stats, and uploads to a dedicated Pinecone index.
    """
    df_metadata = pd.read_csv(metadata_csv_path)

    pc = Pinecone(api_key=PINECONE_API_KEY)

    if STAT_INDEX_NAME not in pc.list_indexes().names():
        print(f"Creating Pinecone index: {STAT_INDEX_NAME}...")
        pc.create_index(
            name=STAT_INDEX_NAME,
            dimension=EMBEDDING_DIMENSION,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )
        print("Index created successfully.")
    else:
        print(f"Index '{STAT_INDEX_NAME}' already exists. Connecting to it.")

    index = pc.Index(STAT_INDEX_NAME)

    vectors_to_upsert = []
    BATCH_SIZE = 100

    print("Processing metadata and calculating statistics for fintech CSVs...")
    for i, row in df_metadata.iterrows():
        table_name = row['Table']
        column_name = row['Row']

        # Load the relevant fintech DataFrame (or get from cache)
        current_fintech_df = load_fintech_df(table_name)
        
        if current_fintech_df is None:
            print(f"Skipping row {i} for '{table_name}.{column_name}' due to DataFrame loading error for statistical analysis.")
            continue

        # Calculate statistical properties
        col_stats = calculate_column_statistics(current_fintech_df, column_name)

        # --- Prepare text for embedding from statistical metadata ---
        # Include all statistical metrics and the new typeofdata in the text for semantic searchability
        text_to_embed = (
            f"Statistics for Table: {table_name}, Column: {column_name}. "
            f"Type of Data: {col_stats['typeofdata']}. "
            f"Nulls: {col_stats['null_count']} (Presence: {col_stats['is_null']}). "
            f"Duplicates: {col_stats['duplicate_count']} (Presence: {col_stats['is_duplicate']}). "
            f"Outliers: {col_stats['outlier_count']} (Presence: {col_stats['is_outlier']})."
        )
        
        embedding = get_embedding(text_to_embed)

        if embedding:
            vector_id = f"{table_name}-{column_name}-stats" # Unique ID for statistical index
            vector_id = vector_id.replace('.csv', '').replace('.', '_').replace('/', '_')

            # Prepare metadata for Pinecone (all statistical fields + original identifiers)
            metadata = {
                "Table": table_name,
                "Row": column_name,
                "typeofdata": col_stats['typeofdata'], # NEW: typeofdata
                "null_count": col_stats['null_count'],
                "is_null": col_stats['is_null'],
                "duplicate_count": col_stats['duplicate_count'],
                "is_duplicate": col_stats['is_duplicate'],
                "outlier_count": col_stats['outlier_count'],
                "is_outlier": col_stats['is_outlier']
            }

            vectors_to_upsert.append({
                "id": vector_id,
                "values": embedding,
                "metadata": metadata
            })

            if len(vectors_to_upsert) >= BATCH_SIZE:
                print(f"Upserting a batch of {len(vectors_to_upsert)} statistical vectors...")
                try:
                    index.upsert(vectors=vectors_to_upsert)
                except Exception as upsert_e:
                    print(f"Error during batch upsert for statistical data: {upsert_e}")
                finally:
                    vectors_to_upsert = []
        else:
            print(f"Skipping row {i} for '{table_name}.{column_name}' due to embedding error for statistical analysis.")

    if vectors_to_upsert:
        print(f"Upserting final batch of {len(vectors_to_upsert)} statistical vectors...")
        try:
            index.upsert(vectors=vectors_to_upsert)
            print("Final batch of statistical data upserted successfully.")
        except Exception as final_upsert_e:
            print(f"Error during final batch upsert for statistical data: {final_upsert_e}")
    else:
        print("No statistical vectors to upsert.")

    print("Waiting a few seconds for indexing of statistical data to complete before checking stats...")
    time.sleep(5)

    print("\nPinecone statistical index stats:")
    print(index.describe_index_stats())

if __name__ == "__main__":
    metadata_csv_file = os.path.join(os.path.dirname(__file__), "..", "data", "metadata.csv")
    upload_statistical_metadata_to_pinecone(metadata_csv_file)