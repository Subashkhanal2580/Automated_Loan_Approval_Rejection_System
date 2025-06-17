import pandas as pd
import ollama
from pinecone import Pinecone, ServerlessSpec
import os
from dotenv import load_dotenv
import time
import numpy as np # For checking pd.isna

# Load environment variables
load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
OLLAMA_MODEL_NAME = os.getenv("OLLAMA_MODEL_NAME")

# --- Configuration for application_train.csv centric processing ---
TARGET_FINTECH_FILE = "application_train.csv"
INDEX_NAME = "fintech-app-train-metadata-index" # index name
EMBEDDING_DIMENSION = 768

# Define the path to your fintech data directory
FINTECH_DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "fintech_data")

# This DataFrame will be used for datatype inference
target_df = None
target_file_path = os.path.join(FINTECH_DATA_DIR, TARGET_FINTECH_FILE)
if os.path.exists(target_file_path):
    try:
        target_df = pd.read_csv(target_file_path)
        print(f"Successfully loaded '{TARGET_FINTECH_FILE}' for datatype inference.")
    except Exception as e:
        print(f"Error loading '{TARGET_FINTECH_FILE}': {e}")
        target_df = None
else:
    print(f"Error: Target fintech file '{TARGET_FINTECH_FILE}' not found at '{target_file_path}'. Please ensure it exists.")


def get_embedding(text):
    """Generates an embedding for a given text using the local Ollama model."""
    try:
        response = ollama.embeddings(model=OLLAMA_MODEL_NAME, prompt=text)
        return response['embedding']
    except Exception as e:
        print(f"Error generating embedding with Ollama for text: '{text[:50]}...': {e}")
        return None

def get_column_datatype_pandas_dtype(df, column_name):
    """
    Infers the datatype of a specific column from a provided pandas DataFrame
    using pandas' df.dtypes attribute.
    """
    if df is None:
        return "dataframe_not_loaded"
    
    if column_name in df.columns:
        dtype_name = str(df[column_name].dtype)
        
        # Simplify common pandas dtypes for better readability/searchability
        if dtype_name == 'object':
            return 'string'
        elif dtype_name.startswith('int'):
            return 'integer'
        elif dtype_name.startswith('float'):
            return 'float'
        elif dtype_name == 'bool':
            return 'boolean'
        elif dtype_name.startswith('datetime'):
            return 'datetime'
        return dtype_name # Return exact pandas dtype name if not simplified
    else:
        return "column_not_found_in_df"

def upload_metadata_to_pinecone(csv_file_path):
    """
    Reads metadata from a CSV, filters for target file, generates embeddings, and uploads to Pinecone.
    """
    df_metadata = pd.read_csv(csv_file_path)

    # --- Filter metadata to be centric to application_train.csv ---
    df_filtered_metadata = df_metadata[df_metadata['Table'] == TARGET_FINTECH_FILE].copy()
    print(f"Processing metadata for '{TARGET_FINTECH_FILE}' only. Found {len(df_filtered_metadata)} relevant rows.")

    if target_df is None:
        print("Cannot proceed with datatype inference as the target DataFrame was not loaded. Exiting.")
        return # Exit if target DataFrame is not available

    pc = Pinecone(api_key=PINECONE_API_KEY)

    if INDEX_NAME not in pc.list_indexes().names():
        print(f"Creating Pinecone index: {INDEX_NAME}...")
        pc.create_index(
            name=INDEX_NAME,
            dimension=EMBEDDING_DIMENSION,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )
        print("Index created successfully.")
    else:
        print(f"Index '{INDEX_NAME}' already exists. Connecting to it.")

    index = pc.Index(INDEX_NAME)

    vectors_to_upsert = []
    BATCH_SIZE = 100

    print("Processing filtered metadata and inferring datatypes...")
    for i, row in df_filtered_metadata.iterrows(): # Iterate over filtered DataFrame
        table_name = row['Table']       # Will always be TARGET_FINTECH_FILE
        column_name = row['Row']

        # Get the datatype using the target_df
        datatype = get_column_datatype_pandas_dtype(target_df, column_name)

        # --- Determine 'isTarget' metadata ---
        is_target_column = (column_name.upper() == 'TARGET') # Case-insensitive check

        # Combine relevant columns for embedding
        text_to_embed = (
            f"Table: {table_name}. Column: {column_name}. "
            f"Description: {row['Description']}. "
            f"Special Notes: {row['Special'] if pd.notna(row['Special']) else ''}. "
            f"Datatype: {datatype}. "
            f"Is Target Column: {is_target_column}" # <--- Include isTarget in text for embedding
        )

        embedding = get_embedding(text_to_embed)

        if embedding:
            vector_id = f"metadata-{i}"
            metadata = {k: str(v) if pd.isna(v) else v for k, v in row.to_dict().items()}
            metadata['datatype'] = datatype         # <--- Add datatype to metadata dict
            metadata['isTarget'] = is_target_column # <--- Add isTarget to metadata dict

            vectors_to_upsert.append({
                "id": vector_id,
                "values": embedding,
                "metadata": metadata
            })

            if len(vectors_to_upsert) >= BATCH_SIZE:
                print(f"Upserting a batch of {len(vectors_to_upsert)} vectors...")
                try:
                    index.upsert(vectors=vectors_to_upsert)
                except Exception as upsert_e:
                    print(f"Error during batch upsert: {upsert_e}")
                finally:
                    vectors_to_upsert = []
        else:
            print(f"Skipping row {i} due to embedding error.")

    if vectors_to_upsert:
        print(f"Upserting final batch of {len(vectors_to_upsert)} vectors...")
        try:
            index.upsert(vectors=vectors_to_upsert)
            print("Final batch upserted successfully.")
        except Exception as final_upsert_e:
            print(f"Error during final batch upsert: {final_upsert_e}")
    else:
        print("No vectors to upsert.")

    print("Waiting a few seconds for indexing to complete before checking stats...")
    time.sleep(5)

    print("\nPinecone index stats:")
    print(index.describe_index_stats())

if __name__ == "__main__":
    csv_file = os.path.join(os.path.dirname(__file__), "..", "data", "metadata.csv")
    upload_metadata_to_pinecone(csv_file)