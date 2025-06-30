import pandas as pd
import ollama
from pinecone import Pinecone, ServerlessSpec
import os
from dotenv import load_dotenv
import time

# Load environment variables
load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
OLLAMA_MODEL_NAME = os.getenv("OLLAMA_MODEL_NAME")

# Pinecone Index details - using a general name for multiple files
INDEX_NAME = "fintech-app-traintestfinal-metadata-index" 
EMBEDDING_DIMENSION = 768

# Define the path to your fintech data directory
FINTECH_DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "fintech_data")

# Global dictionary to store loaded dataframes for efficiency
# This avoids reloading the same large CSV multiple times for different columns
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
            loaded_fintech_dfs[table_name] = None # Mark as not found to avoid repeated attempts
            return None
        try:
            df = pd.read_csv(file_path)
            loaded_fintech_dfs[table_name] = df
            print(f"Loaded '{table_name}' for datatype inference.")
            return df
        except Exception as e:
            print(f"Error loading '{table_name}' for datatype inference: {e}")
            loaded_fintech_dfs[table_name] = None # Mark as error
            return None
    return loaded_fintech_dfs[table_name] # Return from cache

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
        # print(f"Warning: Column '{column_name}' not found in the DataFrame for '{df.name}'.") # df.name might not be set
        return "column_not_found_in_df"

def upload_metadata_to_pinecone(csv_file_path):
    """
    Reads metadata from a CSV, processes all relevant files, generates embeddings,
    and uploads to a single Pinecone index.
    """
    df_metadata = pd.read_csv(csv_file_path)

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

    print("Processing metadata for all listed fintech CSVs...")
    for i, row in df_metadata.iterrows(): # Iterate over ALL rows in metadata DataFrame
        table_name = row['Table']
        column_name = row['Row']

        # Load the relevant fintech DataFrame (or get from cache)
        current_fintech_df = load_fintech_df(table_name)
        
        if current_fintech_df is None:
            print(f"Skipping row {i} for '{table_name}.{column_name}' due to DataFrame loading error.")
            continue # Skip to the next row if the file couldn't be loaded

        # Get the datatype using the current_fintech_df
        datatype = get_column_datatype_pandas_dtype(current_fintech_df, column_name)

        # --- Determine 'isTarget' metadata ---
        is_target_column = (column_name.upper() == 'TARGET') # Case-insensitive check

        # Combine relevant columns for embedding
        text_to_embed = (
            f"Table: {table_name}. Column: {column_name}. "
            f"Description: {row['Description']}. "
            f"Special Notes: {row['Special'] if pd.notna(row['Special']) else ''}. "
            f"Datatype: {datatype}. "
            f"Is Target Column: {is_target_column}"
        )

        embedding = get_embedding(text_to_embed)

        if embedding:
            vector_id = f"{table_name}-{column_name}" # Make ID unique across files
            # Ensure vector_id is valid for Pinecone (no slashes, etc.)
            vector_id = vector_id.replace('.csv', '').replace('.', '_').replace('/', '_')

            metadata = {k: str(v) if pd.isna(v) else v for k, v in row.to_dict().items()}
            metadata['datatype'] = datatype
            metadata['isTarget'] = is_target_column

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
            print(f"Skipping row {i} for '{table_name}.{column_name}' due to embedding error.")

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