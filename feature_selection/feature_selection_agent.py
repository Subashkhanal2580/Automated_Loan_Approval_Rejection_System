import pandas as pd
from pinecone import Pinecone
import os
from dotenv import load_dotenv
import ollama
import re
from sklearn.feature_selection import f_classif
from scipy.stats import chi2_contingency
from langchain_core.tools import tool
from langchain.agents import AgentExecutor, create_react_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate

# --- Setup and Configuration ---

# Load environment variables from a .env file
# IMPORTANT: You now need both PINECONE_API_KEY and GOOGLE_API_KEY in your .env file
load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GOOGLE_API_KEY = os.getenv("GEMINI_API_KEY")

# Initialize Pinecone client
try:
    if not PINECONE_API_KEY:
        raise ValueError("PINECONE_API_KEY not found in .env file. Please set it.")
    pc = Pinecone(api_key=PINECONE_API_KEY)
    index_name = "fintech-statisticalfinal-metadata-index"
    if index_name not in [idx.name for idx in pc.list_indexes().indexes]:
        print(f"Warning: Pinecone index '{index_name}' not found. Tools requiring it will fail.")
        pinecone_index = None
    else:
        pinecone_index = pc.Index(index_name)
        print(f"Successfully connected to Pinecone index '{index_name}'")
except Exception as e:
    print(f"Error initializing Pinecone: {e}")
    pinecone_index = None

# --- Helper Functions (Internal Logic for Tools) ---
# Note: ollama is still used for generating embeddings, but not for the agent's reasoning.

def _get_column_embedding(column_name: str):
    """Generates an embedding for a given column name using Ollama."""
    try:
        response = ollama.embeddings(model="nomic-embed-text", prompt=column_name)
        return response.get("embedding")
    except Exception as e:
        print(f"Error generating embedding for '{column_name}': {e}")
        return None

def _find_exact_match_in_pinecone(column_name: str, file_name: str):
    """Fetches a column's metadata from Pinecone by a specific ID."""
    if pinecone_index is None: return None
    vector_id = f"{file_name}-{column_name}-stats"
    try:
        response = pinecone_index.fetch(ids=[vector_id])
        if response.vectors and vector_id in response.vectors:
            return response.vectors[vector_id].metadata.get("typeofdata", "Not found")
        return None
    except Exception as e:
        print(f"Error fetching vector by ID '{vector_id}': {e}")
        return None

def _find_similar_column_in_pinecone(column_name: str, file_name: str):
    """Finds a similar column in Pinecone using vector similarity search."""
    if pinecone_index is None: return None
    embedding = _get_column_embedding(column_name)
    if not embedding: return None
    try:
        query_response = pinecone_index.query(vector=embedding, top_k=1, include_metadata=True, filter={"Table": file_name})
        if query_response.matches:
            match = query_response.matches[0]
            print(f"Found similar match for '{column_name}': Row='{match.metadata.get('Row', 'N/A')}', Score={match.score:.3f}")
            return match.metadata.get("typeofdata")
        return None
    except Exception as e:
        print(f"Error in similarity search for '{column_name}': {e}")
        return None

def _infer_default_typeofdata(column_name: str):
    """Infers the data type of a column based on its name."""
    if column_name.startswith("FLAG_"): return "boolean_like_categorical"
    if re.match(r"AMT_|CNT_|DAYS_|HOUR_|RATE_|SCORE_|INDEX_", column_name): return "numerical"
    if any(suffix in column_name for suffix in ["_TYPE", "_STATUS", "_MODE", "_CAT"]): return "categorical"
    return "unknown"

# --- LangChain Tools ---
@tool
def get_column_types(file_path: str) -> dict:
    """
    Analyzes a CSV file to determine the data type of each column.
    It uses Pinecone for matching and falls back to name-based inference.
    The 'TARGET' column is automatically ignored.
    """
    print(f"\n🚀 Starting Tool: get_column_types for file: {file_path}")
    try:
        # --- FIX: Clean the file_path string to remove extraneous quotes/ticks ---
        cleaned_path = file_path.strip(" '\"`")
        df = pd.read_csv(cleaned_path)
        file_name = os.path.basename(cleaned_path).replace("balanced_", "")
    except Exception as e:
        return f"Error loading dataframe: {e}"

    column_types = {}
    for column in df.columns:
        if column == "TARGET": continue
        print(f"\nProcessing column: '{column}'...")
        typeofdata = _find_exact_match_in_pinecone(column, file_name)
        if not typeofdata or typeofdata == "Not found":
            typeofdata = _find_similar_column_in_pinecone(column, file_name)
        if not typeofdata or typeofdata == "Not found":
            typeofdata = _infer_default_typeofdata(column)
            print(f"No match in Pinecone for '{column}'; inferred type: '{typeofdata}'")
        column_types[column] = typeofdata
        print(f"-> Assigned final type '{typeofdata}' to column '{column}'")

    print("\n✅ Tool Finished: get_column_types. All columns processed.")
    return column_types

@tool
def perform_statistical_tests_and_select_features(file_path: str, column_types: dict) -> list:
    """
    Performs statistical tests (ANOVA for numerical, Chi-Square for categorical)
    to find features significantly correlated with the 'TARGET' column.
    Returns a list of useful feature names.
    NOTE: A p-value threshold of < 0.5 is used as per the original script's logic.
    """
    print(f"\n🚀 Starting Tool: perform_statistical_tests_and_select_features")
    try:
        # --- FIX: Clean the file_path string ---
        cleaned_path = file_path.strip(" '\"`")
        df = pd.read_csv(cleaned_path)
        if 'TARGET' not in df.columns:
            return "Error: 'TARGET' column not found in the dataframe."
        y = df['TARGET']
    except Exception as e:
        return f"Error loading dataframe: {e}"

    useful_columns = set()
    for column, col_type in column_types.items():
        if column not in df.columns: continue
        if col_type in ['identifier', 'numerical']:
            feature_col = df[column].copy()
            if feature_col.isnull().any(): feature_col.fillna(feature_col.mean(), inplace=True)
            try:
                _, p_value = f_classif(feature_col.values.reshape(-1, 1), y)
                if 0 < p_value[0] < 0.5: useful_columns.add(column)
            except Exception as e: print(f"Could not compute ANOVA for '{column}': {e}")
        elif col_type in ['categorical', 'boolean_like_categorical']:
            feature_col = df[column].copy()
            if feature_col.isnull().any(): feature_col.fillna(feature_col.mode()[0], inplace=True)
            try:
                contingency_table = pd.crosstab(feature_col, y)
                if contingency_table.shape[0] > 1 and contingency_table.shape[1] > 1:
                    _, p_value, _, _ = chi2_contingency(contingency_table)
                    if 0 < p_value < 0.5: useful_columns.add(column)
            except Exception as e: print(f"Could not compute Chi-Square for '{column}': {e}")

    print(f"\n✅ Tool Finished: perform_statistical_tests_and_select_features. Found {len(useful_columns)} useful features.")
    return sorted(list(useful_columns))

@tool
def save_selected_features_to_csv(original_file_path: str, selected_features: list, output_dir: str):
    """
    Saves a new CSV file containing only the selected feature columns and the 'TARGET' column.
    """
    print(f"\n🚀 Starting Tool: save_selected_features_to_csv")
    try:
        # --- FIX: Clean the file_path string ---
        cleaned_path = original_file_path.strip(" '\"`")
        os.makedirs(output_dir, exist_ok=True)
        df = pd.read_csv(cleaned_path)
        final_columns = selected_features[:]
        if 'TARGET' in df.columns and 'TARGET' not in final_columns:
            final_columns.append('TARGET')
        df_selected = df[final_columns]
        base_name = os.path.basename(cleaned_path)
        output_file_path = os.path.join(output_dir, f"selected_features_{base_name}")
        df_selected.to_csv(output_file_path, index=False)
        result_message = f"Successfully saved {len(df_selected.columns)} columns to '{output_file_path}'."
        print(f"✅ Tool Finished: {result_message}")
        return result_message
    except Exception as e:
        error_message = f"Error saving file: {e}"
        print(f"❌ {error_message}")
        return error_message

# --- Agent Definition and Execution ---

def main():
    """Main function to set up and run the LangChain agent."""
    if not GOOGLE_API_KEY:
        print("FATAL ERROR: GOOGLE_API_KEY not found in .env file.")
        print("Please add your Google API key to the .env file to use the Gemini model.")
        return

    input_file_path = './balanced_data/application_train.csv'
    output_directory = './selected_features_output'

    if not os.path.exists(input_file_path):
        print(f"FATAL ERROR: The input file was not found at '{input_file_path}'.")
        return

    tools = [
        get_column_types,
        perform_statistical_tests_and_select_features,
        save_selected_features_to_csv
    ]

    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0, google_api_key=GOOGLE_API_KEY)

    react_prompt_template = """
Answer the following questions as best you can. You have access to the following tools:
{tools}
Use the following format:
Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}
Thought:{agent_scratchpad}
"""
    prompt = PromptTemplate.from_template(react_prompt_template)
    agent = create_react_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(
        agent=agent, 
        tools=tools, 
        verbose=True, 
        handle_parsing_errors=True
    )

    print("\n\n--- Invoking LangChain Agent (with Gemini Pro) for Feature Selection ---")
    
    # Using a cleaner input string without markdown backticks for file paths
    agent_input = {
        "input": f"Start the feature selection workflow for the file at file_path '{input_file_path}'. "
                 f"First, get the column types. Second, run statistical tests to select features. "
                 f"Finally, save the resulting dataframe with only the selected columns "
                 f"to the output_dir '{output_directory}'. Confirm when done."
    }

    agent_executor.invoke(agent_input)

if __name__ == "__main__":
    main()
