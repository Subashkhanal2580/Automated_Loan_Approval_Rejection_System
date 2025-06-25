import pandas as pd
from pinecone import Pinecone
import os
from dotenv import load_dotenv
import ollama
import re
from sklearn.feature_selection import f_classif
from scipy.stats import chi2_contingency

# Load environment variables
load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

i=0
y=0

if not PINECONE_API_KEY:
    print("Warning: PINECONE_API_KEY not found in .env file. Pinecone operations will fail.")

try:
    pc = Pinecone(api_key=PINECONE_API_KEY)
    index_name = "fintech-statisticalfinal-metadata-index"
    if index_name not in [idx.name for idx in pc.list_indexes().indexes]:
        print(f"Index '{index_name}' not found. Please create the index or check the name.")
        index = None
    else:
        index = pc.Index(index_name)
        print(f"Successfully connected to Pinecone index '{index_name}'")
except Exception as e:
    print(f"Error initializing Pinecone: {e}")
    index = None

def get_column_embedding(row_name):
    try:
        response = ollama.embeddings(model="nomic-embed-text", prompt=row_name)
        embedding = response["embedding"]
        if not embedding:
            print(f"Warning: No embedding returned for '{row_name}' from Ollama.")
            return None
        return embedding
    except Exception as e:
        print(f"Error generating embedding for '{row_name}' via Ollama: {e}")
        return None

def infer_default_typeofdata(column_name):
    if column_name.startswith("FLAG_"):
        return "boolean_like_categorical"
    elif re.match(r"AMT_|CNT_|DAYS_|HOUR_|RATE_|SCORE_|INDEX_", column_name):
        return "numerical"
    elif any(suffix in column_name for suffix in ["_TYPE", "_STATUS", "_MODE", "_CAT"]):
        return "categorical"
    else:
        return "unknown"

def find_exact_match(index, column_name, file_name):
    if index is None:
        return None
    vector_id = f"{file_name}-{column_name}-stats"
    try:
        response = index.fetch(ids=[vector_id])
        if vector_id in response.vectors:
            typeofdata = response.vectors[vector_id].metadata.get("typeofdata", "Not found")
            print(f"Found exact match for '{column_name}' with typeofdata: {typeofdata}")
            return typeofdata
        return None
    except Exception as e:
        print(f"Error fetching vector by ID '{vector_id}': {e}")
        return None

def find_similar_column(index, row_name, file_name, top_k=1):
    if index is None:
        print(f"Cannot perform similarity search for '{row_name}': Pinecone index is unavailable.")
        return None

    embedding = get_column_embedding(row_name)
    if embedding is None:
        return None

    try:
        query_response = index.query(
            vector=embedding,
            top_k=top_k,
            include_metadata=True,
            filter={"Table": file_name}
        )
        if query_response.matches:
            match = query_response.matches[0]
            typeofdata = match.metadata.get("typeofdata", "Not found")
            matched_row = match.metadata.get("Row", "unknown")
            score = match.score
            print(f"Found similar match for '{row_name}': Row='{matched_row}', typeofdata='{typeofdata}', score={score:.3f}")
            return typeofdata
        else:
            print(f"No matches found for '{row_name}' in similarity search.")
            return None
    except Exception as e:
        print(f"Error performing similarity search for '{row_name}': {e}")
        return None

def extract_typeofdata(file_path):
    try:
        if not os.path.exists(file_path):
            print(f"File not found: {file_path}")
            return {}
        df = pd.read_csv(file_path)
        file_name = os.path.basename(file_path).replace("balanced_", "")
        print(f"Loaded dataframe from '{file_path}' with {len(df.columns)} columns.")
    except Exception as e:
        print(f"Error loading dataframe: {e}")
        return {}

    column_types = {}
    for column in df.columns:
        if column == "TARGET":
            continue

        print(f"\nProcessing column: '{column}'...")
        typeofdata = find_exact_match(index, column, file_name)

        if typeofdata is None or typeofdata == "Not found":
            typeofdata = find_similar_column(index, column, file_name)

        if typeofdata is None or typeofdata == "Not found":
            typeofdata = infer_default_typeofdata(column)
            print(f"No match found for '{column}'; inferred typeofdata as: '{typeofdata}'")

        column_types[column] = typeofdata
        print(f"-> Assigned final typeofdata '{typeofdata}' to column '{column}'")

    return column_types

file_path = './balanced_data/application_train.csv'

try:
    column_types = extract_typeofdata(file_path)
    print("\n Final column types extracted:")
    print(column_types)
except Exception as e:
    print(f"\nAn error occurred during column type extraction: {e}")
    column_types = {}

if column_types:
    print("\n--- Starting Statistical Analysis ---")
    try:
        df = pd.read_csv(file_path)

        if 'TARGET' not in df.columns:
            print("Error: 'TARGET' column not found in dataframe. Cannot perform tests.")
        else:
            y = df['TARGET']
            anova_results = {}
            chi_results = {}
            useful_numerical_columns = set()
            useful_categorical_columns = set()

            for column, col_type in column_types.items():

                if col_type in ['identifier', 'numerical']:
                    print(f"\nCalculating ANOVA F-test for '{column}' (type: {col_type})...")
                    if column in df.columns:
                        feature_col = df[column].copy()
                        if feature_col.isnull().any():
                            mean_val = feature_col.mean()
                            feature_col.fillna(mean_val, inplace=True)
                            print(f"Imputed missing values with mean ({mean_val:.4f}).")

                        X_feature = feature_col.values.reshape(-1, 1)
                        try:
                            f_statistic, p_value = f_classif(X_feature, y)
                            p_val = p_value[0]
                            anova_results[column] = {'f_statistic': f_statistic[0], 'p_value': p_val}
                            print(f"-> ANOVA Result for '{column}': P-value = {p_val:.6f}")
                            if 0 < p_val < 0.5:
                                useful_numerical_columns.add(column)
                        except Exception as test_e:
                            print(f"Could not compute ANOVA test for '{column}'. Reason: {test_e}")

                elif col_type in ['categorical', 'boolean_like_categorical']:
                    print(f"\nCalculating Chi-Square test for '{column}' (type: {col_type})...")
                    if column in df.columns:
                        feature_col = df[column].copy()
                        if feature_col.isnull().any():
                            mode_val = feature_col.mode()[0]
                            feature_col.fillna(mode_val, inplace=True)
                            print(f"Imputed missing values with mode ('{mode_val}').")
                        try:
                            contingency_table = pd.crosstab(feature_col, y)
                            if contingency_table.shape[0] > 1 and contingency_table.shape[1] > 1:
                                chi2_stat, p_value, _, _ = chi2_contingency(contingency_table)
                                chi_results[column] = {'chi2_statistic': chi2_stat, 'p_value': p_value}
                                print(f"-> Chi-Square Result for '{column}': P-value = {p_value:.6f}")
                                if 0 < p_value < 0.5:
                                    useful_categorical_columns.add(column)
                            else:
                                print(f"Skipped Chi-Square test for '{column}' due to insufficient categories.")
                        except Exception as test_e:
                            print(f"Could not compute Chi-Square test for '{column}'. Reason: {test_e}")

            print("\n✅ Statistical analysis complete.")

            print("\nUseful numerical columns (0 < p < 0.5):")
            for col in useful_numerical_columns:
                print(f"  - {col}")
                i=i+1
            print(f"count of columns:{i}")

            print("\nUseful categorical columns (0 < p < 0.5):")
            for col in useful_categorical_columns:
                print(f"  - {col}")
                y=y+1
            print(f"count of columns:{i}")

    except FileNotFoundError:
        print(f"Error: The file was not found at {file_path}")
    except Exception as e:
        print(f"An unexpected error occurred during the statistical analysis: {e}")
else:
    print("\nSkipping statistical analysis because column types could not be extracted.")
