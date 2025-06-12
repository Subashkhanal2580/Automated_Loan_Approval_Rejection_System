import os
import pandas as pd
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.chat_models import ChatOllama
from langchain.agents.agent_types import AgentType
import re
import string

# Step 2: Set up Google API key
os.environ["GOOGLE_API_KEY"] = "AIzaSyDm11bERs5j8jOsV6zC1IMv6HFQihm2Zhk"

file_name= "application_train.csv"  

# Step 3: Load the CSV data
df = pd.read_csv('./dataset/application_train.csv')  

# Step 4: Define model configurations
model_configs = [
    {
        "type": "google",
        "model": "gemini-2.0-flash",
        "temperature": 0
    },
    {
        "type": "ollama",
        "model": "gemma3:1b",
        "base_url": "http://localhost:11434",
        "temperature": 0
    },
    {
        "type": "ollama",
        "model": "qwen3:0.6b",
        "base_url": "http://localhost:11434",
        "temperature": 0
    }
]

# Step 5: Define functions for agent creation and fallback
def create_agent(df, model_config):
    try:
        if model_config["type"] == "google":
            llm = ChatGoogleGenerativeAI(
                model=model_config["model"],
                temperature=model_config["temperature"],
                google_api_key=os.environ["GOOGLE_API_KEY"]
            )
        elif model_config["type"] == "ollama":
            llm = ChatOllama(
                model=model_config["model"],
                base_url=model_config["base_url"],
                temperature=model_config["temperature"]
            )
        else:
            raise ValueError(f"Unsupported model type: {model_config['type']}")
        
        return create_pandas_dataframe_agent(
            llm,
            df,
            verbose=True,
            agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            allow_dangerous_code=True,
            handle_parsing_errors=True

        )
    except Exception as e:
        print(f"Failed to initialize agent with {model_config['model']}: {e}")
        return None

def run_query_with_fallback(df, query, model_configs):
    for config in model_configs:
        print(f"Trying model: {config['model']}")
        agent = create_agent(df, config)
        if agent is None:
            continue
        try:
            result = agent.run(query)
            print(f"Success with {config['model']}: {result}")
            return True
        except Exception as e:
            print(f"Error with {config['model']}: {e}")
    print(f"Failed to execute query '{query}' with all models")
    return False

# Step 6: Execute cleaning queries
queries = [
    "Detect duplicate rows and list them",
    "Remove duplicate rows from the dataset",
    "Identify columns with missing values and their counts",
    "Detect and handle outliers in numerical columns",
    "Standardize text columns (e.g., convert to lowercase, remove punctuation)",
]

for query in queries:
    success = run_query_with_fallback(df, query, model_configs)
    if not success:
        print(f"Could not process query: {query}")

# Step 7: Save the cleaned DataFrame
df.to_csv(f'./dataset/cleaned_{file_name}.csv', index=False)