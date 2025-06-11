import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain_google_genai import GoogleGenerativeAI
from langchain_community.llms import Ollama

# Setting up Gemini API key
os.environ["GOOGLE_API_KEY"] = "AIzaSyDm11bERs5j8jOsV6zC1IMv6HFQihm2Zhk"

# Loading the dataset
file_path = "../dataset/application_test.csv"
df = pd.read_csv(file_path)

# Model configurations with Gemini as primary and Ollama models as fallbacks
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

# Function to create LLM based on configuration
def create_llm(config):
    if config["type"] == "google":
        return GoogleGenerativeAI(model=config["model"], temperature=config["temperature"])
    elif config["type"] == "ollama":
        return Ollama(model=config["model"], base_url=config["base_url"], temperature=config["temperature"])
    else:
        raise ValueError(f"Unknown model type: {config['type']}")

# Create list of LLMs and corresponding agents
llms = [create_llm(config) for config in model_configs]
agents = [create_pandas_dataframe_agent(llm, df, verbose=True, allow_dangerous_code=True) for llm in llms]

# Function to run query with fallback mechanism
def run_query_with_fallback(agents, query):
    for agent in agents:
        try:
            result = agent.run(query)
            return result
        except Exception as e:
            print(f"Error with agent: {e}")
            continue
    raise Exception("All agents failed to run the query")

# List of EDA queries
eda_queries = [
    "What is the shape of the dataset?",
    "How many missing values are there in each column?",
    "Provide summary statistics for the numerical columns.",
    "List the unique values in the categorical columns.",
    "Provide the data types of each column using the command `df.dtypes`.",
    "Create a correlation heatmap for the numerical columns.",
    "Show the value counts for the categorical columns.",
    "Plot bar charts for the categorical columns."
]

# Execute the queries with fallback
for query in eda_queries:
    print(f"Query: {query}")
    try:
        result = run_query_with_fallback(agents, query)
        print(result)
    except Exception as e:
        print(f"Failed to run query: {e}")