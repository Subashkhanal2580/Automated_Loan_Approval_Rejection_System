# --- src/agents.py ---
import autogen
import pandas as pd
import os
from pathlib import Path

# Agent 0: Data Profiler
class DataProfilerAgent(autogen.UserProxyAgent):
    """
    An agent that loads a CSV file, performs initial data profiling,
    and summarizes key data characteristics for further analysis.
    """
    def __init__(self, name="Data_Profiler_Agent", llm_config=None, code_execution_config=None, **kwargs):
        super().__init__(
            name=name,
            human_input_mode="NEVER",
            max_consecutive_auto_reply=10,
            llm_config=llm_config,
            code_execution_config=code_execution_config,
            **kwargs
        )
        self.update_system_message(
            "You are a skilled Data Profiler. Your **IMMEDIATE and PRIMARY task** is to generate "
            "and execute Python code to load the given CSV file at the specified absolute path "
            "and perform an initial exploratory data analysis. "
            "**You MUST output the Python code inside a code block.** "
            "The code should use pandas to:\n"
            "1.  Load the CSV file into a DataFrame named `df`.\n"
            "2.  Print `df.info()`.\n"
            "3.  Calculate and print the count and percentage of missing values for all columns, sorted descending by percentage.\n"
            "4.  For numerical columns, print `df.describe()`.\n"
            "5.  For categorical/object columns, identify and print up to 10 unique values and their counts, or indicate if a column has high cardinality (more than 50 unique values).\n"
            "6.  Identify and print any obvious outliers or anomalies (e.g., very large numbers like 365243 for 'DAYS_EMPLOYED' meaning unemployment, negative ages for 'DAYS_BIRTH').\n"
            "\nAfter the code execution, provide a concise, structured text report summarizing these findings based on the *output of the executed code*. "
            "Do NOT generate a cleaning plan or code for cleaning, only the data profile summary. "
            "Once the report is generated, clearly state 'DATA_PROFILING_COMPLETE' to signal the next step to the chat manager."
            "Ensure your code outputs are well-formatted for readability."
        )

# Agent 1: Metadata Analyst
class MetadataAnalystAgent(autogen.AssistantAgent):
    """An agent that analyzes metadata AND a data profile to create a data cleaning plan."""
    def __init__(self, name="Metadata_Analyst_Agent", llm_config=None, **kwargs):
        super().__init__(
            name=name,
            system_message=(
                "You are an expert Metadata Analyst for fintech data. "
                "Your task is to analyze BOTH the provided metadata description AND the **textual data profile summary** "
                "from a Data Profiler Agent. You **WILL NOT execute any code** and will not ask for code execution. "
                "Based on this combined information, generate a precise, step-by-step data cleaning plan in natural language. "
                "Crucially, explain your reasoning and rationale for each proposed cleaning step, explicitly referencing findings from the data profile (e.g., 'Due to 90% missing values in X column...', 'Given 'Y' is an object type with 5000 unique values...'). "
                "The plan must cover dropping irrelevant columns, handling missing values (imputation with mean, median, or mode, or dropping), "
                "data type conversions, normalization, encoding, and special considerations for PII and currency data. "
                "For PII, suggest redaction or anonymization. For currency, suggest standardization to a common format (e.g., USD) and format uniformity (e.g., removing symbols). "
                "Ensure your plan is clear, unambiguous, and ready for a Python developer to implement using Pandas. "
                "DO NOT generate code, only the natural language plan. "
                "You will receive the metadata as a Pandas DataFrame string representation AND a text-based data profile. "
                "Identify the target CSV file from the 'Table' column in the metadata (which may be different from the actual file name provided in the prompt for profiling, but the columns are consistent) and explicitly state its full path in your plan for the Data Cleaner Executor Agent to use."
                "Present the plan with clear headings for each section (e.g., '1. Column Dropping:', '2. Missing Value Handling:', '3. Data Type Conversion & Formatting:', '4. Encoding:', '5. Special Handling (PII & Currency):')."
                "\n\n**Special Instructions for Optimal Cleaning Specific to Fintech Data:**"
                "   - **Time-based Features (e.g., 'DAYS_BIRTH', 'DAYS_EMPLOYED'):** For columns marked 'time only relative to the application' (or similar descriptions implying days from application), convert them from negative days to positive years (e.g., divide by -365.25). Pay very special attention to 'DAYS_EMPLOYED' where an extremely large positive value (e.g., 365243) is an outlier often indicating unemployment. Propose a robust handling strategy for this specific outlier (e.g., replacing with NaN for imputation, or creating a new binary 'is_unemployed' feature and then imputing the original column)."
                "   - **Normalized Features (e.g., 'REGION_POPULATION_RELATIVE', 'EXT_SOURCE_X', building-related 'APARTMENTS_AVG/MODE/MEDI'):** Columns explicitly marked 'normalized' generally do not require further scaling, but their missing values still need appropriate imputation (e.g., median for skewed distributions, mean for symmetric, or specific domain knowledge based imputation)."
                "   - **Currency/Amount Features (e.g., 'AMT_INCOME_TOTAL', 'AMT_CREDIT'):** Ensure these columns are treated as numeric. Check for and propose strategies to handle potential outliers (e.g., winsorization, capping, or log transformation if highly skewed) to improve model performance. Ensure no non-numeric characters remain."
                "   - **Building Information Features (e.g., columns with '_AVG', '_MODE', '_MEDI' suffixes):** These describe building characteristics and are notorious for having a high percentage of missing values. Suggest strategies like imputing with median/mode, or if a large portion is missing, consider creating a 'is_missing' binary flag for these columns and then imputing, or even dropping sets of these highly incomplete features if they don't add significant value. Prioritize 'MODE' for imputation if available, as it represents the most frequent value."
                "   - **Flag/Binary Features (e.g., 'FLAG_OWN_CAR', 'FLAG_DOCUMENT_X'):** These are typically binary (0/1). Ensure their data type is set to integer or boolean and handle any unexpected values."
                "   - **Categorical Data (e.g., 'NAME_CONTRACT_TYPE', 'CODE_GENDER', 'OCCUPATION_TYPE'):** Clearly identify nominal (no inherent order) and ordinal (with inherent order) categorical columns. Suggest appropriate encoding strategies (e.g., one-hot encoding for nominal categories with a reasonable number of unique values, label encoding for ordinal, or even target encoding for high cardinality nominal features, if applicable)."
                "   - **ID Column ('SK_ID_CURR'):** This is a unique identifier. Ensure it is preserved (e.g., as the DataFrame index) but explicitly state it should *not* be used as a predictive feature directly as it carries no predictive power on its own."
                "   - **Dropping Redundant/Irrelevant Columns:** Based on descriptions AND actual data sparsity/variance from the data profile, identify and propose dropping columns that are clearly irrelevant or overwhelmingly sparse for predictive modeling."
                "\n\nAfter presenting the full data cleaning plan, clearly state 'DATA_CLEANING_PLAN_COMPLETE' to signal the next step to the chat manager."
            ),
            llm_config=llm_config,
            **kwargs
        )

# Agent 2: Data Cleaner Executor
class DataCleanerExecutorAgent(autogen.UserProxyAgent):
    """An agent that executes a data cleaning plan by writing and running Python code."""
    def __init__(self, name="Data_Cleaner_Executor_Agent", llm_config=None, **kwargs):
        super().__init__(
            name=name,
            human_input_mode="NEVER",
            max_consecutive_auto_reply=10,
            is_termination_msg=lambda x: x.get("content", "").rstrip().endswith("TERMINATE"),
            llm_config=llm_config,
            **kwargs
        )
        self.update_system_message(
            "You are a master Python Data Engineer. You will receive a data cleaning plan with ABSOLUTE file paths. "
            "Your task is to write **and execute** Python code to perform the cleaning. "
            "**You MUST output the Python code inside a code block.** "
            "Before writing any code, outline your thought process and step-by-step approach to implementing the cleaning plan, explaining your logic for each coding decision. "
            "1. **Load the data** using the provided absolute path for the input file.\n"
            "2. **Perform the cleaning steps** as described in the plan.\n"
            "3. **Save the cleaned file**: "
            "   Identify the absolute path for the *output directory* provided in the prompt (e.g., `/your/project/cleaned_data`). "
            "   From the *input file path*, extract the original filename (e.g., `application_test.csv`). "
            "   Construct the full absolute output file path by joining the output directory path with the original filename, adding a `_cleaned` suffix before the `.csv` extension. "
            "   For example, if the original input file is `/path/to/data/application_test.csv` and the output directory is `/path/to/cleaned_data`, the cleaned file should be saved as `/path/to/cleaned_data/application_test_cleaned.csv`. "
            "   Use `os.path.join` or `pathlib.Path` for robust path construction in your code. "
            "Ensure your code is a single, complete, executable block. Do not use relative paths for file operations. "
            "After successful execution, confirm the absolute path where the file was saved and then clearly state 'DATA_CLEANING_EXECUTION_COMPLETE'."
        )

# Agent 3: Code Reviewer
class CodeReviewerAgent(autogen.AssistantAgent):
    """
    An agent that reviews the code execution output from Data_Cleaner_Executor_Agent.
    It identifies errors, provides debugging information, and suggests fixes.
    """
    def __init__(self, name="Code_Reviewer_Agent", llm_config=None, **kwargs):
        super().__init__(
            name=name,
            system_message=(
                "You are an expert Code Reviewer and Debugger. Your role is to analyze the **code execution output received in the previous turn** "
                "from the Data_Cleaner_Executor_Agent. "
                "If the execution results in an error (e.g., a traceback), you **MUST** identify the root cause "
                "of the error directly from the provided output. Explain it clearly and suggest specific Python code modifications "
                "to fix it. You **DO NOT** need to ask for `config.yaml`, operating system details, Python version, or `pip list` output; "
                "assume the environment is correctly set up for running Python code. "
                "**If you need to inspect the data for debugging or verify paths, you have code execution capabilities to load the data yourself and run small diagnostic snippets within a code block.** "
                "Your primary focus is on the Python code logic. "
                "If the execution is successful, simply confirm the success. "
                "Your feedback should be actionable and aimed at helping the Data_Cleaner_Executor_Agent "
                "produce correct and working code for the next attempt. Once you have provided your review or confirmed success, clearly state 'CODE_REVIEW_COMPLETE'."
            ),
            llm_config=llm_config,
            **kwargs
        )