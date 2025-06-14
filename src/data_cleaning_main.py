# --- src/main.py ---
import os
import autogen
import pandas as pd
from pathlib import Path
import io
from dotenv import load_dotenv
from agents import DataProfilerAgent, DataCleanerExecutorAgent, MetadataAnalystAgent, CodeReviewerAgent

# Ensure the .env file is loaded
load_dotenv()

# Get the directory where this script (main.py) is located.
SRC_DIR = Path(__file__).parent.resolve()
# Get the root directory of the project (one level up from 'src').
PROJECT_ROOT = SRC_DIR.parent.resolve()
# Define absolute paths for data and output directories.
DATASET_DIR = PROJECT_ROOT / "dataset"
CLEANED_DIR = PROJECT_ROOT / "cleaned_data"

# Ensure the output directory exists.
CLEANED_DIR.mkdir(exist_ok=True)

print(f"--- Path Information ---")
print(f"Project Root Directory: {PROJECT_ROOT}")
print(f"Dataset Directory: {DATASET_DIR}")
print(f"Cleaned Data Directory: {CLEANED_DIR}")
print(f"------------------------\n")

# --- Autogen Configuration ---
config_list = [
    {
        "model": os.environ.get("OPENAI_MODEL_NAME", "gpt-4o"), # Default to gpt-4o if not set
        "api_key": os.environ.get("OPENAI_API_KEY"),
        "api_type": "openai"
    },
    {
        "model": "gemini-1.5-flash", # Ensure this model is available and configured
        "api_key": os.environ.get("GOOGLE_API_KEY"),
        "api_type": "google"
    },
]

llm_config = {
    "config_list": config_list,
    "temperature": 0.5,
    "timeout": 300, # Increased timeout for potentially long operations
}

# --- Target Data File ---
TARGET_DATA_FILE = 'application_test.csv'
TARGET_DATA_PATH = DATASET_DIR / TARGET_DATA_FILE

# --- Metadata Loading ---
METADATA_FILE_CONTENT = """Table,Row,Description,Special
1,application_train.csv,SK_ID_CURR,ID of loan in our sample,
2,application_train.csv,TARGET,"Target variable (1 - client with payment difficulties: he/she had late payment more than X days on at least one of the first Y installments of the loan in our sample, 0 - all other cases)",
5,application_train.csv,NAME_CONTRACT_TYPE,Identification if loan is cash or revolving,
6,application_train.csv,CODE_GENDER,Gender of the client,
7,application_train.csv,FLAG_OWN_CAR,Flag if the client owns a car,
8,application_train.csv,FLAG_OWN_REALTY,Flag if client owns a house or flat,
9,application_train.csv,CNT_CHILDREN,Number of children the client has,
10,application_train.csv,AMT_INCOME_TOTAL,Income of the client,
11,application_train.csv,AMT_CREDIT,Credit amount of the loan,
12,application_train.csv,AMT_ANNUITY,Loan annuity,
13,application_train.csv,AMT_GOODS_PRICE,For consumer loans it is the price of the goods for which the loan is given,
14,application_train.csv,NAME_TYPE_SUITE,Who was accompanying client when he was applying for the loan,
15,application_train.csv,NAME_INCOME_TYPE,"Clients income type (businessman, working, maternity leave, )",
16,application_train.csv,NAME_EDUCATION_TYPE,Level of highest education the client achieved,
17,application_train.csv,NAME_FAMILY_STATUS,Family status of the client,
18,application_train.csv,NAME_HOUSING_TYPE,"What is the housing situation of the client (renting, living with parents, ...)",
19,application_train.csv,REGION_POPULATION_RELATIVE,Normalized population of region where client lives (higher number means the client lives in more populated region),normalized
20,application_train.csv,DAYS_BIRTH,Client's age in days at the time of application,time only relative to the application
21,application_train.csv,DAYS_EMPLOYED,How many days before the application the person started current employment,time only relative to the application
22,application_train.csv,DAYS_REGISTRATION,How many days before the application did client change his registration,time only relative to the application
23,application_train.csv,DAYS_ID_PUBLISH,How many days before the application did client change the identity document with which he applied for the loan,time only relative to the application
24,application_train.csv,OWN_CAR_AGE,Age of client's car,
25,application_train.csv,FLAG_MOBIL,"Did client provide mobile phone (1=YES, 0=NO)",
26,application_train.csv,FLAG_EMP_PHONE,"Did client provide work phone (1=YES, 0=NO)",
27,application_train.csv,FLAG_WORK_PHONE,"Did client provide home phone (1=YES, 0=NO)",
28,application_train.csv,FLAG_CONT_MOBILE,"Was mobile phone reachable (1=YES, 0=NO)",
29,application_train.csv,FLAG_PHONE,"Did client provide home phone (1=YES, 0=NO)",
30,application_train.csv,FLAG_EMAIL,"Did client provide email (1=YES, 0=NO)",
31,application_train.csv,OCCUPATION_TYPE,What kind of occupation does the client have,
32,application_train.csv,CNT_FAM_MEMBERS,How many family members does client have,
33,application_train.csv,REGION_RATING_CLIENT,"Our rating of the region where client lives (1,2,3)",
34,application_train.csv,REGION_RATING_CLIENT_W_CITY,"Our rating of the region where client lives with taking city into account (1,2,3)",
35,application_train.csv,WEEKDAY_APPR_PROCESS_START,On which day of the week did the client apply for the loan,
36,application_train.csv,HOUR_APPR_PROCESS_START,Approximately at what hour did the client apply for the loan,rounded
37,application_train.csv,REG_REGION_NOT_LIVE_REGION,"Flag if client's permanent address does not match contact address (1=different, 0=same, at region level)",
38,application_train.csv,REG_REGION_NOT_WORK_REGION,"Flag if client's permanent address does not match work address (1=different, 0=same, at region level)",
39,application_train.csv,LIVE_REGION_NOT_WORK_REGION,"Flag if client's contact address does not match work address (1=different, 0=same, at region level)",
40,application_train.csv,REG_CITY_NOT_LIVE_CITY,"Flag if client's permanent address does not match contact address (1=different, 0=same, at city level)",
41,application_train.csv,REG_CITY_NOT_WORK_CITY,"Flag if client's contact address does not match work address (1=different, 0=same, at city level)",
42,application_train.csv,LIVE_CITY_NOT_WORK_CITY,"Flag if client's contact address does not match work address (1=different, 0=same, at city level)",
43,application_train.csv,ORGANIZATION_TYPE,Type of organization where client works,
44,application_train.csv,EXT_SOURCE_1,Normalized score from external data source,normalized
45,application_train.csv,EXT_SOURCE_2,Normalized score from external data source,normalized
46,application_train.csv,EXT_SOURCE_3,Normalized score from external data source,normalized
47,application_train.csv,APARTMENTS_AVG,"Normalized information about building where the client lives, What is average (_AVG suffix), modus (_MODE suffix), median (_MEDI suffix) apartment size, common area, living area, age of building, number of elevators, number of entrances, state of the building, number of floor",normalized
48,application_train.csv,BASEMENTAREA_AVG,"Normalized information about building where the client lives, What is average (_AVG suffix), modus (_MODE suffix), median (_MEDI suffix) apartment size, common area, living area, age of building, number of elevators, number of entrances, state of the building, number of floor",normalized
49,application_train.csv,YEARS_BEGINEXPLUATATION_AVG,"Normalized information about building where the client lives, What is average (_AVG suffix), modus (_MODE suffix), median (_MEDI suffix) apartment size, common area, living area, age of building, number of elevators, number of entrances, state of the building, number of floor",normalized
50,application_train.csv,YEARS_BUILD_AVG,"Normalized information about building where the client lives, What is average (_AVG suffix), modus (_MODE suffix), median (_MEDI suffix) apartment size, common area, living area, age of building, number of elevators, number of entrances, state of the building, number of floor",normalized
51,application_train.csv,COMMONAREA_AVG,"Normalized information about building where the client lives, What is average (_AVG suffix), modus (_MODE suffix), median (_MEDI suffix) apartment size, common area, living area, age of building, number of elevators, number of entrances, state of the building, number of floor",normalized
52,application_train.csv,ELEVATORS_AVG,"Normalized information about building where the client lives, What is average (_AVG suffix), modus (_MODE suffix), median (_MEDI suffix) apartment size, common area, living area, age of building, number of elevators, number of entrances, state of the building, number of floor",normalized
53,application_train.csv,ENTRANCES_AVG,"Normalized information about building where the client lives, What is average (_AVG suffix), modus (_MODE suffix), median (_MEDI suffix) apartment size, common area, living area, age of building, number of elevators, number of entrances, state of the building, number of floor",normalized
54,application_train.csv,FLOORSMAX_AVG,"Normalized information about building where the client lives, What is average (_AVG suffix), modus (_MODE suffix), median (_MEDI suffix) apartment size, common area, living area, age of building, number of elevators, number of entrances, state of the building, number of floor",normalized
55,application_train.csv,FLOORSMIN_AVG,"Normalized information about building where the client lives, What is average (_AVG suffix), modus (_MODE suffix), median (_MEDI suffix) apartment size, common area, living area, age of building, number of elevators, number of entrances, state of the building, number of floor",normalized
56,application_train.csv,LANDAREA_AVG,"Normalized information about building where the client lives, What is average (_AVG suffix), modus (_MODE suffix), median (_MEDI suffix) apartment size, common area, living area, age of building, number of elevators, number of entrances, state of the building, number of floor",normalized
57,application_train.csv,LIVINGAPARTMENTS_AVG,"Normalized information about building where the client lives, What is average (_AVG suffix), modus (_MODE suffix), median (_MEDI suffix) apartment size, common area, living area, age of building, number of elevators, number of entrances, state of the building, number of floor",normalized
58,application_train.csv,LIVINGAREA_AVG,"Normalized information about building where the client lives, What is average (_AVG suffix), modus (_MODE suffix), median (_MEDI suffix) apartment size, common area, living area, age of building, number of elevators, number of entrances, state of the building, number of floor",normalized
59,application_train.csv,NONLIVINGAPARTMENTS_AVG,"Normalized information about building where the client lives, What is average (_AVG suffix), modus (_MODE suffix), median (_MEDI suffix) apartment size, common area, living area, age of building, number of elevators, number of entrances, state of the building, number of floor",normalized
60,application_train.csv,NONLIVINGAREA_AVG,"Normalized information about building where the client lives, What is average (_AVG suffix), modus (_MODE suffix), median (_MEDI suffix) apartment size, common area, living area, age of building, number of elevators, number of entrances, state of the building, number of floor",normalized
61,application_train.csv,APARTMENTS_MODE,"Normalized information about building where the client lives, What is average (_AVG suffix), modus (_MODE suffix), median (_MEDI suffix) apartment size, common area, living area, age of building, number of elevators, number of entrances, state of the building, number of floor",normalized
62,application_train.csv,BASEMENTAREA_MODE,"Normalized information about building where the client lives, What is average (_AVG suffix), modus (_MODE suffix), median (_MEDI suffix) apartment size, common area, living area, age of building, number of elevators, number of entrances, state of the building, number of floor",normalized
63,application_train.csv,YEARS_BEGINEXPLUATATION_MODE,"Normalized information about building where the client lives, What is average (_AVG suffix), modus (_MODE suffix), median (_MEDI suffix) apartment size, common area, living area, age of building, number of elevators, number of entrances, state of the building, number of floor",normalized
64,application_train.csv,YEARS_BUILD_MODE,"Normalized information about building where the client lives, What is average (_AVG suffix), modus (_MODE suffix), median (_MEDI suffix) apartment size, common area, living area, age of building, number of elevators, number of entrances, state of the building, number of floor",normalized
65,application_train.csv,COMMONAREA_MODE,"Normalized information about building where the client lives, What is average (_AVG suffix), modus (_MODE suffix), median (_MEDI suffix) apartment size, common area, living area, age of building, number of elevators, number of entrances, state of the building, number of floor",normalized
66,application_train.csv,ELEVATORS_MODE,"Normalized information about building where the client lives, What is average (_AVG suffix), modus (_MODE suffix), median (_MEDI suffix) apartment size, common area, living area, age of building, number of elevators, number of entrances, state of the building, number of floor",normalized
67,application_train.csv,ENTRANCES_MODE,"Normalized information about building where the client lives, What is average (_AVG suffix), modus (_MODE suffix), median (_MEDI suffix) apartment size, common area, living area, age of building, number of elevators, number of entrances, state of the building, number of floor",normalized
68,application_train.csv,FLOORSMAX_MODE,"Normalized information about building where the client lives, What is average (_AVG suffix), modus (_MODE suffix), median (_MEDI suffix) apartment size, common area, living area, age of building, number of elevators, number of entrances, state of the building, number of floor",normalized
69,application_train.csv,FLOORSMIN_MODE,"Normalized information about building where the client lives, What is average (_AVG suffix), modus (_MODE suffix), median (_MEDI suffix) apartment size, common area, living area, age of building, number of elevators, number of entrances, state of the building, number of floor",normalized
70,application_train.csv,LANDAREA_MODE,"Normalized information about building where the client lives, What is average (_AVG suffix), modus (_MODE suffix), median (_MEDI suffix) apartment size, common area, living area, age of building, number of elevators, number of entrances, state of the building, number of floor",normalized
71,application_train.csv,LIVINGAPARTMENTS_MODE,"Normalized information about building where the client lives, What is average (_AVG suffix), modus (_MODE suffix), median (_MEDI suffix) apartment size, common area, living area, age of building, number of elevators, number of entrances, state of the building, number of floor",normalized
72,application_train.csv,LIVINGAREA_MODE,"Normalized information about building where the client lives, What is average (_AVG suffix), modus (_MODE suffix), median (_MEDI suffix) apartment size, common area, living area, age of building, number of elevators, number of entrances, state of the building, number of floor",normalized
73,application_train.csv,NONLIVINGAPARTMENTS_MODE,"Normalized information about building where the client lives, What is average (_AVG suffix), modus (_MODE suffix), median (_MEDI suffix) apartment size, common area, living area, age of building, number of elevators, number of entrances, state of the building, number of floor",normalized
74,application_train.csv,NONLIVINGAREA_MODE,"Normalized information about building where the client lives, What is average (_AVG suffix), modus (_MODE suffix), median (_MEDI suffix) apartment size, common area, living area, age of building, number of elevators, number of entrances, state of the building, number of floor",normalized
75,application_train.csv,APARTMENTS_MEDI,"Normalized information about building where the client lives, What is average (_AVG suffix), modus (_MODE suffix), median (_MEDI suffix) apartment size, common area, living area, age of building, number of elevators, number of entrances, state of the building, number of floor",normalized
76,application_train.csv,BASEMENTAREA_MEDI,"Normalized information about building where the client lives, What is average (_AVG suffix), modus (_MODE suffix), median (_MEDI suffix) apartment size, common area, living area, age of building, number of elevators, number of entrances, state of the building, number of floor",normalized
77,application_train.csv,YEARS_BEGINEXPLUATATION_MEDI,"Normalized information about building where the client lives, What is average (_AVG suffix), modus (_MODE suffix), median (_MEDI suffix) apartment size, common area, living area, age of building, number of elevators, number of entrances, state of the building, number of floor",normalized
78,application_train.csv,YEARS_BUILD_MEDI,"Normalized information about building where the client lives, What is average (_AVG suffix), modus (_MODE suffix), median (_MEDI suffix) apartment size, common area, living area, age of building, number of elevators, number of entrances, state of the building, number of floor",normalized
79,application_train.csv,COMMONAREA_MEDI,"Normalized information about building where the client lives, What is average (_AVG suffix), modus (_MODE suffix), median (_MEDI suffix) apartment size, common area, living area, age of building, number of elevators, number of entrances, state of the building, number of floor",normalized
80,application_train.csv,ELEVATORS_MEDI,"Normalized information about building where the client lives, What is average (_AVG suffix), modus (_MODE suffix), median (_MEDI suffix) apartment size, common area, living area, age of building, number of elevators, number of entrances, state of the building, number of floor",normalized
81,application_train.csv,ENTRANCES_MEDI,"Normalized information about building where the client lives, What is average (_AVG suffix), modus (_MODE suffix), median (_MEDI suffix) apartment size, common area, living area, age of building, number of elevators, number of entrances, state of the building, number of floor",normalized
82,application_train.csv,FLOORSMAX_MEDI,"Normalized information about building where the client lives, What is average (_AVG suffix), modus (_MODE suffix), median (_MEDI suffix) apartment size, common area, living area, age of building, number of elevators, number of entrances, state of the building, number of floor",normalized
83,application_train.csv,FLOORSMIN_MEDI,"Normalized information about building where the client lives, What is average (_AVG suffix), modus (_MODE suffix), median (_MEDI suffix) apartment size, common area, living area, age of building, number of elevators, number of entrances, state of the building, number of floor",normalized
84,application_train.csv,LANDAREA_MEDI,"Normalized information about building where the client lives, What is average (_AVG suffix), modus (_MODE suffix), median (_MEDI suffix) apartment size, common area, living area, age of building, number of elevators, number of entrances, state of the building, number of floor",normalized
85,application_train.csv,LIVINGAPARTMENTS_MEDI,"Normalized information about building where the client lives, What is average (_AVG suffix), modus (_MODE suffix), median (_MEDI suffix) apartment size, common area, living area, age of building, number of elevators, number of entrances, state of the building, number of floor",normalized
86,application_train.csv,LIVINGAREA_MEDI,"Normalized information about building where the client lives, What is average (_AVG suffix), modus (_MODE suffix), median (_MEDI suffix) apartment size, common area, living area, age of building, number of elevators, number of entrances, state of the building, number of floor",normalized
87,application_train.csv,NONLIVINGAPARTMENTS_MEDI,"Normalized information about building where the client lives, What is average (_AVG suffix), modus (_MODE suffix), median (_MEDI suffix) apartment size, common area, living area, age of building, number of elevators, number of entrances, state of the building, number of floor",normalized
88,application_train.csv,NONLIVINGAREA_MEDI,"Normalized information about building where the client lives, What is average (_AVG suffix), modus (_MODE suffix), median (_MEDI suffix) apartment size, common area, living area, age of building, number of elevators, number of entrances, state of the building, number of floor",normalized
89,application_train.csv,FONDKAPREMONT_MODE,"Normalized information about building where the client lives, What is average (_AVG suffix), modus (_MODE suffix), median (_MEDI suffix) apartment size, common area, living area, age of building, number of elevators, number of entrances, state of the building, number of floor",normalized
90,application_train.csv,HOUSETYPE_MODE,"Normalized information about building where the client lives, What is average (_AVG suffix), modus (_MODE suffix), median (_MEDI suffix) apartment size, common area, living area, age of building, number of elevators, number of entrances, state of the building, number of floor",normalized
91,application_train.csv,TOTALAREA_MODE,"Normalized information about building where the client lives, What is average (_AVG suffix), modus (_MODE suffix), median (_MEDI suffix) apartment size, common area, living area, age of building, number of elevators, number of entrances, state of the building, number of floor",normalized
92,application_train.csv,WALLSMATERIAL_MODE,"Normalized information about building where the client lives, What is average (_AVG suffix), modus (_MODE suffix), median (_MEDI suffix) apartment size, common area, living area, age of building, number of elevators, number of entrances, state of the building, number of floor",normalized
93,application_train.csv,EMERGENCYSTATE_MODE,"Normalized information about building where the client lives, What is average (_AVG suffix), modus (_MODE suffix), median (_MEDI suffix) apartment size, common area, living area, age of building, number of elevators, number of entrances, state of the building, number of floor",normalized
94,application_train.csv,OBS_30_CNT_SOCIAL_CIRCLE,How many observation of client's social surroundings with observable 30 DPD (days past due) default,
95,application_train.csv,DEF_30_CNT_SOCIAL_CIRCLE,How many observation of client's social surroundings defaulted on 30 DPD (days past due) ,
96,application_train.csv,OBS_60_CNT_SOCIAL_CIRCLE,How many observation of client's social surroundings with observable 60 DPD (days past due) default,
97,application_train.csv,DEF_60_CNT_SOCIAL_CIRCLE,How many observation of client's social surroundings defaulted on 60 (days past due) DPD,
98,application_train.csv,DAYS_LAST_PHONE_CHANGE,How many days before application did client change phone,
99,application_train.csv,FLAG_DOCUMENT_2,Did client provide document 2,
100,application_train.csv,FLAG_DOCUMENT_3,Did client provide document 3,
101,application_train.csv,FLAG_DOCUMENT_4,Did client provide document 4,
102,application_train.csv,FLAG_DOCUMENT_5,Did client provide document 5,
103,application_train.csv,FLAG_DOCUMENT_6,Did client provide document 6,
104,application_train.csv,FLAG_DOCUMENT_7,Did client provide document 7,
105,application_train.csv,FLAG_DOCUMENT_8,Did client provide document 8,
106,application_train.csv,FLAG_DOCUMENT_9,Did client provide document 9,
107,application_train.csv,FLAG_DOCUMENT_10,Did client provide document 10,
108,application_train.csv,FLAG_DOCUMENT_11,Did client provide document 11,
109,application_train.csv,FLAG_DOCUMENT_12,Did client provide document 12,
110,application_train.csv,FLAG_DOCUMENT_13,Did client provide document 13,
111,application_train.csv,FLAG_DOCUMENT_14,Did client provide document 14,
112,application_train.csv,FLAG_DOCUMENT_15,Did client provide document 15,
113,application_train.csv,FLAG_DOCUMENT_16,Did client provide document 16,
114,application_train.csv,FLAG_DOCUMENT_17,Did client provide document 17,
115,application_train.csv,FLAG_DOCUMENT_18,Did client provide document 18,
116,application_train.csv,FLAG_DOCUMENT_19,Did client provide document 19,
117,application_train.csv,FLAG_DOCUMENT_20,Did client provide document 20,
118,application_train.csv,FLAG_DOCUMENT_21,Did client provide document 21,
119,application_train.csv,AMT_REQ_CREDIT_BUREAU_HOUR,Number of enquiries to Credit Bureau about the client one hour before application,
120,application_train.csv,AMT_REQ_CREDIT_BUREAU_DAY,Number of enquiries to Credit Bureau about the client one day before application (excluding one hour before application),
121,application_train.csv,AMT_REQ_CREDIT_BUREAU_WEEK,Number of enquiries to Credit Bureau about the client one week before application (excluding one day before application),
122,application_train.csv,AMT_REQ_CREDIT_BUREAU_MON,Number of enquiries to Credit Bureau about the client one month before application (excluding one week before application),
123,application_train.csv,AMT_REQ_CREDIT_BUREAU_QRT,Number of enquiries to Credit Bureau about the client 3 month before application (excluding one month before application),
124,application_train.csv,AMT_REQ_CREDIT_BUREAU_YEAR,Number of enquiries to Credit Bureau about the client one day year (excluding last 3 months before application),"""

metadata_df_full = pd.read_csv(io.StringIO(METADATA_FILE_CONTENT))
target_metadata = metadata_df_full[metadata_df_full['Table'].str.strip() == 'application_train.csv']
metadata_string = target_metadata.to_string()

# --- Agent and Workflow Initialization ---
data_profiler = DataProfilerAgent(
    llm_config=llm_config,
    code_execution_config={
        "work_dir": str(PROJECT_ROOT),
        "use_docker": False,
    }
)

metadata_analyst = MetadataAnalystAgent(llm_config=llm_config)

data_cleaner = DataCleanerExecutorAgent(
    llm_config=llm_config,
    code_execution_config={
        "work_dir": str(CLEANED_DIR),
        "use_docker": False,
    }
)

code_reviewer = CodeReviewerAgent(
    llm_config=llm_config,
    code_execution_config={
        "work_dir": str(PROJECT_ROOT), # Allow Code Reviewer to access project files for debugging
        "use_docker": False,
    }
)

# --- Group Chat Orchestration ---
groupchat = autogen.GroupChat(
    agents=[data_profiler, metadata_analyst, data_cleaner, code_reviewer],
    messages=[],
    max_round=20, # Increased max_round to allow more iterations for debugging
    speaker_selection_method="auto"
)

manager = autogen.GroupChatManager(groupchat=groupchat, llm_config=llm_config)

# --- Task Execution ---
initial_prompt = f"""
Please initiate a comprehensive data cleaning workflow.

**Step 1: Data Profiling**
First, the Data_Profiler_Agent will load the dataset located at the absolute path: '{TARGET_DATA_PATH}'.
It will perform a detailed data profile of this CSV file by generating and executing Python code, then summarize its characteristics.

**Step 2: Data Cleaning Plan Generation**
Once the data profile is complete and 'DATA_PROFILING_COMPLETE' is stated, the Metadata_Analyst_Agent will use this profile ALONGSIDE the following metadata description to generate a detailed data cleaning plan. It will state 'DATA_CLEANING_PLAN_COMPLETE' upon finishing.

METADATA DESCRIPTION:
{metadata_string}

**Step 3: Data Cleaning Code Execution and Review**
Once the plan is complete and 'DATA_CLEANING_PLAN_COMPLETE' is stated, the Data_Cleaner_Executor_Agent will receive the cleaning plan and implement it by writing and executing Python code. The cleaned file should be saved in the directory: '{CLEANED_DIR}'. It will state 'DATA_CLEANING_EXECUTION_COMPLETE' upon finishing.
The Code_Reviewer_Agent will then monitor the execution, provide feedback for any errors or necessary improvements, and state 'CODE_REVIEW_COMPLETE' upon finishing its review.
"""

print("\nInitiating the data cleaning workflow with refactored agents and logic...")
data_profiler.initiate_chat(
    manager,
    message=initial_prompt,
)

print("Data cleaning workflow finished.")