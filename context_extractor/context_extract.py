import autogen
from autogen import UserProxyAgent, AssistantAgent, GroupChat, GroupChatManager
from pathlib import Path
from autogen.coding import LocalCommandLineCodeExecutor
from dotenv import load_dotenv
import os



load_dotenv()

api_key=os.getenv("GOOGLE_API_KEY")


# Load LLM configuration 

OAI_CONFIG_LIST = [
   {
     "model": "gemini-2.0-flash",
     "api_type": "google",
     "api_key": api_key
   },
   {
     "model":"gemma3:1b",
     "api_type": "ollama",
     "stream": False,
     "client_host":"http://localhost:11434"
   },
   {
     "model":"qwen3:0.6b",
     "api_type": "ollama",
     "stream": False,
     "client_host":"http://localhost:11434"
   }
]


config_list = OAI_CONFIG_LIST
llm_config = {"config_list": config_list, "cache_seed": 42}

#saving the json config to a fil
config_path = Path("json_configs")
config_path.mkdir(exist_ok=True)
code_executor = LocalCommandLineCodeExecutor(work_dir=config_path)

# Step 1: Define the User Proxy Agent
user_proxy = UserProxyAgent(
    name="UserProxy",
    system_message="A human admin. You will execute any code blocks provided by the agent and report the result. Reply TERMINATE when the agent gives the final TERMINATE command.",
    code_execution_config={"executor":code_executor},
    human_input_mode="NEVER",
    is_termination_msg=lambda messsage: "End" in messsage.get("content")
)


#Defining system message for specialized agent

system_message='''
You are a specialized agent that extracts context from user prompts and saves it to a file.

Your workflow is a **strict, one-time execution per user prompt** and must be followed precisely:

1.  **Analyze the User's Prompt**: First, silently analyze the user's request to determine the following attributes based on the rules provided:
    * **Problem Type**: Identify if it's a 'classification' problem (e.g., 'approve/reject', 'yes/no'). Default to 'classification' if unclear.
    * **Target Variable**: Identify the prediction goal (e.g., 'loan approval status'). Default to 'loan approval status' if not specified.
    * **Evaluation Metrics**:
        * If the prompt emphasizes maximizing approvals (e.g., 'provide more loans'), prioritize **'recall'**.
        * If the prompt emphasizes minimizing false approvals (e.g., 'discard only critical requests'), prioritize **'precision'**.
        * Otherwise, suggest **['accuracy', 'F1-score']**.

2.  **Write a Single Python Script**: Your sole output for this task is a **single Python script**. Do NOT output the JSON directly or offer to run the task again. The script you write will be executed once and must perform the following actions:
    * Import the `json` and `os` libraries.
    * Define the extracted context from step 1 as a Python dictionary variable.
    * Implement a counter for the filename. Start with `output_1.json`. Before saving, check if the file already exists. If it does, increment the counter (`output_2.json`, `output_3.json`, etc.) until you find a filename that does not exist. This logic ensures one unique file is created.
    * Use `json.dump()` to save the dictionary to the unique file.
    * **Print a simple, text-only confirmation message upon success. For example: `print(f"Context saved successfully to output_3.json")`. Do NOT use emojis or other special characters.**

3.  **Provide Your Response**: Your entire response MUST be only the complete Python script, enclosed in a ```python ... ``` block.

4.  **Final Action: Terminate**: **This is the most important rule.** Your task is complete once you have provided the script. After the script is executed, you will see a confirmation message from the user proxy (e.g., with the content "Context saved successfully to..."). **Your only permitted action at this point is to reply with a final message containing nothing but the word TERMINATE.**
    * **DO NOT** write code again.
    * **DO NOT** say thank you or offer further assistance.
    * **DO NOT** analyze the confirmation message for any other purpose.
    * Your job is done. Simply reply: **TERMINATE**
'''
# Step 2: Define Specialized Agents for Context Extraction
Context_Extractor_Agent = AssistantAgent(
    name="ClassificationDetector",
    system_message=system_message,
    llm_config=llm_config
)


# Step 3: Set Up Group Chat for Collaboration
group_chat = GroupChat(
    agents=[user_proxy, Context_Extractor_Agent],
    messages=[],
    max_round=10,
    speaker_selection_method='round_robin'
)

group_chat_manager = GroupChatManager(
    groupchat=group_chat,
    llm_config=llm_config,
)

# Step 4: Execute the Workflow
prompt = "Help me build a loan approval/rejection system by prioritizing loan approvals while minimizing defaults. The system should approve loans based on user data and financial history, ensuring that as many loans as possible are granted without compromising the risk of defaults."
prompt1="Help me build a loan approval/rejection system by priortizing precision over recall."
user_proxy.initiate_chat(
    group_chat_manager,
    message=prompt
)