# Automated Context Extraction and EDA

## Introduction

This project provides a framework for automating the extraction of context from user prompts and performing exploratory data analysis (EDA) on datasets using AI agents powered by language models. The context extraction identifies key attributes (e.g., problem type, target variable, evaluation metrics) from prompts related to tasks like loan approval systems, while the EDA automation leverages multiple language models to generate insights from data.

## Project Structure

- `context_extract.py`: Script for extracting context from user prompts and saving it to JSON files.
- `automate_eda.py`: Script for automating exploratory data analysis on datasets.
- `requirements.txt`: List of dependencies required for the project.
- `.env`: File for storing environment variables (not included in the repository).

## Features

- **Context Extraction**: Analyzes user prompts to determine problem type, target variable, and evaluation metrics.
- **Automated EDA**: Performs exploratory data analysis on datasets using AI agents.
- **Multiple Language Models**: Utilizes Gemini and Ollama models with a fallback mechanism for robustness.
- **JSON Output**: Saves extracted context to uniquely named JSON files.
- **EDA Reports**: Generates summary statistics, data types, visualizations, and more for datasets.

## Installation

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/yourusername/your-repo.git
   cd your-repo
   ```

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Set Up Environment Variables**:
   Create a `.env` file in the project root:
   ```bash
   echo "GOOGLE_API_KEY=your_google_api_key" > .env
   ```

   Ensure the Google API key is valid for accessing the Gemini model.

## Usage

### Context Extraction

Run the context extraction script:
```bash
python context_extract.py
```

The default prompt is hardcoded in `context_extract.py`:
```
Help me build a loan approval/rejection system by prioritizing loan approvals while minimizing defaults. The system should approve loans based on user data and financial history, ensuring that as many loans as possible are granted without compromising the risk of defaults.
```
To use a different prompt, modify the `prompt` variable in `context_extract.py`.

### EDA Automation

Run the EDA automation script:
```bash
python automate_eda.py
```

Ensure your dataset is located at `../dataset/application_test.csv` relative to the script, or update the `file_path` variable in `automate_eda.py` to point to your dataset.

## Configuration

### Environment Variables

The `.env` file must include:
```
GOOGLE_API_KEY=your_google_api_key
```

### Model Configurations

- **Context Extraction**: Models are defined in `OAI_CONFIG_LIST` in `context_extract.py`. Modify this list to adjust the Gemini and Ollama models used.
- **EDA Automation**: Models are specified in `model_configs` in `automate_eda.py`. Update this list to change the primary (Gemini) and fallback (Ollama) models.

## Workflow

### Context Extraction

The workflow in `context_extract.py` involves a specialized agent that processes a user prompt in a single execution:

1. **Prompt Analysis**:
   - The agent determines:
     - **Problem Type**: Defaults to "classification" (e.g., approve/reject).
     - **Target Variable**: Defaults to "loan approval status" if unspecified.
     - **Evaluation Metrics**: Based on prompt emphasis (e.g., "recall" for maximizing approvals, "precision" for minimizing false approvals, or ["accuracy", "F1-score"] otherwise).
   - Example: For the default prompt, it identifies a classification problem, targets "loan approval status," and prioritizes "recall" due to the focus on maximizing approvals.

2. **Script Generation**:
   - The agent generates a Python script that:
     - Defines the extracted context as a dictionary.
     - Implements a filename counter (e.g., `output_1.json`, `output_2.json`) to ensure uniqueness.
     - Saves the dictionary to a JSON file using `json.dump()`.

3. **Execution**:
   - The user proxy executes the generated script, saving the context to a file in the `json_configs` directory and printing a confirmation (e.g., "Context saved successfully to output_1.json").

4. **Termination**:
   - The agent terminates the process by responding with "TERMINATE" after execution.

### EDA Automation

The workflow in `automate_eda.py` automates EDA on a dataset using multiple language models:

1. **Dataset Loading**:
   - Loads the dataset (e.g., `application_test.csv`) into a Pandas DataFrame.

2. **Agent Creation**:
   - Initializes AI agents with:
     - Primary model: Gemini (`gemini-2.0-flash`).
     - Fallback models: Ollama (`gemma3:1b`, `qwen3:0.6b`).
   - Each agent is linked to the DataFrame for querying.

3. **Query Execution**:
   - Runs predefined EDA queries (e.g., dataset shape, missing values, summary statistics, correlation heatmap) using a fallback mechanism:
     - Tries the primary model first.
     - If it fails, attempts the next model in sequence until success or all fail.

4. **Output**:
   - Prints the results of each query, providing insights such as data types, value counts, and visualizations (e.g., bar charts, heatmaps).

## Contributing

Contributions are welcome! To contribute:

1. Fork the repository.
2. Create a branch for your feature or bugfix:
   ```bash
   git checkout -b feature-name
   ```
3. Commit your changes:
   ```bash
   git commit -m "Description of changes"
   ```
4. Push to your fork:
   ```bash
   git push origin feature-name
   ```
5. Submit a pull request.

For bug reports or feature requests, please open an issue on the GitHub repository.