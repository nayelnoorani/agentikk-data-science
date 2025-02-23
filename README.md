# agentikk-data-science - Hacklytics 2025 with DS club @ GT

Our project focuses on building an AI-based multi agent system that, given a dataset, will perform exploratory data analysis, feature engineering, model training, validation and testing, followed by insight generation. Each of the individual tasks will be taken care of by separate AI Agents. 

The first script (eda_agent) performs an initial analysis of the dataset. It takes in the raw csv file as input along with the api_key for Gemini calls. The agent starts by checking the row count of the dataframe, data types of all variables, duplicates, missing values, and memory usage. It removes the duplicate rows, fills in the missing values for numeric and categorical columns with their median and mode values respectively. This is followed by calculating basic statistics - for numerical columns, it provides mean, median, standard deviation, min, max, quantiles, and skewness whereas for categorical columns, it summarizes the unique value count and top 5 values. The last step is analyzing correlations for all numerical columns and producing a detailed EDA report and a cleaned version of the dataset.

The second script (set_schema_agent) analyzes the dataset structure and generates schema recommendations using Google’s Gemini AI. It categorizes features into relevant groups (e.g., transaction, temporal, behavioral), suggests derived features, and applies transformations such as missing value imputation and feature scaling. The transformed data is saved for downstream analysis, along with a schema suggestion report.

The last script (fraud_detection_agent) builds a fraud detection model using machine learning. It first analyzes the dataset to determine fraud prevalence and feature types. It then requests feature engineering recommendations from Gemini AI and implements transformations such as encoding categorical variables and handling numerical anomalies. The script trains a Random Forest model, evaluates its performance using AUC-ROC and classification metrics, and generates a final fraud detection report, highlighting key findings and feature importance.

The agents have been designed with system prompts with the potential of giving user prompts in future for more accurate results. Overall, these scripts work together to preprocess data, define schema, extract meaningful features, and train a fraud detection model, making the entire fraud analysis pipeline efficient and automated.
