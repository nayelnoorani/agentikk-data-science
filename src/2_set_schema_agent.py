import pandas as pd
import numpy as np
import google.generativeai as genai
from typing import Dict, Any, Tuple
import json

class GeminiSchemaAgent:
    def __init__(self, api_key: str):
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-pro')
        self.data = None
        self.schema = None
        
    def load_data(self, filepath: str) -> None:
        """Load the CSV data."""
        self.data = pd.read_csv(filepath)
        
    def get_data_summary(self) -> str:
        """Create a summary of the data for Gemini to analyze."""
        summary = []
        summary.append(f"Dataset shape: {self.data.shape[0]} rows, {self.data.shape[1]} columns")
        summary.append("\nColumns and their properties:")
        for col in self.data.columns:
            dtype = str(self.data[col].dtype)
            n_unique = self.data[col].nunique()
            n_missing = self.data[col].isnull().sum()
            sample_values = self.data[col].dropna().sample(min(3, len(self.data))).tolist()
            
            col_info = (f"\n- {col}:"
                       f"\n  Type: {dtype}"
                       f"\n  Unique values: {n_unique}"
                       f"\n  Missing values: {n_missing}"
                       f"\n  Sample values: {sample_values}")
            summary.append(col_info)
        return "\n".join(summary)
    
    def transform_data(self, schema: Dict) -> pd.DataFrame:
        """Transform the data according to the schema suggestions."""
        transformed_data = self.data.copy()
        transformations_applied = []
        
        # Handle feature categorization
        if 'feature_categories' in schema:
            # Create category labels for features
            for category, features in schema['feature_categories'].items():
                for feature in features:
                    if feature in transformed_data.columns:
                        transformations_applied.append(f"Categorized {feature} as {category}")
        
        # Apply derived features
        if 'derived_features' in schema:
            for feature_name, calculation in schema['derived_features'].items():
                try:
                    # Example derivations based on common patterns
                    if 'ratio' in calculation.lower():
                        cols = [col for col in transformed_data.columns if col in calculation]
                        if len(cols) >= 2:
                            transformed_data[feature_name] = transformed_data[cols[0]] / transformed_data[cols[1]].replace(0, np.nan)
                            transformations_applied.append(f"Created ratio feature: {feature_name}")
                            
                    elif 'average' in calculation.lower() or 'mean' in calculation.lower():
                        cols = [col for col in transformed_data.columns if col in calculation]
                        if cols:
                            transformed_data[feature_name] = transformed_data[cols].mean(axis=1)
                            transformations_applied.append(f"Created average feature: {feature_name}")
                            
                    elif 'time' in calculation.lower() or 'duration' in calculation.lower():
                        # Handle time-based features
                        time_cols = [col for col in transformed_data.columns if 'time' in col.lower() or 'date' in col.lower()]
                        if time_cols:
                            transformed_data[feature_name] = transformed_data[time_cols[0]].diff()
                            transformations_applied.append(f"Created time-based feature: {feature_name}")
                except Exception as e:
                    transformations_applied.append(f"Failed to create {feature_name}: {str(e)}")
        
        # Apply data quality recommendations
        if 'data_quality' in schema:
            # Handle missing values
            if 'missing_data_handling' in schema['data_quality']:
                for col in transformed_data.columns:
                    if transformed_data[col].isnull().sum() > 0:
                        if transformed_data[col].dtype in ['int64', 'float64']:
                            transformed_data[col].fillna(transformed_data[col].mean(), inplace=True)
                            transformations_applied.append(f"Filled missing values in {col} with mean")
                        else:
                            transformed_data[col].fillna(transformed_data[col].mode()[0], inplace=True)
                            transformations_applied.append(f"Filled missing values in {col} with mode")
        
        # Apply schema optimizations
        if 'schema_optimization' in schema:
            # Remove suggested features
            if 'feature_removal' in schema['schema_optimization']:
                for feature in schema['schema_optimization']['feature_removal']:
                    if feature in transformed_data.columns:
                        transformed_data.drop(columns=[feature], inplace=True)
                        transformations_applied.append(f"Removed feature: {feature}")
            
            # Apply transformations
            if 'data_transformations' in schema['schema_optimization']:
                for transform in schema['schema_optimization']['data_transformations']:
                    if 'normalize' in transform.lower():
                        numeric_cols = transformed_data.select_dtypes(include=[np.number]).columns
                        for col in numeric_cols:
                            transformed_data[f"{col}_normalized"] = (transformed_data[col] - transformed_data[col].mean()) / transformed_data[col].std()
                            transformations_applied.append(f"Normalized feature: {col}")
        
        return transformed_data, transformations_applied

    def analyze_and_transform(self, target_column: str = None) -> Tuple[Dict[str, Any], pd.DataFrame, list]:
        """Analyze the data, get schema suggestions, and transform the data."""
        
        # Get schema suggestions
        schema = self.analyze_schema(target_column)
        
        # Transform data according to schema
        if 'error' not in schema:
            transformed_data, transformations = self.transform_data(schema)
        else:
            return schema, self.data, ["No transformations applied due to schema error"]
        
        return schema, transformed_data, transformations

    def analyze_schema(self, target_column: str = None) -> Dict[str, Any]:
        """Get schema suggestions from Gemini."""
        data_summary = self.get_data_summary()
        target_info = f"The target column indicating fraud is: {target_column}" if target_column else "No target column specified."
        
        prompt = f"""You are a fraud detection expert. Analyze this Ethereum blockchain transaction dataset and create a schema for fraud detection.

Current Data Summary:
{data_summary}

{target_info}

Respond ONLY with a JSON object. No explanations or additional text. The JSON must follow this exact structure:

{{
    "feature_categories": {{
        "transaction_features": ["list", "of", "transaction", "columns"],
        "temporal_features": ["list", "of", "time", "columns"],
        "behavioral_features": ["list", "of", "behavioral", "columns"],
        "token_features": ["list", "of", "token", "columns"]
    }},
    "derived_features": {{
        "feature_1_name": "calculation description",
        "feature_2_name": "calculation description"
    }},
    "feature_importance": {{
        "high_importance": ["list", "of", "important", "features"],
        "medium_importance": ["list", "of", "medium", "features"],
        "low_importance": ["list", "of", "low", "features"]
    }},
    "data_quality": {{
        "missing_data_handling": ["list", "of", "recommendations"],
        "feature_engineering": ["list", "of", "suggestions"],
        "data_cleaning": ["list", "of", "tasks"]
    }},
    "schema_optimization": {{
        "feature_selection": ["list", "of", "features", "to", "keep"],
        "feature_removal": ["list", "of", "features", "to", "remove"],
        "data_transformations": ["list", "of", "transformations"]
    }}
}}"""
        
        try:
            response = self.model.generate_content(prompt)
            
            # Try to find JSON in the response
            import re
            json_match = re.search(r'\{[\s\S]*\}', response.text)
            if json_match:
                try:
                    schema_suggestions = json.loads(json_match.group())
                    return schema_suggestions
                except json.JSONDecodeError as je:
                    return {
                        "error": f"Failed to parse JSON: {str(je)}",
                        "raw_response": response.text
                    }
            else:
                return {
                    "error": "No JSON found in response",
                    "raw_response": response.text
                }
            
        except Exception as e:
            return {
                "error": f"Failed to generate schema suggestions: {str(e)}",
                "raw_prompt": prompt
            }

def main():
    import os
    from dotenv import load_dotenv
    
    # Load environment variables
    load_dotenv()
    
    # Get API key
    api_key = os.getenv('GOOGLE_API_KEY')
    if not api_key:
        print("Error: GOOGLE_API_KEY not found in environment variables")
        return
        
    # Initialize agent
    agent = GeminiSchemaAgent(api_key)
    
    # Get input file and target column
    input_file = input("Enter the path to your CSV file: ")
    target_column = input("Enter the name of the fraud indicator column (or press Enter if none): ").strip()
    
    try:
        # Load data
        agent.load_data(input_file)
        
        # Get schema suggestions and transform data
        schema, transformed_data, transformations = agent.analyze_and_transform(target_column if target_column else None)
        
        # Save schema suggestions
        schema_file = 'gemini_schema_suggestions.json'
        with open(schema_file, 'w') as f:
            json.dump(schema, f, indent=2)
        
        # Save transformed data
        transformed_file = 'transformed_data.csv'
        transformed_data.to_csv(transformed_file, index=False)
        
        # Save transformation log
        log_file = 'transformation_log.txt'
        with open(log_file, 'w') as f:
            f.write('\n'.join(transformations))
            
        print(f"\nAnalysis complete!")
        print(f"- Schema suggestions saved to: {schema_file}")
        print(f"- Transformed data saved to: {transformed_file}")
        print(f"- Transformation log saved to: {log_file}")
        
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()
