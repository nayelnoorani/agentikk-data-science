# src/fraud_detection_agent.py

import os
import re
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv
import google.generativeai as genai
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt
import seaborn as sns

class FraudDetectionAgent:
    def __init__(self):
        """Initialize the Fraud Detection Agent"""
        self._setup_environment()
        self._setup_logging()
        self.current_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.results_dir = Path('analysis_results')
        self.results_dir.mkdir(exist_ok=True)
        
    def _setup_environment(self):
        """Set up environment variables and Gemini API"""
        load_dotenv()
        api_key = os.getenv('GOOGLE_API_KEY')
        if not api_key:
            raise ValueError("GOOGLE_API_KEY not found in .env file")
        
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-pro')
        
    def _setup_logging(self):
        """Configure logging to file only"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[logging.FileHandler('transformation_log.txt')]
        )
        
    def save_results(self, filename, content):
        """Save results to a file in the analysis_results directory"""
        filepath = self.results_dir / filename
        with open(filepath, 'w') as f:
            if isinstance(content, (dict, list)):
                json.dump(content, f, indent=2)
            else:
                f.write(str(content))
        logging.info(f"Saved results to {filepath}")
        
    def analyze_dataset(self, df):
        """Initial dataset analysis"""
        analysis = {
            'total_rows': len(df),
            'total_columns': len(df.columns),
            'missing_values': df.isnull().sum().to_dict(),
            'fraud_rate': (df['FLAG'].sum() / len(df)) * 100,
            'numeric_columns': df.select_dtypes(include=['int64', 'float64']).columns.tolist(),
            'categorical_columns': df.select_dtypes(include=['object', 'category']).columns.tolist()
        }
        
        self.save_results(f'initial_analysis_{self.current_timestamp}.json', analysis)
        return analysis
        
    def get_feature_engineering_suggestions(self, df, analysis):
        """Get feature engineering suggestions from Gemini"""
        prompt = (
            f"As a fraud detection expert, analyze this dataset:\n\n"
            f"Numeric columns: {analysis['numeric_columns']}\n"
            f"Categorical columns: {analysis['categorical_columns']}\n"
            f"Fraud rate: {analysis['fraud_rate']:.2f}%\n\n"
            "Suggest specific feature engineering steps for fraud detection.\n"
            "Focus on:\n"
            "1. Temporal patterns if date/time exists\n"
            "2. Statistical aggregations\n"
            "3. Interaction features\n"
            "4. Risk indicators\n\n"
            "Return a JSON with these exact keys:\n"
            "{\n"
            '  "temporal_features": ["feature1", "feature2"],\n'
            '  "statistical_features": ["feature1", "feature2"],\n'
            '  "interaction_features": ["feature1", "feature2"],\n'
            '  "risk_indicators": ["feature1", "feature2"]\n'
            "}"
        )
        
        try:
            response = self.model.generate_content(prompt)
            suggestions = json.loads(response.text)
            self.save_results(f'feature_suggestions_{self.current_timestamp}.json', suggestions)
            return suggestions
        except Exception as e:
            logging.error(f"Error getting feature suggestions: {e}")
            return {
                "temporal_features": [],
                "statistical_features": [],
                "interaction_features": [],
                "risk_indicators": []
            }
            
    def engineer_features(self, df, suggestions):
        """Create features based on suggestions"""
        features = df.copy()
        feature_log = []
        
        try:
            # Handle temporal features if datetime columns exist
            datetime_cols = df.select_dtypes(include=['datetime64']).columns
            for col in datetime_cols:
                new_features = {
                    f'{col}_hour': df[col].dt.hour,
                    f'{col}_day': df[col].dt.day,
                    f'{col}_dayofweek': df[col].dt.dayofweek
                }
                features.update(new_features)
                feature_log.append(f"Created temporal features for {col}")
                
            # Create statistical aggregations
            numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
            for col in numeric_cols:
                if col != 'FLAG':
                    new_features = {
                        f'{col}_rolling_mean': df[col].rolling(window=3).mean(),
                        f'{col}_rolling_std': df[col].rolling(window=3).std()
                    }
                    features.update(new_features)
                    feature_log.append(f"Created statistical features for {col}")
                    
            # Handle missing values
            features = features.fillna(features.mean())
            
            # Save feature engineering log
            self.save_results(f'feature_engineering_log_{self.current_timestamp}.txt', '\n'.join(feature_log))
            
            return features
            
        except Exception as e:
            logging.error(f"Error in feature engineering: {e}")
            return df
            
    # Add this method to the FraudDetectionAgent class

    def identify_index_columns(self, df):
        """Identify columns that are likely to be index columns"""
        index_columns = []
        reasons = {}
        
        for col in df.columns:
            if col == 'FLAG':
                continue
                
            # Check 1: Almost unique values (high unique ratio)
            unique_ratio = df[col].nunique() / len(df)
            
            # Check 2: Column name contains typical index keywords
            index_keywords = ['id', 'index', 'key', 'number', 'no', 'num', 'uuid', 'guid', 'hash']
            has_index_keyword = any(keyword in col.lower() for keyword in index_keywords)
            
            # Check 3: Monotonic increasing/decreasing
            is_monotonic = df[col].is_monotonic_increasing or df[col].is_monotonic_decreasing
            
            # Check 4: All unique values
            is_unique = df[col].is_unique
            
            # Check 5: Contains hash-like patterns (for string columns)
            is_hash_like = False
            if df[col].dtype == object:
                # Sample first non-null value
                sample_val = df[col].dropna().iloc[0] if not df[col].dropna().empty else ''
                if isinstance(sample_val, str):
                    # Check for hash-like patterns (long hex strings)
                    is_hash_like = bool(re.match(r'^[a-fA-F0-9]{32,}$', sample_val))
                    
            # Decision logic
            if any([
                unique_ratio > 0.9,  # Very high unique ratio
                (has_index_keyword and unique_ratio > 0.5),  # Index keyword with moderate uniqueness
                (is_monotonic and unique_ratio > 0.5),  # Monotonic with moderate uniqueness
                is_unique,  # Completely unique values
                is_hash_like  # Hash-like strings
            ]):
                index_columns.append(col)
                reasons[col] = {
                    'unique_ratio': unique_ratio,
                    'has_index_keyword': has_index_keyword,
                    'is_monotonic': is_monotonic,
                    'is_unique': is_unique,
                    'is_hash_like': is_hash_like
                }
                
        return index_columns, reasons

    def identify_columns_to_drop(self, df):
        """Use both Gemini and direct analysis to identify columns that should be dropped"""
        try:
            # First, identify index columns directly
            index_columns, index_reasons = self.identify_index_columns(df)
            
            # Create column analysis for remaining columns
            column_info = {}
            for col in df.columns:
                if col not in index_columns and col != 'FLAG':
                    column_info[col] = {
                        'unique_count': df[col].nunique(),
                        'total_count': len(df),
                        'unique_ratio': df[col].nunique() / len(df),
                        'dtype': str(df[col].dtype),
                        'sample_values': df[col].head(3).tolist(),
                        'null_count': df[col].isnull().sum()
                    }
            
            # Create prompt for Gemini
            prompt = f"""
            Analyze these columns in a fraud detection dataset:

            {json.dumps(column_info, indent=2)}

            Some columns have already been identified as index columns and will be dropped:
            {json.dumps(index_columns, indent=2)}

            Identify additional columns that should be dropped because they:
            1. Are timestamp columns that are redundant
            2. Have too many unique values without fraud signal
            3. Have no predictive value for fraud
            4. Are redundant or highly correlated with other columns

            Return a JSON with these keys:
            {{
                "additional_columns_to_drop": ["col1", "col2"],
                "reasons": {{"col1": "reason", "col2": "reason"}}
            }}
            """

            response = self.model.generate_content(prompt)
            gemini_analysis = json.loads(response.text)
            
            # Combine results
            all_columns_to_drop = index_columns + gemini_analysis.get('additional_columns_to_drop', [])
            
            # Combine reasons
            all_reasons = {}
            for col in index_columns:
                all_reasons[col] = f"Index column: {json.dumps(index_reasons[col])}"
            all_reasons.update(gemini_analysis.get('reasons', {}))
            
            # Save complete analysis
            analysis_results = {
                'index_columns': {
                    'columns': index_columns,
                    'reasons': index_reasons
                },
                'additional_columns': {
                    'columns': gemini_analysis.get('additional_columns_to_drop', []),
                    'reasons': gemini_analysis.get('reasons', {})
                },
                'final_columns_to_drop': all_columns_to_drop,
                'final_reasons': all_reasons
            }
            
            self.save_results(
                f'column_drop_analysis_{self.current_timestamp}.json',
                analysis_results
            )
            
            return all_columns_to_drop, all_reasons
            
        except Exception as e:
            logging.error(f"Error in column analysis: {e}")
            return [], {}

    def preprocess_features(self, df):
        """Preprocess features including handling categorical and hex values"""
        features = df.copy()
        preprocessing_log = []
        
        try:
            # First identify columns to drop
            columns_to_drop, drop_reasons = self.identify_columns_to_drop(features)
            
            # Log columns being dropped
            preprocessing_log.append("\nColumns dropped from analysis:")
            for col in columns_to_drop:
                if col in features.columns:  # Check if column exists
                    features.drop(col, axis=1, inplace=True)
                    preprocessing_log.append(f"- Dropped {col}: {drop_reasons.get(col, 'No reason provided')}")
            
            # Identify column types in remaining columns
            numeric_cols = []
            categorical_cols = []
            hex_cols = []
            
            for col in features.columns:
                if col == 'FLAG':
                    continue
                    
                # Check if column contains hex values
                if (features[col].dtype == object and 
                    isinstance(features[col].iloc[0], str) and 
                    features[col].str.contains('0x').any()):
                    hex_cols.append(col)
                # Check if column is numeric
                elif pd.api.types.is_numeric_dtype(features[col]):
                    numeric_cols.append(col)
                else:
                    categorical_cols.append(col)
            
            # Log column types
            preprocessing_log.append("\nColumn type classification:")
            preprocessing_log.append(f"Numeric columns: {numeric_cols}")
            preprocessing_log.append(f"Categorical columns: {categorical_cols}")
            preprocessing_log.append(f"Hex columns: {hex_cols}")
            
            # Handle hex values - convert to numeric
            for col in hex_cols:
                try:
                    # Convert hex to integers
                    features[f'{col}_numeric'] = features[col].apply(
                        lambda x: int(str(x), 16) if pd.notnull(x) and isinstance(x, str) else None
                    )
                    features.drop(col, axis=1, inplace=True)
                    preprocessing_log.append(f"Converted hex column {col} to numeric")
                except Exception as e:
                    preprocessing_log.append(f"Error converting hex column {col}: {e}")
                    features.drop(col, axis=1, inplace=True)
            
            # Handle categorical columns
            label_encoders = {}
            for col in categorical_cols:
                try:
                    le = LabelEncoder()
                    features[col] = le.fit_transform(features[col].fillna('missing'))
                    label_encoders[col] = le
                    preprocessing_log.append(f"Label encoded column {col}")
                except Exception as e:
                    preprocessing_log.append(f"Error encoding column {col}: {e}")
                    features.drop(col, axis=1, inplace=True)
            
            # Handle missing values in numeric columns
            for col in numeric_cols:
                features[col] = pd.to_numeric(features[col], errors='coerce')
                features[col].fillna(features[col].mean(), inplace=True)
            
            # Save preprocessing log
            self.save_results(
                f'preprocessing_log_{self.current_timestamp}.txt',
                '\n'.join(preprocessing_log)
            )
            
            return features, preprocessing_log
            
        except Exception as e:
            logging.error(f"Error in preprocessing: {str(e)}")
            raise
    
    def train_evaluate_model(self, features, target):
        """Train and evaluate the fraud detection model"""
        try:
            # Preprocess features
            processed_features, preprocessing_log = self.preprocess_features(features)
            
            # Remove any non-numeric columns that might remain
            numeric_features = processed_features.select_dtypes(include=['int64', 'float64'])
            
            if 'FLAG' in numeric_features.columns:
                X = numeric_features.drop(['FLAG'], axis=1)
            else:
                X = numeric_features
                
            # Prepare data
            y = target
            
            # Log feature shapes
            logging.info(f"Features shape after preprocessing: {X.shape}")
            logging.info(f"Target shape: {y.shape}")
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Train model
            model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                class_weight='balanced'
            )
            
            model.fit(X_train_scaled, y_train)
            
            # Get predictions
            y_pred = model.predict(X_test_scaled)
            y_prob = model.predict_proba(X_test_scaled)[:, 1]
            
            # Calculate metrics
            results = {
                'auc_roc': float(roc_auc_score(y_test, y_prob)),
                'classification_report': classification_report(y_test, y_pred),
                'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
                'feature_importance': dict(zip(X.columns, model.feature_importances_.tolist())),
                'preprocessing_summary': preprocessing_log
            }
            
            # Save results
            self.save_results(f'model_results_{self.current_timestamp}.json', results)
            
            # Plot feature importance
            self.plot_feature_importance(model, X.columns)
            
            return results, model
            
        except Exception as e:
            logging.error(f"Error in model training: {str(e)}")
            raise
            
    def plot_feature_importance(self, model, feature_names):
        """Create feature importance plot"""
        plt.figure(figsize=(12, 6))
        importance = model.feature_importances_
        indices = np.argsort(importance)[::-1]
        
        plt.title("Feature Importances for Fraud Detection")
        plt.bar(range(len(importance)), importance[indices])
        plt.xticks(
            range(len(importance)),
            [feature_names[i] for i in indices],
            rotation=45,
            ha='right'
        )
        plt.tight_layout()
        plt.savefig(self.results_dir / f'feature_importance_{self.current_timestamp}.png')
        plt.close()
        
    def generate_report(self, analysis, results):
        """Generate comprehensive fraud detection report"""
        report_content = f"""
# Fraud Detection Analysis Report
Generated on: {self.current_timestamp}

## 1. Dataset Overview
- Total Records: {analysis['total_rows']}
- Features Analyzed: {analysis['total_columns']}
- Fraud Rate: {analysis['fraud_rate']:.2f}%

## 2. Model Performance
- AUC-ROC Score: {results['auc_roc']:.4f}

### Classification Report
```
{results['classification_report']}
```

## 3. Top Important Features
"""
        
        # Add top features
        sorted_features = sorted(
            results['feature_importance'].items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        for feature, importance in sorted_features[:10]:
            report_content += f"- {feature}: {importance:.4f}\n"
            
        # Save report
        self.save_results(f'fraud_detection_report_{self.current_timestamp}.md', report_content)
            
    def process_data(self, df):
        """Main method to process data and detect fraud"""
        try:
            # Initial analysis
            logging.info("Starting fraud detection analysis...")
            analysis = self.analyze_dataset(df)
            
            # Get feature engineering suggestions
            suggestions = self.get_feature_engineering_suggestions(df, analysis)
            
            # Engineer features
            logging.info("Engineering features...")
            features = self.engineer_features(df, suggestions)
            
            # Train and evaluate model
            logging.info("Training model...")
            results, model = self.train_evaluate_model(features, df['FLAG'])
            
            # Generate report
            self.generate_report(analysis, results)
            
            logging.info("Fraud detection analysis completed successfully")
            return results, model
            
        except Exception as e:
            logging.error(f"Error in fraud detection process: {e}")
            raise

def main():
    """Main function to run the fraud detection agent"""
    try:
        # Get file path from user
        file_path = input("Please enter the path to your CSV file: ").strip()
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
            
        # Read the CSV file
        df = pd.read_csv(file_path)
        
        # Initialize and run the agent
        agent = FraudDetectionAgent()
        results, model = agent.process_data(df)
        
        print("Analysis completed. Check the 'analysis_results' directory for all outputs.")
        
    except Exception as e:
        logging.error(f"Error in main execution: {str(e)}")
        print("An error occurred. Check transformation_log.txt for details.")
        sys.exit(1)

if __name__ == "__main__":
    main()