import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import google.generativeai as genai
from datetime import datetime
import json
import time
import os
from dotenv import load_dotenv

class EDAAgent:
    def __init__(self, api_key):
        """Initialize the EDA agent with API key"""
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-pro')
        self.cleaning_steps = []  # Add this to track cleaning steps
        
    def load_data(self, file_path):
        """Load the CSV file and store original data"""
        self.df = pd.read_csv(file_path)
        self.original_df = self.df.copy()
        self.original_file = file_path
        print(f"Loaded dataset with shape: {self.df.shape}")

    def document_cleaning_step(self, description, details=None):
        """Helper method to document cleaning steps"""
        step = {"description": description}
        if details:
            step["details"] = details
        self.cleaning_steps.append(step)
        
    def get_dataset_stats(self, df):
        """Get comprehensive statistics for entire dataset"""
        try:
            stats = {
                'shape': list(df.shape),
                'missing_values': {col: int(count) for col, count in df.isnull().sum().items()},
                'dtypes': {col: str(dtype) for col, dtype in df.dtypes.items()},
                'duplicates': int(len(df[df.duplicated()])),
                'memory_usage': float(df.memory_usage(deep=True).sum() / 1024**2),  # In MB
            }
            
            # Calculate statistics for numeric columns
            numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
            stats['numeric_summary'] = {}
            
            for col in numeric_cols:
                q1 = float(df[col].quantile(0.25))
                q3 = float(df[col].quantile(0.75))
                iqr = q3 - q1
                outlier_count = len(df[(df[col] < q1 - 1.5 * iqr) | (df[col] > q3 + 1.5 * iqr)])
                
                stats['numeric_summary'][col] = {
                    'mean': float(df[col].mean()),
                    'median': float(df[col].median()),
                    'std': float(df[col].std()),
                    'min': float(df[col].min()),
                    'max': float(df[col].max()),
                    'q1': q1,
                    'q3': q3,
                    'outliers': outlier_count,
                    'skewness': float(df[col].skew())
                }
            
            # Handle duplicate rows
            if stats['duplicates'] > 0:
                self.df.drop_duplicates(inplace=True)
                self.document_cleaning_step(
                    "Removed duplicate rows",
                    {"count": stats['duplicates']}
                )

            # Handle missing values
            for col, count in stats['missing_values'].items():
                if count > 0:
                    if col in numeric_cols:
                        # Fill numeric columns with median
                        self.df[col].fillna(self.df[col].median(), inplace=True)
                        self.document_cleaning_step(
                            f"Filled missing values in '{col}' with median",
                            {"count": count, "median_value": float(self.df[col].median())}
                        )
                    else:
                        # Fill categorical columns with mode
                        self.df[col].fillna(self.df[col].mode()[0], inplace=True)
                        self.document_cleaning_step(
                            f"Filled missing values in '{col}' with mode",
                            {"count": count, "mode_value": str(self.df[col].mode()[0])}
                        )
            
            # For categorical columns
            cat_cols = df.select_dtypes(include=['object', 'category']).columns
            stats['categorical_summary'] = {}
            
            for col in cat_cols:
                value_counts = df[col].value_counts()
                stats['categorical_summary'][col] = {
                    'unique_count': int(df[col].nunique()),
                    'top_5_values': {str(k): int(v) for k, v in value_counts.head().items()},
                    'completeness': float((len(df) - df[col].isnull().sum()) / len(df) * 100)
                }
            
            return stats
        except Exception as e:
            print(f"Error in get_dataset_stats: {str(e)}")
            return None

    def analyze_column_stats(self, col_name, stats, is_numeric):
        """Generate analysis prompt for a single column"""
        try:
            if is_numeric:
                col_stats = stats['numeric_summary'][col_name]
                prompt = f"""
                Analyze this numerical column brief statistical summary (2-3 sentences):
                Column: {col_name}
                - Mean: {col_stats['mean']:.2f}
                - Median: {col_stats['median']:.2f}
                - Standard Deviation: {col_stats['std']:.2f}
                - Outliers: {col_stats['outliers']}
                - Skewness: {col_stats['skewness']:.2f}
                
                Focus on:
                1. Distribution shape (based on skewness)
                2. Outlier presence and impact
                3. Central tendency
                """
            else:
                col_stats = stats['categorical_summary'][col_name]
                prompt = f"""
                Analyze this categorical column brief summary (2-3 sentences):
                Column: {col_name}
                - Unique Values: {col_stats['unique_count']}
                - Completeness: {col_stats['completeness']:.2f}%
                - Top Values: {json.dumps(col_stats['top_5_values'], indent=2)}
                
                Focus on:
                1. Value distribution
                2. Dominant categories
                3. Data completeness
                """
            
            response = self.model.generate_content(prompt)
            return response.text.strip()
        except Exception as e:
            print(f"Error analyzing column {col_name}: {str(e)}")
            return f"Could not analyze column {col_name} due to an error."

    def analyze_correlations(self):
        """Analyze correlations between numerical variables"""
        try:
            numeric_cols = self.df.select_dtypes(include=['int64', 'float64']).columns
            
            if len(numeric_cols) < 2:
                return "Not enough numerical columns for correlation analysis."
            
            # Calculate correlations
            corr_matrix = self.df[numeric_cols].corr()
            
            # Create correlation heatmap
            plt.figure(figsize=(10, 8))
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
            plt.title('Correlation Matrix')
            plt.tight_layout()
            plt.savefig('correlations.png')
            plt.close()
            
            # Find strong correlations
            strong_correlations = []
            for i in range(len(numeric_cols)):
                for j in range(i + 1, len(numeric_cols)):
                    corr = abs(corr_matrix.iloc[i, j])
                    if corr > 0.5:
                        strong_correlations.append({
                            'columns': [numeric_cols[i], numeric_cols[j]],
                            'correlation': float(corr)
                        })
            
            # Generate analysis for strong correlations
            if strong_correlations:
                prompt = f"""
                Analyze these strong correlations (2-3 sentences):
                {json.dumps(strong_correlations, indent=2)}
                Focus on the practical implications of these relationships.
                """
                response = self.model.generate_content(prompt)
                return response.text.strip()
            else:
                return "No strong correlations (>0.5) found between numerical variables."
                
        except Exception as e:
            print(f"Error in correlation analysis: {str(e)}")
            return "Could not complete correlation analysis due to an error."

    def generate_report(self):
        """Generate comprehensive EDA report"""
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            report_file = f'eda_report_{timestamp}.md'
            cleaned_file = f'cleaned_data_{timestamp}.csv'
            
            # Get complete statistics
            print("Calculating dataset statistics...")
            stats = self.get_dataset_stats(self.df)
            
            # Start report
            report = ["# Exploratory Data Analysis Report\n"]
            report.append(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            
            # Dataset overview
            report.extend([
                "## Dataset Overview\n",
                f"- Rows: {stats['shape'][0]:,}",
                f"- Columns: {stats['shape'][1]:,}",
                f"- Memory Usage: {stats['memory_usage']:.2f} MB",
                f"- Duplicate Rows: {stats['duplicates']:,}\n",
                "\n### Missing Values\n"
            ])
            
            # Missing values analysis
            missing_vals = {col: count for col, count in stats['missing_values'].items() if count > 0}
            if missing_vals:
                for col, count in missing_vals.items():
                    percentage = (count / stats['shape'][0]) * 100
                    report.append(f"- {col}: {count:,} ({percentage:.2f}%)")
            else:
                report.append("- No missing values found")
            
            # Add Data Cleaning Section
            report.append("\n## Data Cleaning Steps\n")
            if self.cleaning_steps:
                for i, step in enumerate(self.cleaning_steps, 1):
                    report.append(f"\n### Step {i}: {step['description']}\n")
                    if 'details' in step:
                        for key, value in step['details'].items():
                            report.append(f"- {key}: {value}")
            else:
                report.append("No cleaning steps were necessary for this dataset.")
            
            # Column Analysis
            report.append("\n## Column Analysis\n")
            
            # Analyze numeric columns
            numeric_cols = self.df.select_dtypes(include=['int64', 'float64']).columns
            if len(numeric_cols) > 0:
                report.append("\n### Numerical Columns\n")
                for col in numeric_cols:
                    print(f"Analyzing numerical column: {col}")
                    report.append(f"\n#### {col}\n")
                    analysis = self.analyze_column_stats(col, stats, is_numeric=True)
                    report.append(analysis)
                    time.sleep(1)  # Rate limiting
            
            # Analyze categorical columns
            cat_cols = self.df.select_dtypes(include=['object', 'category']).columns
            if len(cat_cols) > 0:
                report.append("\n### Categorical Columns\n")
                for col in cat_cols:
                    print(f"Analyzing categorical column: {col}")
                    report.append(f"\n#### {col}\n")
                    analysis = self.analyze_column_stats(col, stats, is_numeric=False)
                    report.append(analysis)
                    time.sleep(1)  # Rate limiting
            
            # Correlation Analysis
            print("Analyzing correlations...")
            report.append("\n## Variable Relationships\n")
            corr_analysis = self.analyze_correlations()
            report.append(corr_analysis)
            
            # Save report and cleaned data
            with open(report_file, 'w') as f:
                f.write('\n'.join(report))
            
            self.df.to_csv(cleaned_file, index=False)
            
            print(f"\nAnalysis complete!")
            print(f"Report saved as: {report_file}")
            print(f"Cleaned data saved as: {cleaned_file}")
            
            return report_file, cleaned_file
            
        except Exception as e:
            print(f"Error generating report: {str(e)}")
            return None, None

def run_eda(csv_file, api_key):
    """Main function to run the EDA process"""
    agent = EDAAgent(api_key)
    print("Loading data...")
    agent.load_data(csv_file)
    
    print("Generating EDA report...")
    report_file, cleaned_file = agent.generate_report()
    
    if report_file and cleaned_file:
        print("EDA completed successfully!")
        return report_file, cleaned_file
    else:
        print("EDA failed to complete.")
        return None, None

if __name__ == "__main__":
    # Load environment variables
    load_dotenv()
    
    # Get CSV path from user
    csv_path = input("Enter the path to your CSV file (e.g., data.csv): ")
    
    # Get API key from environment
    api_key = os.getenv('GOOGLE_API_KEY')
    
    if not api_key:
        print("Error: GOOGLE_API_KEY not found in .env file")
    elif not os.path.exists(csv_path):
        print(f"Error: File '{csv_path}' not found")
    else:
        print(f"Analyzing {csv_path}...")
        run_eda(csv_path, api_key)
