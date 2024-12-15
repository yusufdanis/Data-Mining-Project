import pandas as pd
import numpy as np

# Load dataset
df = pd.read_csv('../data/raw_dataset.csv')

# Save results to file
with open('../results/data_analysis_report.txt', 'w', encoding='utf-8') as f:
    # 1. GENERAL INFORMATION
    f.write("=== FOOTBALL PLAYER INJURY DATASET ANALYSIS ===\n\n")
    f.write("1. GENERAL INFORMATION\n")
    f.write(f"Number of Rows: {df.shape[0]}\n")
    f.write(f"Number of Columns: {df.shape[1]}\n")
    f.write(f"Dataset Size: {df.memory_usage().sum() / 1024:.2f} KB\n")
    f.write(f"Total Data Points: {df.shape[0] * df.shape[1]}\n")
    f.write(f"Memory Usage (Column-wise):\n{df.memory_usage(deep=True).to_string()}\n\n")
    
    # 2. COLUMN INFORMATION
    f.write("2. COLUMNS AND DATA TYPES\n")
    f.write(df.dtypes.to_string())
    f.write("\n\nColumn Names:\n")
    for col in df.columns:
        f.write(f"- {col}\n")
    f.write("\n")
    
    # 3. MISSING VALUE ANALYSIS
    f.write("3. MISSING VALUE ANALYSIS\n")
    missing_values = df.isnull().sum()
    missing_percentages = (missing_values / len(df)) * 100
    total_missing = df.isnull().sum().sum()
    total_cells = np.prod(df.shape)
    total_missing_percentage = (total_missing / total_cells) * 100
    
    missing_info = pd.DataFrame({
        'Missing Values': missing_values,
        'Missing Percentage': missing_percentages
    })
    f.write(missing_info[missing_info['Missing Values'] > 0].to_string())
    f.write(f"\n\nTotal Missing Values: {total_missing}\n")
    f.write(f"Total Missing Percentage: {total_missing_percentage:.2f}%\n\n")
    
    # 4. BASIC STATISTICS
    f.write("4. BASIC STATISTICS\n")
    f.write(df.describe(include='all').to_string())
    f.write("\n\n")
    
    # 5. CATEGORICAL VARIABLE INFORMATION
    f.write("5. CATEGORICAL VARIABLE INFORMATION\n")
    categorical_columns = df.select_dtypes(include=['object']).columns
    for col in categorical_columns:
        f.write(f"\n{col}:\n")
        f.write(f"Number of unique values: {df[col].nunique()}\n")
        f.write(f"First 5 unique values: {df[col].unique()[:5]}\n")
        f.write("Value distribution:\n")
        f.write(df[col].value_counts().head().to_string())
        f.write(f"\nMost frequent value: {df[col].mode()[0]}\n")
        f.write("\n")
    
    # 6. OUTLIER INFORMATION
    f.write("\n6. OUTLIER ANALYSIS\n")
    numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
    for col in numeric_columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)][col]
        if len(outliers) > 0:
            f.write(f"\n{col}:\n")
            f.write(f"Number of outliers: {len(outliers)}\n")
            f.write(f"Outlier percentage: {(len(outliers) / len(df)) * 100:.2f}%\n")
            f.write(f"Outlier range: [{outliers.min():.2f}, {outliers.max():.2f}]\n")
            f.write(f"Normal range: [{lower_bound:.2f}, {upper_bound:.2f}]\n")
    
    # 7. DATA CONSISTENCY CHECK
    f.write("\n7. DATA CONSISTENCY CHECK\n")
    if 'dob' in df.columns:
        f.write("\nDate of birth analysis:\n")
        f.write(f"Earliest date: {df['dob'].min()}\n")
        f.write(f"Latest date: {df['dob'].max()}\n")
    
    if 'height_cm' in df.columns and 'weight_kg' in df.columns:
        f.write("\nHeight and weight values:\n")
        df['height_cm'] = pd.to_numeric(df['height_cm'], errors='coerce')
        df['weight_kg'] = pd.to_numeric(df['weight_kg'], errors='coerce')
        
        f.write(f"Height range: [{df['height_cm'].min():.2f}, {df['height_cm'].max():.2f}] cm\n")
        f.write(f"Weight range: [{df['weight_kg'].min():.2f}, {df['weight_kg'].max():.2f}] kg\n")
        
        # BMI calculation and analysis
        df['bmi'] = df['weight_kg'] / ((df['height_cm']/100) ** 2)
        f.write(f"\nBMI Statistics:\n")
        f.write(f"Average BMI: {df['bmi'].mean():.2f}\n")
        f.write(f"BMI range: [{df['bmi'].min():.2f}, {df['bmi'].max():.2f}]\n")
    
    # 8. CORRELATION INFORMATION
    f.write("\n8. CORRELATION ANALYSIS\n")
    correlation_matrix = df[numeric_columns].corr()
    
    # All correlations
    f.write("\nAll Correlations:\n")
    f.write(correlation_matrix.to_string())
    
    # High correlations
    high_correlation = np.where(np.abs(correlation_matrix) > 0.7)
    high_correlation = [(correlation_matrix.index[x], correlation_matrix.columns[y], correlation_matrix.iloc[x, y]) 
                       for x, y in zip(*high_correlation) if x != y]
    if high_correlation:
        f.write("\n\nHighly correlated variables (>0.7):\n")
        for var1, var2, corr in high_correlation:
            f.write(f"{var1} - {var2}: {corr:.2f}\n")
    
    # 9. DATA QUALITY SCORE
    f.write("\n9. DATA QUALITY METRICS\n")
    completeness = 1 - (total_missing / total_cells)
    uniqueness = np.mean([df[col].nunique()/len(df) for col in df.columns])
    f.write(f"Data Completeness Score: {completeness:.2%}\n")
    f.write(f"Data Uniqueness Score: {uniqueness:.2%}\n")