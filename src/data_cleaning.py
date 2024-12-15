import pandas as pd
import numpy as np

# Load dataset
df = pd.read_csv('../data/raw_dataset.csv')

# Rename p_id2 to p_id
df = df.rename(columns={'p_id2': 'p_id'})

# Pre-cleaning status report
with open('../results/data_cleaning_report.txt', 'w', encoding='utf-8') as f:
    f.write("=== DATA CLEANING REPORT ===\n\n")
    f.write("INITIAL STATE:\n")
    f.write(f"Number of Rows: {df.shape[0]}\n")
    f.write(f"Number of Columns: {df.shape[1]}\n")
    f.write("\nMissing Values:\n")
    f.write(df.isnull().sum().to_string())
    
    # Define important columns
    important_columns = [
        'cumulative_minutes_played',
        'cumulative_games_played',
        'avg_days_injured_prev_seasons',
        'avg_games_per_season_prev_seasons',
        'significant_injury_prev_season',
        'cumulative_days_injured',
        'season_days_injured_prev_season',
        'minutes_per_game_prev_seasons'
    ]
    
    # Check missing values in important columns
    f.write("\n\nMISSING VALUES IN IMPORTANT COLUMNS:\n")
    for col in important_columns:
        missing_count = df[col].isnull().sum()
        missing_percentage = (missing_count / len(df)) * 100
        f.write(f"{col}: {missing_count} (%{missing_percentage:.2f})\n")
    
    # Remove rows with missing values in these columns
    initial_rows = len(df)
    df = df.dropna(subset=important_columns)
    removed_rows = initial_rows - len(df)
    
    f.write(f"\nREMOVED ROWS:\n")
    f.write(f"Total rows removed: {removed_rows}\n")
    f.write(f"Percentage of rows removed: %{(removed_rows/initial_rows)*100:.2f}\n")
    
    f.write("\nSTATE AFTER CLEANING:\n")
    f.write(f"Remaining rows: {len(df)}\n")
    f.write(f"Number of columns: {df.shape[1]}\n")
    f.write("\nRemaining Missing Values:\n")
    f.write(df.isnull().sum().to_string())
    
    # P_ID and DOB analysis
    f.write("\n\n=== P_ID AND DOB CONSISTENCY ANALYSIS ===\n")
    
    # Check number of unique DOBs for each p_id
    id_dob_counts = df.groupby('p_id')['dob'].nunique()
    inconsistent_ids = id_dob_counts[id_dob_counts > 1]
    
    if len(inconsistent_ids) > 0:
        f.write(f"\nNumber of p_ids with different DOBs: {len(inconsistent_ids)}")
        f.write("\n\nDetails of inconsistent records:\n")
        
        for pid in inconsistent_ids.index:
            inconsistent_records = df[df['p_id'] == pid]
            f.write(f"\nP_ID: {pid}")
            f.write("\n" + inconsistent_records[['p_id', 'dob', 'nationality', 'position', 'age', 'fifa_rating']].to_string())
            
            # Check unique DOBs for this p_id
            unique_dobs = inconsistent_records['dob'].unique()
            
            # Create new p_id for each unique DOB
            for dob in unique_dobs:
                birth_year = dob[:4]
                mask = (df['p_id'] == pid) & (df['dob'] == dob)
                new_id = f"{pid}_{birth_year}"
                df.loc[mask, 'p_id'] = new_id
                f.write(f"\nNew P_ID created: {new_id}")
    
    # Final state check
    f.write("\n\n=== STATE AFTER P_ID RENAMING ===\n")
    
    # DOB check
    last_check = df.groupby('p_id')['dob'].nunique()
    remaining_inconsistent = last_check[last_check > 1]
    
    if len(remaining_inconsistent) > 0:
        f.write(f"\nNumber of p_ids still with DOB inconsistency: {len(remaining_inconsistent)}")
        f.write("\nRemaining inconsistent records:\n")
        for pid in remaining_inconsistent.index:
            inconsistent_records = df[df['p_id'] == pid]
            f.write(f"\nP_ID: {pid}\n")
            f.write(inconsistent_records[['p_id', 'dob', 'nationality', 'position', 'age']].to_string())
    else:
        f.write("\nAll P_ID and DOB inconsistencies resolved.")
    
    # FIFA Rating check
    rating_check = df.groupby('p_id')['fifa_rating'].nunique()
    rating_inconsistent = rating_check[rating_check > 1]
    
    if len(rating_inconsistent) > 0:
        f.write(f"\n\nNumber of p_ids with FIFA Rating inconsistency: {len(rating_inconsistent)}")
        f.write("\nDetails of inconsistent records:\n")
        for pid in rating_inconsistent.index:
            inconsistent_records = df[df['p_id'] == pid]
            f.write(f"\nP_ID: {pid}")
            f.write("\n" + inconsistent_records[['p_id', 'dob', 'fifa_rating', 'start_year']].to_string())
    else:
        f.write("\nNo FIFA Rating inconsistencies found.")
    
    # Position check
    position_check = df.groupby('p_id')['position'].nunique()
    position_inconsistent = position_check[position_check > 1]
    
    if len(position_inconsistent) > 0:
        f.write(f"\n\nNumber of p_ids with Position inconsistency: {len(position_inconsistent)}")
        f.write("\nDetails of inconsistent records:\n")
        for pid in position_inconsistent.index:
            inconsistent_records = df[df['p_id'] == pid]
            f.write(f"\nP_ID: {pid}")
            f.write("\n" + inconsistent_records[['p_id', 'dob', 'position', 'start_year']].to_string())
    else:
        f.write("\nNo Position inconsistencies found.")
    
    # Keep specific p_id-position mappings
    correct_mappings = {
        'adamsmith_1991': 'Defender',
        'adamsmith_1992': 'Goalkeeper',
        'ashleywestwood': 'Midfielder',
        'dannyrose_1988': 'Midfielder',
        'dannyrose_1990': 'Defender',
        'dannyrose_1993': 'Midfielder',
        'dannyward_1990': 'Forward',
        'dannyward_1993': 'Goalkeeper',
        'jamescollins_1983': 'Defender',
        'jamescollins_1990': 'Forward',
        'tommysmith_1992': 'Defender'
    }

    # For p_ids with position inconsistency
    for pid in position_inconsistent.index:
        if pid in correct_mappings:
            # Remove records with incorrect position
            mask = (df['p_id'] == pid) & (df['position'] != correct_mappings[pid])
            df.drop(df[mask].index, inplace=True)
            
    f.write("\n\nRecords with incorrect position mappings removed.")
    
    # Position check after removal
    f.write("\n\n=== POSITION CHECK AFTER RECORD REMOVAL ===\n")
    position_check_after = df.groupby('p_id')['position'].nunique()
    position_inconsistent_after = position_check_after[position_check_after > 1]
    
    if len(position_inconsistent_after) > 0:
        f.write(f"\nNumber of p_ids still with position inconsistency: {len(position_inconsistent_after)}")
        f.write("\nDetails of remaining inconsistent records:\n")
        for pid in position_inconsistent_after.index:
            inconsistent_records = df[df['p_id'] == pid]
            f.write(f"\nP_ID: {pid}")
            f.write("\n" + inconsistent_records[['p_id', 'dob', 'position', 'start_year']].to_string())
    else:
        f.write("\nAll position inconsistencies resolved.")
    
    # PACE AND PHYSIC ANALYSIS
    f.write("\n\n=== PACE AND PHYSIC MISSING VALUE ANALYSIS ===\n")
    missing_both = df[df['pace'].isnull() & df['physic'].isnull()]
    f.write(f"\nNumber of rows missing both values: {len(missing_both)}")
    
    f.write("\n\nDetails of rows with missing values:\n")
    columns_to_show = ['p_id', 'position', 'age', 'fifa_rating', 'pace', 'physic']
    f.write(missing_both[columns_to_show].to_string())
    
    # Fill pace and physic values from same p_id
    f.write("\n\n=== FILLING PACE AND PHYSIC FROM SAME P_ID ===\n")
    filled_count = 0
    
    for idx, row in missing_both.iterrows():
        # Find other records with same p_id and sort by start_year
        same_id_records = df[(df['p_id'] == row['p_id']) & 
                           (df['pace'].notna()) & 
                           (df['physic'].notna())].sort_values('start_year', ascending=False)
        
        if len(same_id_records) > 0:
            # Get values from record with highest start_year
            pace_value = same_id_records.iloc[0]['pace']
            physic_value = same_id_records.iloc[0]['physic']
            
            # Fill values
            df.loc[idx, 'pace'] = pace_value
            df.loc[idx, 'physic'] = physic_value
            filled_count += 1
            
            f.write(f"\nP_ID: {row['p_id']}")
            f.write(f"\nFilled values - Pace: {pace_value}, Physic: {physic_value}")
    
    f.write(f"\n\nTotal {filled_count} records filled.")
    
    # Position correction for specific player
    df.loc[df['p_id'] == 'costelpantilimon', 'position'] = 'Goalkeeper'
    
    # Update position numeric values
    position_mapping = {
        'Goalkeeper': 1,
        'Defender': 2,
        'Midfielder': 3,
        'Forward': 4
    }
    df['position_numeric'] = df['position'].map(position_mapping)
    
    # Fill pace and physic values from similar players
    f.write("\n\n=== FILLING PACE AND PHYSIC FROM SIMILAR PLAYERS ===\n")
    similar_filled_count = 0
    
    # Find records still missing values
    still_missing = df[df['pace'].isnull() & df['physic'].isnull()]
    
    for idx, row in still_missing.iterrows():
        if pd.isna(row['position']):
            continue
            
        # Find players with same position and values
        same_position = df[
            (df['position'] == row['position']) & 
            (df['pace'].notna()) & 
            (df['physic'].notna())
        ].copy()
        
        if len(same_position) > 0:
            # Calculate similarity scores
            same_position.loc[:, 'age_diff'] = abs(same_position['age'] - row['age'])
            same_position.loc[:, 'bmi_diff'] = abs(same_position['bmi'] - row['bmi'])
            same_position.loc[:, 'rating_diff'] = abs(same_position['fifa_rating'] - row['fifa_rating'])
            
            # Normalize differences
            same_position.loc[:, 'age_diff_norm'] = same_position['age_diff'] / same_position['age_diff'].max()
            same_position.loc[:, 'bmi_diff_norm'] = same_position['bmi_diff'] / same_position['bmi_diff'].max()
            same_position.loc[:, 'rating_diff_norm'] = same_position['rating_diff'] / same_position['rating_diff'].max()
            
            # Weighted similarity score (lower = more similar)
            same_position.loc[:, 'similarity_score'] = (
                0.4 * same_position['age_diff_norm'] +    # More weight to age
                0.2 * same_position['bmi_diff_norm'] +    # Less weight to BMI
                0.4 * same_position['rating_diff_norm']   # More weight to rating
            )
            
            # Find most similar player
            most_similar = same_position.nsmallest(1, 'similarity_score').iloc[0]
            
            # Fill values
            df.loc[idx, 'pace'] = most_similar['pace']
            df.loc[idx, 'physic'] = most_similar['physic']
            similar_filled_count += 1
            
            f.write(f"\nP_ID: {row['p_id']}")
            f.write(f"\nSimilar player P_ID: {most_similar['p_id']}")
            f.write(f"\nSimilarity metrics:")
            f.write(f"\n  Age difference: {most_similar['age_diff']:.2f}")
            f.write(f"\n  BMI difference: {most_similar['bmi_diff']:.2f}")
            f.write(f"\n  Rating difference: {most_similar['rating_diff']:.2f}")
            f.write(f"\nFilled values - Pace: {most_similar['pace']}, Physic: {most_similar['physic']}")
    
    f.write(f"\n\nTotal {similar_filled_count} records filled from similar players.")
    
    # Analysis after filling
    f.write("\n\n=== PACE AND PHYSIC ANALYSIS AFTER FILLING ===\n")
    missing_both_after = df[df['pace'].isnull() & df['physic'].isnull()]
    f.write(f"\nNumber of rows still missing both values: {len(missing_both_after)}")
    
    if len(missing_both_after) > 0:
        f.write("\n\nDetails of remaining rows with missing values:\n")
        f.write(missing_both_after[columns_to_show].to_string())
    
    # Final dataset state analysis
    f.write("\n\n=== FINAL DATASET STATE ANALYSIS ===\n")
    
    # General information
    f.write("\nGeneral Information:")
    f.write(f"\nNumber of Rows: {df.shape[0]}")
    f.write(f"\nNumber of Columns: {df.shape[1]}")
    
    # Missing values
    f.write("\n\nMissing Values:")
    f.write("\n" + df.isnull().sum().to_string())
    
    # Data Cleaning Analysis
    f.write("\n\n=== DATA CLEANING ANALYSIS ===\n")
    
    # 1. Outlier Analysis
    f.write("\n1. OUTLIER ANALYSIS\n")
    numeric_cols = ['age', 'fifa_rating', 'bmi', 'pace', 'physic']
    
    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
        
        f.write(f"\n{col.upper()} Outlier Analysis:")
        f.write(f"\nLower bound: {lower_bound:.2f}")
        f.write(f"\nUpper bound: {upper_bound:.2f}")
        f.write(f"\nNumber of outliers: {len(outliers)}")
        if len(outliers) > 0:
            f.write("\nOutliers:")
            f.write("\n" + outliers[['p_id', col, 'position']].to_string())
    
    # 2. Consistency Analysis
    f.write("\n\n2. CONSISTENCY ANALYSIS\n")
    
    # Age-Rating Consistency
    f.write("\nAge-Rating Consistency:")
    young_high_rated = df[(df['age'] < 20) & (df['fifa_rating'] > 85)]
    old_high_rated = df[(df['age'] > 35) & (df['fifa_rating'] > 85)]
    
    if len(young_high_rated) > 0:
        f.write("\nYoung players (under 20) with high rating:")
        f.write("\n" + young_high_rated[['p_id', 'age', 'fifa_rating']].to_string())
    
    if len(old_high_rated) > 0:
        f.write("\nOld players (over 35) with high rating:")
        f.write("\n" + old_high_rated[['p_id', 'age', 'fifa_rating']].to_string())
    
    # Position-Pace/Physic Consistency
    f.write("\n\nPosition-Pace/Physic Consistency:")
    slow_forwards = df[(df['position'] == 'Forward') & (df['pace'] < 60)]
    weak_defenders = df[(df['position'] == 'Defender') & (df['physic'] < 60)]
    
    if len(slow_forwards) > 0:
        f.write("\nForwards with low pace:")
        f.write("\n" + slow_forwards[['p_id', 'position', 'pace']].to_string())
    
    if len(weak_defenders) > 0:
        f.write("\nDefenders with low physic:")
        f.write("\n" + weak_defenders[['p_id', 'position', 'physic']].to_string())
    
    # BMI Consistency
    f.write("\n\nBMI Consistency:")
    abnormal_bmi = df[(df['bmi'] < 18.5) | (df['bmi'] > 30)]
    if len(abnormal_bmi) > 0:
        f.write("\nAbnormal BMI values:")
        f.write("\n" + abnormal_bmi[['p_id', 'bmi']].to_string())
    
    # 3. Data Type Checks
    f.write("\n\n3. DATA TYPE CHECKS\n")
    f.write("\nColumn Data Types:")
    f.write("\n" + df.dtypes.to_string())
    
    # 4. Correlation Analysis
    f.write("\n\n4. CORRELATION ANALYSIS\n")
    
    # Select all numeric columns
    numeric_cols_corr = [
        'age', 'fifa_rating', 'bmi', 'pace', 'physic',
        'season_days_injured', 'total_days_injured',
        'season_minutes_played', 'season_games_played',
        'season_matches_in_squad', 'total_minutes_played',
        'total_games_played', 'cumulative_minutes_played',
        'cumulative_games_played', 'minutes_per_game_prev_seasons',
        'avg_days_injured_prev_seasons', 'avg_games_per_season_prev_seasons',
        'position_numeric', 'work_rate_numeric',
        'cumulative_days_injured', 'season_days_injured_prev_season'
    ]
    
    corr_matrix = df[numeric_cols_corr].corr()
    high_corr = []
    
    for i in range(len(numeric_cols_corr)):
        for j in range(i+1, len(numeric_cols_corr)):
            if abs(corr_matrix.iloc[i,j]) > 0.9:
                high_corr.append(f"{numeric_cols_corr[i]} - {numeric_cols_corr[j]}: {corr_matrix.iloc[i,j]:.2f}")
    
    if high_corr:
        f.write("\nHighly Correlated Variables (>0.9):")
        f.write("\n" + "\n".join(high_corr))
    
    # Remove highly correlated variables
    columns_to_drop = [
        'season_games_played',
        'total_games_played',
        'cumulative_games_played'
    ]
    
    df = df.drop(columns=columns_to_drop)
    f.write("\n\nRemoved Highly Correlated Variables:")
    f.write("\n" + ", ".join(columns_to_drop))
    
    # Data Quality Score
    f.write("\n\n=== DATA QUALITY SCORE ===\n")
    
    # 1. Missing Data Score
    missing_score = (1 - df.isnull().sum().sum() / (df.shape[0] * df.shape[1])) * 100
    f.write(f"\n1. Missing Data Score: %{missing_score:.2f}")
    
    # 2. Consistency Score
    consistency_issues = 0
    # Age-Rating inconsistency
    consistency_issues += len(df[(df['age'] < 20) & (df['fifa_rating'] > 85)])
    consistency_issues += len(df[(df['age'] > 35) & (df['fifa_rating'] > 85)])
    # Position-Pace/Physic inconsistency
    consistency_issues += len(df[(df['position'] == 'Forward') & (df['pace'] < 60)])
    consistency_issues += len(df[(df['position'] == 'Defender') & (df['physic'] < 60)])
    # BMI inconsistency
    consistency_issues += len(df[(df['bmi'] < 18.5) | (df['bmi'] > 30)])
    
    consistency_score = (1 - consistency_issues / len(df)) * 100
    f.write(f"\n2. Consistency Score: %{consistency_score:.2f}")
    
    # 3. Outlier Score
    outlier_count = 0
    numeric_cols = ['age', 'fifa_rating', 'bmi', 'pace', 'physic']
    
    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        outlier_count += len(df[(df[col] < Q1 - 1.5 * IQR) | (df[col] > Q3 + 1.5 * IQR)])
    
    outlier_score = (1 - outlier_count / (len(df) * len(numeric_cols))) * 100
    f.write(f"\n3. Outlier Score: %{outlier_score:.2f}")
    
    # 4. Data Type Consistency Score
    # Updated numeric columns list (removed dropped columns)
    numeric_cols_corr = [
        'age', 'fifa_rating', 'bmi', 'pace', 'physic',
        'season_days_injured', 'total_days_injured',
        'season_minutes_played', 'season_matches_in_squad', 
        'total_minutes_played', 'cumulative_minutes_played',
        'minutes_per_game_prev_seasons',
        'avg_days_injured_prev_seasons', 'avg_games_per_season_prev_seasons',
        'position_numeric', 'work_rate_numeric',
        'cumulative_days_injured', 'season_days_injured_prev_season'
    ]
    
    # Check if all numeric columns are actually numeric
    type_issues = 0
    for col in numeric_cols_corr:
        if not np.issubdtype(df[col].dtype, np.number):
            type_issues += 1
    
    type_score = (1 - type_issues / len(numeric_cols_corr)) * 100
    f.write(f"\n4. Data Type Consistency Score: %{type_score:.2f}")
    
    # Overall Quality Score (weighted average)
    quality_score = (
        0.35 * missing_score +      # More weight to missing data
        0.30 * consistency_score +   # Important for consistency
        0.20 * outlier_score +      # Medium weight for outliers
        0.15 * type_score          # Less weight for data types
    )
    
    f.write(f"\n\nOVERALL DATA QUALITY SCORE: %{quality_score:.2f}")

# Save cleaned dataset
df.to_csv('../data/cleaned_dataset.csv', index=False)
print("Cleaned dataset saved as 'cleaned_dataset.csv'.")