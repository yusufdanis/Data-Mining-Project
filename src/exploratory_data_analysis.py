import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Support for special characters
plt.rcParams['font.family'] = 'DejaVu Sans'

# Load cleaned data
df = pd.read_csv('../data/cleaned_dataset.csv')

# Set figure size for visualizations
plt.figure(figsize=(12, 8))

# 1. Target Variable Analysis (Injury Duration)
plt.subplot(2, 2, 1)
sns.histplot(data=df, x='season_days_injured', bins=30)
plt.title('Distribution of Season Injury Duration')
plt.xlabel('Injury Duration (Days)')
plt.ylabel('Frequency')

# 2. Injury Duration by Position
plt.subplot(2, 2, 2)
sns.boxplot(data=df, x='position', y='season_days_injured')
plt.title('Injury Duration by Position')
plt.xlabel('Position')
plt.ylabel('Injury Duration (Days)')
plt.xticks(rotation=45)

# 3. Age and Injury Duration Relationship
plt.subplot(2, 2, 3)
sns.scatterplot(data=df, x='age', y='season_days_injured')
plt.title('Age vs Injury Duration')
plt.xlabel('Age')
plt.ylabel('Injury Duration (Days)')

# 4. FIFA Rating and Injury Duration Relationship
plt.subplot(2, 2, 4)
sns.scatterplot(data=df, x='fifa_rating', y='season_days_injured')
plt.title('FIFA Rating vs Injury Duration')
plt.xlabel('FIFA Rating')
plt.ylabel('Injury Duration (Days)')

plt.tight_layout()
plt.savefig('../results/figures/eda_plots_1.png')
plt.close()

# Save individual plots for first set
plt.figure(figsize=(10, 6))
sns.histplot(data=df, x='season_days_injured', bins=30)
plt.title('Distribution of Season Injury Duration')
plt.xlabel('Injury Duration (Days)')
plt.ylabel('Frequency')
plt.tight_layout()
plt.savefig('../results/figures/1_injury_duration_distribution.png')
plt.close()

plt.figure(figsize=(10, 6))
sns.boxplot(data=df, x='position', y='season_days_injured')
plt.title('Injury Duration by Position')
plt.xlabel('Position')
plt.ylabel('Injury Duration (Days)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('../results/figures/2_injury_duration_by_position.png')
plt.close()

plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='age', y='season_days_injured')
plt.title('Age vs Injury Duration')
plt.xlabel('Age')
plt.ylabel('Injury Duration (Days)')
plt.tight_layout()
plt.savefig('../results/figures/3_age_vs_injury.png')
plt.close()

plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='fifa_rating', y='season_days_injured')
plt.title('FIFA Rating vs Injury Duration')
plt.xlabel('FIFA Rating')
plt.ylabel('Injury Duration (Days)')
plt.tight_layout()
plt.savefig('../results/figures/4_fifa_rating_vs_injury.png')
plt.close()

# Second set of visualizations
plt.figure(figsize=(12, 8))

# 5. BMI Distribution
plt.subplot(2, 2, 1)
sns.histplot(data=df, x='bmi', bins=30)
plt.title('BMI Distribution')
plt.xlabel('BMI')
plt.ylabel('Frequency')

# 6. Pace and Physic by Position
plt.subplot(2, 2, 2)
df_melted = df.melt(id_vars=['position'], value_vars=['pace', 'physic'])
sns.boxplot(data=df_melted, x='position', y='value', hue='variable')
plt.title('Pace and Physic by Position')
plt.xlabel('Position')
plt.ylabel('Value')
plt.xticks(rotation=45)

# 7. Previous vs Current Season Injury Relationship
plt.subplot(2, 2, 3)
sns.scatterplot(data=df, x='season_days_injured_prev_season', y='season_days_injured')
plt.title('Previous vs Current Season Injury Duration')
plt.xlabel('Previous Season Injury (Days)')
plt.ylabel('Current Season Injury (Days)')

# 8. Minutes Played vs Injury Relationship
plt.subplot(2, 2, 4)
sns.scatterplot(data=df, x='season_minutes_played', y='season_days_injured')
plt.title('Minutes Played vs Injury Duration')
plt.xlabel('Minutes Played')
plt.ylabel('Injury Duration (Days)')

plt.tight_layout()
plt.savefig('../results/figures/eda_plots_2.png')
plt.close()

# Save individual plots for second set
plt.figure(figsize=(10, 6))
sns.histplot(data=df, x='bmi', bins=30)
plt.title('BMI Distribution')
plt.xlabel('BMI')
plt.ylabel('Frequency')
plt.tight_layout()
plt.savefig('../results/figures/5_bmi_distribution.png')
plt.close()

plt.figure(figsize=(10, 6))
df_melted = df.melt(id_vars=['position'], value_vars=['pace', 'physic'])
sns.boxplot(data=df_melted, x='position', y='value', hue='variable')
plt.title('Pace and Physic by Position')
plt.xlabel('Position')
plt.ylabel('Value')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('../results/figures/6_pace_physic_by_position.png')
plt.close()

plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='season_days_injured_prev_season', y='season_days_injured')
plt.title('Previous vs Current Season Injury Duration')
plt.xlabel('Previous Season Injury (Days)')
plt.ylabel('Current Season Injury (Days)')
plt.tight_layout()
plt.savefig('../results/figures/7_previous_vs_current_season_injury.png')
plt.close()

plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='season_minutes_played', y='season_days_injured')
plt.title('Minutes Played vs Injury Duration')
plt.xlabel('Minutes Played')
plt.ylabel('Injury Duration (Days)')
plt.tight_layout()
plt.savefig('../results/figures/8_minutes_played_vs_injury.png')
plt.close()

# Third set of visualizations
plt.figure(figsize=(12, 8))

# 9. Work Rate and Injury Relationship
plt.subplot(2, 2, 1)
sns.boxplot(data=df, x='work_rate', y='season_days_injured')
plt.title('Work Rate vs Injury Duration')
plt.xlabel('Work Rate')
plt.ylabel('Injury Duration (Days)')
plt.xticks(rotation=45)

# 10. Previous Season Significant Injury Impact
plt.subplot(2, 2, 2)
sns.boxplot(data=df, x='significant_injury_prev_season', y='season_days_injured')
plt.title('Impact of Previous Season Significant Injury')
plt.xlabel('Previous Season Significant Injury')
plt.ylabel('Injury Duration (Days)')

# 11. Injury Duration by Nationality (Top 10)
plt.subplot(2, 2, 3)
nationality_injury = df.groupby('nationality')['season_days_injured'].mean().sort_values(ascending=False).head(10)
sns.barplot(x=nationality_injury.index, y=nationality_injury.values)
plt.title('Average Injury Duration by Nationality (Top 10)')
plt.xlabel('Nationality')
plt.ylabel('Avg. Injury Duration (Days)')
plt.xticks(rotation=90)

# 12. Injury Trend by Season
plt.subplot(2, 2, 4)
season_injury = df.groupby('start_year')['season_days_injured'].mean()
sns.lineplot(x=season_injury.index, y=season_injury.values)
plt.title('Average Injury Duration by Season')
plt.xlabel('Season')
plt.ylabel('Avg. Injury Duration (Days)')

plt.tight_layout()
plt.savefig('../results/figures/eda_plots_3.png')
plt.close()

# Save individual plots for third set
plt.figure(figsize=(10, 6))
sns.boxplot(data=df, x='work_rate', y='season_days_injured')
plt.title('Work Rate vs Injury Duration')
plt.xlabel('Work Rate')
plt.ylabel('Injury Duration (Days)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('../results/figures/9_workrate_vs_injury.png')
plt.close()

plt.figure(figsize=(10, 6))
sns.boxplot(data=df, x='significant_injury_prev_season', y='season_days_injured')
plt.title('Impact of Previous Season Significant Injury')
plt.xlabel('Previous Season Significant Injury')
plt.ylabel('Injury Duration (Days)')
plt.tight_layout()
plt.savefig('../results/figures/10_impact_of_previous_season_injury.png')
plt.close()

plt.figure(figsize=(10, 6))
nationality_injury = df.groupby('nationality')['season_days_injured'].mean().sort_values(ascending=False).head(10)
sns.barplot(x=nationality_injury.index, y=nationality_injury.values)
plt.title('Average Injury Duration by Nationality (Top 10)')
plt.xlabel('Nationality')
plt.ylabel('Avg. Injury Duration (Days)')
plt.xticks(rotation=90)
plt.tight_layout()
plt.savefig('../results/figures/11_injury_duration_by_nationality.png')
plt.close()

plt.figure(figsize=(10, 6))
season_injury = df.groupby('start_year')['season_days_injured'].mean()
sns.lineplot(x=season_injury.index, y=season_injury.values)
plt.title('Average Injury Duration by Season')
plt.xlabel('Season')
plt.ylabel('Avg. Injury Duration (Days)')
plt.tight_layout()
plt.savefig('../results/figures/12_injury_trend_by_season.png')
plt.close()

# Expand the report
with open('../results/eda_report.txt', 'w', encoding='utf-8') as f:
    f.write("=== DETAILED DATA ANALYSIS REPORT ===\n")
    
    # 1. Injury Duration Distribution
    injury_stats = df['season_days_injured'].describe()
    f.write("\n1. INJURY DURATION DISTRIBUTION ANALYSIS")
    f.write(f"\n- Mean injury duration: {injury_stats['mean']:.2f} days")
    f.write(f"\n- Median injury duration: {injury_stats['50%']:.2f} days")
    f.write(f"\n- Standard deviation: {injury_stats['std']:.2f} days")
    f.write("\n- Distribution is right-skewed (positive skewness)")
    f.write("\n- Most injuries last between 0-100 days")
    f.write(f"\n- Maximum injury duration: {injury_stats['max']:.0f} days")

    # 2. Position Analysis
    f.write("\n\n2. ANALYSIS BY POSITION")
    for pos in df['position'].unique():
        pos_data = df[df['position'] == pos]['season_days_injured']
        outliers = pos_data[pos_data > pos_data.quantile(0.75) + 1.5 * (pos_data.quantile(0.75) - pos_data.quantile(0.25))]
        
        f.write(f"\n\n{pos} Position:")
        f.write(f"\n- Average injury duration: {pos_data.mean():.2f} days")
        f.write(f"\n- Median injury duration: {pos_data.median():.2f} days")
        f.write(f"\n- Number of outliers: {len(outliers)}")
        if len(outliers) > 0:
            f.write(f"\n- Highest outlier value: {outliers.max():.0f} days")

    # 3. Age-Injury Relationship
    f.write("\n\n3. AGE-INJURY RELATIONSHIP")
    age_bins = [0, 20, 25, 30, 35, 100]
    age_labels = ['<20', '20-25', '25-30', '30-35', '>35']
    df['age_group'] = pd.cut(df['age'], bins=age_bins, labels=age_labels)
    age_analysis = df.groupby('age_group', observed=True)['season_days_injured'].agg(['mean', 'median', 'count'])
    
    for age_group in age_labels:
        stats = age_analysis.loc[age_group]
        f.write(f"\n\n{age_group} age group:")
        f.write(f"\n- Average injury duration: {stats['mean']:.2f} days")
        f.write(f"\n- Median injury duration: {stats['median']:.2f} days")
        f.write(f"\n- Number of players: {stats['count']}")
    
    # 4. FIFA Rating Analysis
    f.write("\n\n4. FIFA RATING ANALYSIS")
    rating_bins = [0, 65, 70, 75, 80, 85, 90, 100]
    rating_labels = ['<65', '65-70', '70-75', '75-80', '80-85', '85-90', '>90']
    df['rating_group'] = pd.cut(df['fifa_rating'], bins=rating_bins, labels=rating_labels)
    rating_analysis = df.groupby('rating_group', observed=True)['season_days_injured'].agg(['mean', 'median', 'count'])
    
    for rating_group in rating_labels:
        if rating_group in rating_analysis.index:
            stats = rating_analysis.loc[rating_group]
            f.write(f"\n\n{rating_group} rating group:")
            f.write(f"\n- Average injury duration: {stats['mean']:.2f} days")
            f.write(f"\n- Median injury duration: {stats['median']:.2f} days")
            f.write(f"\n- Number of players: {stats['count']}")

    # 5. BMI Distribution Analysis
    bmi_stats = df['bmi'].describe()
    f.write("\n\n5. BMI DISTRIBUTION ANALYSIS")
    f.write(f"\n- BMI range: {bmi_stats['min']:.1f} - {bmi_stats['max']:.1f}")
    f.write(f"\n- Average BMI: {bmi_stats['mean']:.1f}")
    f.write(f"\n- Median BMI: {bmi_stats['50%']:.1f}")
    f.write(f"\n- Standard deviation: {bmi_stats['std']:.1f}")

    # 6. Pace and Physic Analysis
    f.write("\n\n6. PACE AND PHYSIC ANALYSIS")
    for pos in df['position'].unique():
        pos_data = df[df['position'] == pos]
        f.write(f"\n\n{pos} Position:")
        f.write(f"\n- Average Pace: {pos_data['pace'].mean():.1f}")
        f.write(f"\n- Average Physic: {pos_data['physic'].mean():.1f}")
        f.write(f"\n- Pace range: {pos_data['pace'].min():.0f}-{pos_data['pace'].max():.0f}")
        f.write(f"\n- Physic range: {pos_data['physic'].min():.0f}-{pos_data['physic'].max():.0f}")

    # 7. Previous Season Injury Analysis
    f.write("\n\n7. PREVIOUS SEASON INJURY ANALYSIS")
    prev_injury_stats = df.groupby('significant_injury_prev_season')['season_days_injured'].describe()
    for injury_status in [0, 1]:
        stats = prev_injury_stats.loc[injury_status]
        status_text = "Players with Significant Injury" if injury_status == 1 else "Players without Significant Injury"
        f.write(f"\n\n{status_text}:")
        f.write(f"\n- Number of players: {stats['count']:.0f}")
        f.write(f"\n- Average injury duration: {stats['mean']:.1f} days")
        f.write(f"\n- Median injury duration: {stats['50%']:.1f} days")
        f.write(f"\n- Maximum injury duration: {stats['max']:.0f} days")

    # 8. Season Minutes-Injury Analysis
    f.write("\n\n8. SEASON MINUTES-INJURY ANALYSIS")
    minutes_stats = df.groupby('season_days_injured')['season_minutes_played'].describe()
    f.write(f"\n- Average minutes played: {df['season_minutes_played'].mean():.1f}")
    f.write(f"\n- Median minutes played: {df['season_minutes_played'].median():.1f}")
    f.write(f"\n- Maximum minutes played: {df['season_minutes_played'].max():.0f}")
    f.write(f"\n- Minimum minutes played: {df['season_minutes_played'].min():.0f}")