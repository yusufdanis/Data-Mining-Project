import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import RobustScaler
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import SMOTE

# Load data
df = pd.read_csv('../data/cleaned_dataset.csv')

# Categorize injury durations
def categorize_injury(days):
    if days <= 7:  # 0-7 days
        return 1   # Minimal injury
    elif days <= 28:
        return 2   # Mild/Moderate injury
    elif days <= 84:
        return 3   # Severe injury
    else:
        return 4   # Long-term injury

# Recategorize target variable
df['injury_category'] = df['season_days_injured'].apply(categorize_injury)

# Feature selection - All meaningful features
features = [
    # Demographic and Physical Features
    'age',
    'bmi',
    
    # FIFA Features
    'fifa_rating',
    'pace',
    'physic',
    'work_rate_numeric',
    
    # Game Statistics
    'season_minutes_played',
    'season_matches_in_squad',
    'minutes_per_game_prev_seasons',
    'avg_games_per_season_prev_seasons',
    
    # Injury History
    'avg_days_injured_prev_seasons',
    'cumulative_days_injured'
]

X = df[features]
y = df['injury_category']

# First scaling, then split
scaler = RobustScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Apply SMOTE only on training data
smote = SMOTE(sampling_strategy='auto', random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# Create models
dt_model = DecisionTreeClassifier(
    max_depth=10,
    min_samples_split=10,
    min_samples_leaf=5,
    random_state=42
)
rf_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=15,
    min_samples_split=10,
    min_samples_leaf=5,
    random_state=42
)

# Train models
dt_model.fit(X_train_resampled, y_train_resampled)
rf_model.fit(X_train_resampled, y_train_resampled)

def evaluate_model(model, X_test, y_test, model_name):
    # Predictions
    y_pred = model.predict(X_test)
    
    # Metrics
    metrics = {
        'Model': model_name,
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred, average='weighted'),
        'Recall': recall_score(y_test, y_pred, average='weighted'),
        'F1-score': f1_score(y_test, y_pred, average='weighted')
    }
    
    return metrics, y_pred

# Feature importance analysis
def plot_feature_importance(model, features, model_name):
    importance = model.feature_importances_
    feat_importance = pd.DataFrame({
        'Feature': features,
        'Importance': importance
    }).sort_values('Importance', ascending=False)
    
    plt.figure(figsize=(12, 8))
    sns.barplot(x='Importance', y='Feature', data=feat_importance)
    plt.title(f'Feature Importance - {model_name}')
    plt.tight_layout()
    plt.savefig(f'../results/figures/feature_importance_{model_name.lower().replace(" ", "_")}.png')
    plt.close()
    
    return feat_importance

# Evaluate results
dt_metrics, dt_pred = evaluate_model(dt_model, X_test, y_test, 'Decision Tree')
rf_metrics, rf_pred = evaluate_model(rf_model, X_test, y_test, 'Random Forest')

# Feature importance analysis
dt_importance = plot_feature_importance(dt_model, features, 'Decision Tree')
rf_importance = plot_feature_importance(rf_model, features, 'Random Forest')

# Save results
with open('../results/model_results.txt', 'w', encoding='utf-8') as f:
    f.write("=== MODEL EVALUATION REPORT ===\n\n")
    
    # Injury category distribution
    f.write("1. INJURY CATEGORY DISTRIBUTION\n")
    category_dist = df['injury_category'].value_counts().sort_index()
    f.write(category_dist.to_string())
    f.write("\n\nCategory Descriptions:")
    f.write("\n1: Minimal injury (1-7 days)")
    f.write("\n2: Mild/Moderate injury (8-28 days)")
    f.write("\n3: Severe injury (29-84 days)")
    f.write("\n4: Long-term injury (>84 days)")
    
    # Model metrics
    f.write("\n\n2. MODEL METRICS\n")
    results_df = pd.DataFrame([dt_metrics, rf_metrics])
    f.write(results_df.to_string())
    
    # Detailed classification reports
    f.write("\n\n3. DETAILED CLASSIFICATION REPORT\n")
    f.write("\nDecision Tree:\n")
    f.write(classification_report(y_test, dt_pred))
    f.write("\nRandom Forest:\n")
    f.write(classification_report(y_test, rf_pred))
    
    # Feature importance
    f.write("\n\n4. FEATURE IMPORTANCE ANALYSIS\n")
    f.write("\nDecision Tree - All Features:\n")
    f.write(dt_importance.to_string())
    f.write("\n\nRandom Forest - All Features:\n")
    f.write(rf_importance.to_string())

# Confusion Matrix visualizations
def plot_confusion_matrix(y_true, y_pred, model_name):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix - {model_name}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(f'../results/figures/confusion_matrix_{model_name.lower().replace(" ", "_")}.png')
    plt.close()

plot_confusion_matrix(y_test, dt_pred, 'Decision Tree')
plot_confusion_matrix(y_test, rf_pred, 'Random Forest')