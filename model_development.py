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

# Veriyi yükle
df = pd.read_csv('cleaned_dataset.csv')

# Yaralanma sürelerini kategorilere ayır
def categorize_injury(days):
    if days <= 7:  # 0-7 gün arası
        return 1   # Minimal injury
    elif days <= 28:
        return 2   # Mild/Moderate injury
    elif days <= 84:
        return 3   # Severe injury
    else:
        return 4   # Long-term injury

# Hedef değişkeni yeniden kategorize et
df['injury_category'] = df['season_days_injured'].apply(categorize_injury)

# Özellik seçimi - Tüm anlamlı feature'lar
features = [
    # Demografik ve Fiziksel Özellikler
    'age',
    'bmi',
    
    # FIFA Özellikleri
    'fifa_rating',
    'pace',
    'physic',
    'work_rate_numeric',
    
    # Oyun İstatistikleri
    'season_minutes_played',
    'season_matches_in_squad',
    'minutes_per_game_prev_seasons',
    'avg_games_per_season_prev_seasons',
    
    # Yaralanma Geçmişi
    'avg_days_injured_prev_seasons',
    'cumulative_days_injured'
]

X = df[features]
y = df['injury_category']

# Önce scaling, sonra split
scaler = RobustScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# SMOTE'u sadece train verisi üzerinde uygula
smote = SMOTE(sampling_strategy='auto', random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# Modelleri oluştur
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

# Modelleri eğit
dt_model.fit(X_train_resampled, y_train_resampled)
rf_model.fit(X_train_resampled, y_train_resampled)

def evaluate_model(model, X_test, y_test, model_name):
    # Tahminler
    y_pred = model.predict(X_test)
    
    # Metrikler
    metrics = {
        'Model': model_name,
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred, average='weighted'),
        'Recall': recall_score(y_test, y_pred, average='weighted'),
        'F1-score': f1_score(y_test, y_pred, average='weighted')
    }
    
    return metrics, y_pred

# Feature importance analizi
def plot_feature_importance(model, features, model_name):
    importance = model.feature_importances_
    feat_importance = pd.DataFrame({
        'Feature': features,
        'Importance': importance
    }).sort_values('Importance', ascending=False)
    
    plt.figure(figsize=(12, 8))  # Grafik boyutunu büyüttüm
    sns.barplot(x='Importance', y='Feature', data=feat_importance)
    plt.title(f'Feature Importance - {model_name}')
    plt.tight_layout()
    plt.savefig(f'feature_importance_{model_name.lower().replace(" ", "_")}.png')
    plt.close()
    
    return feat_importance

# Sonuçları değerlendir
dt_metrics, dt_pred = evaluate_model(dt_model, X_test, y_test, 'Decision Tree')
rf_metrics, rf_pred = evaluate_model(rf_model, X_test, y_test, 'Random Forest')

# Feature importance analizi
dt_importance = plot_feature_importance(dt_model, features, 'Decision Tree')
rf_importance = plot_feature_importance(rf_model, features, 'Random Forest')

# Sonuçları kaydet
with open('model_results.txt', 'w', encoding='utf-8') as f:
    f.write("=== MODEL DEĞERLENDİRME RAPORU ===\n\n")
    
    # Yaralanma kategorileri dağılımı
    f.write("1. YARALANMA KATEGORİLERİ DAĞILIMI\n")
    category_dist = df['injury_category'].value_counts().sort_index()
    f.write(category_dist.to_string())
    f.write("\n\nKategori Açıklamaları:")
    f.write("\n0: No injury (0 gün)")
    f.write("\n1: Minimal injury (1-7 gün)")
    f.write("\n2: Mild/Moderate injury (8-28 gün)")
    f.write("\n3: Severe injury (29-84 gün)")
    f.write("\n4: Long-term injury (>84 gün)")
    
    # Model metrikleri
    f.write("\n\n2. MODEL METRİKLERİ\n")
    results_df = pd.DataFrame([dt_metrics, rf_metrics])
    f.write(results_df.to_string())
    
    # Detaylı sınıflandırma raporları
    f.write("\n\n3. DETAYLI SINIFLANDIRMA RAPORU\n")
    f.write("\nDecision Tree:\n")
    f.write(classification_report(y_test, dt_pred))
    f.write("\nRandom Forest:\n")
    f.write(classification_report(y_test, rf_pred))
    
    # Feature importance
    f.write("\n\n4. ÖZELLİK ÖNEMİ ANALİZİ\n")
    f.write("\nDecision Tree - Tüm Özellikler:\n")
    f.write(dt_importance.to_string())
    f.write("\n\nRandom Forest - Tüm Özellikler:\n")
    f.write(rf_importance.to_string())

# Confusion Matrix görselleştirmeleri
def plot_confusion_matrix(y_true, y_pred, model_name):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))  # Grafik boyutunu büyüttüm
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix - {model_name}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(f'confusion_matrix_{model_name.lower().replace(" ", "_")}.png')
    plt.close()

plot_confusion_matrix(y_test, dt_pred, 'Decision Tree')
plot_confusion_matrix(y_test, rf_pred, 'Random Forest')

# Model sonuçlarını değerlendirirken
print("\nKullanılan Feature'lar:")
for f in features:
    print(f"- {f}")

print("\nSınıf Dağılımı:")
print(df['injury_category'].value_counts(sort=False)) 