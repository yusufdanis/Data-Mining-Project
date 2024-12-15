import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Türkçe karakter desteği
plt.rcParams['font.family'] = 'DejaVu Sans'

# Temizlenmiş veriyi yükle
df = pd.read_csv('cleaned_dataset.csv')

# Görselleştirmeler için figür boyutunu ayarla
plt.figure(figsize=(12, 8))

# 1. Hedef Değişken Analizi (Yaralanma Süresi)
plt.subplot(2, 2, 1)
sns.histplot(data=df, x='season_days_injured', bins=30)
plt.title('Sezon İçi Yaralanma Süresi Dağılımı')
plt.xlabel('Yaralanma Süresi (Gün)')
plt.ylabel('Frekans')

# 2. Pozisyonlara Göre Yaralanma Süresi
plt.subplot(2, 2, 2)
sns.boxplot(data=df, x='position', y='season_days_injured')
plt.title('Pozisyonlara Göre Yaralanma Süresi')
plt.xlabel('Pozisyon')
plt.ylabel('Yaralanma Süresi (Gün)')
plt.xticks(rotation=45)

# 3. Yaş ve Yaralanma Süresi İlişkisi
plt.subplot(2, 2, 3)
sns.scatterplot(data=df, x='age', y='season_days_injured')
plt.title('Yaş ve Yaralanma Süresi İlişkisi')
plt.xlabel('Yaş')
plt.ylabel('Yaralanma Süresi (Gün)')

# 4. FIFA Rating ve Yaralanma Süresi İlişkisi
plt.subplot(2, 2, 4)
sns.scatterplot(data=df, x='fifa_rating', y='season_days_injured')
plt.title('FIFA Rating ve Yaralanma Süresi İlişkisi')
plt.xlabel('FIFA Rating')
plt.ylabel('Yaralanma Süresi (Gün)')

plt.tight_layout()
plt.savefig('eda_plots_1.png')
plt.close()

# İkinci set görselleştirmeler
plt.figure(figsize=(12, 8))

# 5. BMI Dağılımı
plt.subplot(2, 2, 1)
sns.histplot(data=df, x='bmi', bins=30)
plt.title('BMI Dağılımı')
plt.xlabel('BMI')
plt.ylabel('Frekans')

# 6. Pozisyonlara Göre Pace ve Physic
plt.subplot(2, 2, 2)
df_melted = df.melt(id_vars=['position'], value_vars=['pace', 'physic'])
sns.boxplot(data=df_melted, x='position', y='value', hue='variable')
plt.title('Pozisyonlara Göre Pace ve Physic')
plt.xlabel('Pozisyon')
plt.ylabel('Değer')
plt.xticks(rotation=45)

# 7. Önceki Sezon Yaralanma Süresi ve Bu Sezon İlişkisi
plt.subplot(2, 2, 3)
sns.scatterplot(data=df, x='season_days_injured_prev_season', y='season_days_injured')
plt.title('Önceki ve Bu Sezon Yaralanma Süresi')
plt.xlabel('Önceki Sezon Yaralanma (Gün)')
plt.ylabel('Bu Sezon Yaralanma (Gün)')

# 8. Oynanan Dakika ve Yaralanma İlişkisi
plt.subplot(2, 2, 4)
sns.scatterplot(data=df, x='season_minutes_played', y='season_days_injured')
plt.title('Oynanan Dakika ve Yaralanma Süresi')
plt.xlabel('Oynanan Dakika')
plt.ylabel('Yaralanma Süresi (Gün)')

plt.tight_layout()
plt.savefig('eda_plots_2.png')
plt.close()

# Üçüncü set görselleştirmeler
plt.figure(figsize=(12, 8))

# 9. Work Rate ve Yaralanma İlişkisi
plt.subplot(2, 2, 1)
sns.boxplot(data=df, x='work_rate', y='season_days_injured')
plt.title('Work Rate ve Yaralanma Süresi')
plt.xlabel('Work Rate')
plt.ylabel('Yaralanma Süresi (Gün)')
plt.xticks(rotation=45)

# 10. Önceki Sezon Önemli Yaralanma ve Bu Sezon Yaralanma İlişkisi
plt.subplot(2, 2, 2)
sns.boxplot(data=df, x='significant_injury_prev_season', y='season_days_injured')
plt.title('Önceki Sezon Önemli Yaralanma Etkisi')
plt.xlabel('Önceki Sezon Önemli Yaralanma')
plt.ylabel('Yaralanma Süresi (Gün)')

# 11. Milliyetlere Göre Yaralanma (En çok yaralanan top 10)
plt.subplot(2, 2, 3)
nationality_injury = df.groupby('nationality')['season_days_injured'].mean().sort_values(ascending=False).head(10)
sns.barplot(x=nationality_injury.index, y=nationality_injury.values)
plt.title('Milliyetlere Göre Ortalama Yaralanma Süresi (Top 10)')
plt.xlabel('Milliyet')
plt.ylabel('Ort. Yaralanma Süresi (Gün)')
plt.xticks(rotation=90)

# 12. Sezonlara Göre Yaralanma Trendi
plt.subplot(2, 2, 4)
season_injury = df.groupby('start_year')['season_days_injured'].mean()
sns.lineplot(x=season_injury.index, y=season_injury.values)
plt.title('Sezonlara Göre Ortalama Yaralanma Süresi')
plt.xlabel('Sezon')
plt.ylabel('Ort. Yaralanma Süresi (Gün)')

plt.tight_layout()
plt.savefig('eda_plots_3.png')
plt.close()

# Raporu genişlet
with open('eda_report.txt', 'w', encoding='utf-8') as f:
    f.write("=== DETAYLI VERİ ANALİZİ RAPORU ===\n")
    
    # 1. Yaralanma Süresi Dağılımı
    injury_stats = df['season_days_injured'].describe()
    f.write("\n1. YARALANMA SÜRESİ DAĞILIMI ANALİZİ")
    f.write(f"\n- Ortalama yaralanma süresi: {injury_stats['mean']:.2f} gün")
    f.write(f"\n- Medyan yaralanma süresi: {injury_stats['50%']:.2f} gün")
    f.write(f"\n- Standart sapma: {injury_stats['std']:.2f} gün")
    f.write("\n- Dağılım sağa çarpık (pozitif çarpıklık)")
    f.write("\n- Çoğu yaralanma 0-100 gün arasında")
    f.write(f"\n- Maksimum yaralanma süresi: {injury_stats['max']:.0f} gün")

    # 2. Pozisyonlara Göre Analiz
    f.write("\n\n2. POZİSYONLARA GÖRE ANALİZ")
    for pos in df['position'].unique():
        pos_data = df[df['position'] == pos]['season_days_injured']
        outliers = pos_data[pos_data > pos_data.quantile(0.75) + 1.5 * (pos_data.quantile(0.75) - pos_data.quantile(0.25))]
        
        f.write(f"\n\n{pos} Pozisyonu:")
        f.write(f"\n- Ortalama yaralanma: {pos_data.mean():.2f} gün")
        f.write(f"\n- Medyan yaralanma: {pos_data.median():.2f} gün")
        f.write(f"\n- Aykırı değer sayısı: {len(outliers)}")
        if len(outliers) > 0:
            f.write(f"\n- En yüksek aykırı değer: {outliers.max():.0f} gün")

    # 3. Yaş-Yaralanma İlişkisi
    f.write("\n\n3. YAŞ-YARALANMA İLİŞKİSİ")
    age_bins = [0, 20, 25, 30, 35, 100]
    age_labels = ['<20', '20-25', '25-30', '30-35', '>35']
    df['age_group'] = pd.cut(df['age'], bins=age_bins, labels=age_labels)
    age_analysis = df.groupby('age_group', observed=True)['season_days_injured'].agg(['mean', 'median', 'count'])
    
    for age_group in age_labels:
        stats = age_analysis.loc[age_group]
        f.write(f"\n\n{age_group} yaş grubu:")
        f.write(f"\n- Ortalama yaralanma: {stats['mean']:.2f} gün")
        f.write(f"\n- Medyan yaralanma: {stats['median']:.2f} gün")
        f.write(f"\n- Oyuncu sayısı: {stats['count']}")
    
    # 4. FIFA Rating Analizi
    f.write("\n\n4. FIFA RATING ANALİZİ")
    # ... (aynı içerik)

    # 5. BMI Dağılımı
    bmi_stats = df['bmi'].describe()
    f.write("\n\n5. BMI DAĞILIMI ANALİZİ")
    f.write(f"\n- BMI aralığı: {bmi_stats['min']:.1f} - {bmi_stats['max']:.1f}")
    f.write(f"\n- Ortalama BMI: {bmi_stats['mean']:.1f}")
    f.write(f"\n- Medyan BMI: {bmi_stats['50%']:.1f}")
    f.write(f"\n- Standart sapma: {bmi_stats['std']:.1f}")

    # 6. Pace ve Physic Analizi
    f.write("\n\n6. PACE VE PHYSIC ANALİZİ")
    for pos in df['position'].unique():
        pos_data = df[df['position'] == pos]
        f.write(f"\n\n{pos} Pozisyonu:")
        f.write(f"\n- Ortalama Pace: {pos_data['pace'].mean():.1f}")
        f.write(f"\n- Ortalama Physic: {pos_data['physic'].mean():.1f}")
        f.write(f"\n- Pace aralığı: {pos_data['pace'].min():.0f}-{pos_data['pace'].max():.0f}")
        f.write(f"\n- Physic aralığı: {pos_data['physic'].min():.0f}-{pos_data['physic'].max():.0f}")

    # 7. Önceki Sezon Yaralanma Analizi
    f.write("\n\n7. ÖNCEKİ SEZON YARALANMA ANALİZİ")
    prev_injury_stats = df.groupby('significant_injury_prev_season')['season_days_injured'].describe()
    for injury_status in [0, 1]:
        stats = prev_injury_stats.loc[injury_status]
        status_text = "Önemli Yaralanma Geçirenler" if injury_status == 1 else "Önemli Yaralanma Geçirmeyenler"
        f.write(f"\n\n{status_text}:")
        f.write(f"\n- Oyuncu sayısı: {stats['count']:.0f}")
        f.write(f"\n- Ortalama yaralanma süresi: {stats['mean']:.1f} gün")
        f.write(f"\n- Medyan yaralanma süresi: {stats['50%']:.1f} gün")
        f.write(f"\n- Maksimum yaralanma süresi: {stats['max']:.0f} gün")

    # 8. Sezon Dakika-Yaralanma Analizi
    f.write("\n\n8. SEZON DAKİKA-YARALANMA ANALİZİ")
    minutes_stats = df.groupby('season_days_injured')['season_minutes_played'].describe()
    f.write(f"\n- Ortalama oynanan dakika: {df['season_minutes_played'].mean():.1f}")
    f.write(f"\n- Medyan oynanan dakika: {df['season_minutes_played'].median():.1f}")
    f.write(f"\n- Maksimum oynanan dakika: {df['season_minutes_played'].max():.0f}")
    f.write(f"\n- Minimum oynanan dakika: {df['season_minutes_played'].min():.0f}")