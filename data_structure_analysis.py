import pandas as pd
import numpy as np

# Veri setini yükleme
df = pd.read_csv('dataset.csv')

# Sonuçları dosyaya kaydet
with open('data_structure_analysis_report.txt', 'w', encoding='utf-8') as f:
    # 1. GENEL BİLGİLER
    f.write("=== FUTBOLCU YARALANMA VERİ SETİ DETAYLI ANALİZİ ===\n\n")
    f.write("1. GENEL BİLGİLER\n")
    f.write(f"Satır Sayısı: {df.shape[0]}\n")
    f.write(f"Sütun Sayısı: {df.shape[1]}\n")
    f.write(f"Veri Seti Boyutu: {df.memory_usage().sum() / 1024:.2f} KB\n")
    f.write(f"Toplam Veri Noktası Sayısı: {df.shape[0] * df.shape[1]}\n")
    f.write(f"Bellek Kullanımı (Sütun Bazlı):\n{df.memory_usage(deep=True).to_string()}\n\n")
    
    # 2. SÜTUN BİLGİLERİ
    f.write("2. SÜTUNLAR VE VERİ TİPLERİ\n")
    f.write(df.dtypes.to_string())
    f.write("\n\nSütun İsimleri:\n")
    for col in df.columns:
        f.write(f"- {col}\n")
    f.write("\n")
    
    # 3. EKSİK DEĞER ANALİZİ
    f.write("3. EKSİK DEĞER ANALİZİ\n")
    missing_values = df.isnull().sum()
    missing_percentages = (missing_values / len(df)) * 100
    total_missing = df.isnull().sum().sum()
    total_cells = np.prod(df.shape)
    total_missing_percentage = (total_missing / total_cells) * 100
    
    missing_info = pd.DataFrame({
        'Eksik Değer Sayısı': missing_values,
        'Eksik Değer Yüzdesi': missing_percentages
    })
    f.write(missing_info[missing_info['Eksik Değer Sayısı'] > 0].to_string())
    f.write(f"\n\nToplam Eksik Değer Sayısı: {total_missing}\n")
    f.write(f"Toplam Eksik Değer Yüzdesi: {total_missing_percentage:.2f}%\n\n")
    
    # 4. TEMEL İSTATİSTİKLER
    f.write("4. TEMEL İSTATİSTİKLER\n")
    f.write(df.describe(include='all').to_string())  # Tüm sütunlar için istatistikler
    f.write("\n\n")
    
    # 5. KATEGORİK DEĞİŞKEN BİLGİLERİ
    f.write("5. KATEGORİK DEĞİŞKEN BİLGİLERİ\n")
    categorical_columns = df.select_dtypes(include=['object']).columns
    for col in categorical_columns:
        f.write(f"\n{col}:\n")
        f.write(f"Unique değer sayısı: {df[col].nunique()}\n")
        f.write(f"İlk 5 unique değer: {df[col].unique()[:5]}\n")
        f.write("Değer dağılımı:\n")
        f.write(df[col].value_counts().head().to_string())
        f.write(f"\nEn sık görülen değer: {df[col].mode()[0]}\n")
        f.write("\n")
    
    # 6. AYKIRI DEĞER BİLGİLERİ
    f.write("\n6. AYKIRI DEĞER BİLGİLERİ\n")
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
            f.write(f"Aykırı değer sayısı: {len(outliers)}\n")
            f.write(f"Aykırı değer yüzdesi: {(len(outliers) / len(df)) * 100:.2f}%\n")
            f.write(f"Aykırı değer aralığı: [{outliers.min():.2f}, {outliers.max():.2f}]\n")
            f.write(f"Normal aralık: [{lower_bound:.2f}, {upper_bound:.2f}]\n")
    
    # 7. VERİ TUTARLILIĞI KONTROLÜ
    f.write("\n7. VERİ TUTARLILIĞI KONTROLÜ\n")
    if 'dob' in df.columns:
        f.write("\nDoğum tarihi analizi:\n")
        f.write(f"En küçük tarih: {df['dob'].min()}\n")
        f.write(f"En büyük tarih: {df['dob'].max()}\n")
    
    if 'height_cm' in df.columns and 'weight_kg' in df.columns:
        f.write("\nBoy ve kilo değerleri:\n")
        # Convert to numeric, handling any errors
        df['height_cm'] = pd.to_numeric(df['height_cm'], errors='coerce')
        df['weight_kg'] = pd.to_numeric(df['weight_kg'], errors='coerce')
        
        f.write(f"Boy aralığı: [{df['height_cm'].min():.2f}, {df['height_cm'].max():.2f}] cm\n")
        f.write(f"Kilo aralığı: [{df['weight_kg'].min():.2f}, {df['weight_kg'].max():.2f}] kg\n")
        
        # BMI hesaplama ve analizi
        df['bmi'] = df['weight_kg'] / ((df['height_cm']/100) ** 2)
        f.write(f"\nBMI İstatistikleri:\n")
        f.write(f"Ortalama BMI: {df['bmi'].mean():.2f}\n")
        f.write(f"BMI aralığı: [{df['bmi'].min():.2f}, {df['bmi'].max():.2f}]\n")
    
    # 8. KORELASYON BİLGİLERİ
    f.write("\n8. KORELASYON BİLGİLERİ\n")
    correlation_matrix = df[numeric_columns].corr()
    
    # Tüm korelasyonlar
    f.write("\nTüm Korelasyonlar:\n")
    f.write(correlation_matrix.to_string())
    
    # Yüksek korelasyonlar
    high_correlation = np.where(np.abs(correlation_matrix) > 0.7)
    high_correlation = [(correlation_matrix.index[x], correlation_matrix.columns[y], correlation_matrix.iloc[x, y]) 
                       for x, y in zip(*high_correlation) if x != y]
    if high_correlation:
        f.write("\n\nYüksek korelasyonlu değişkenler (>0.7):\n")
        for var1, var2, corr in high_correlation:
            f.write(f"{var1} - {var2}: {corr:.2f}\n")
    
    # 9. VERİ KALİTESİ SKORU
    f.write("\n9. VERİ KALİTESİ SKORU\n")
    completeness = 1 - (total_missing / total_cells)
    uniqueness = np.mean([df[col].nunique()/len(df) for col in df.columns])
    f.write(f"Veri Tamlık Skoru: {completeness:.2%}\n")
    f.write(f"Veri Benzersizlik Skoru: {uniqueness:.2%}\n")