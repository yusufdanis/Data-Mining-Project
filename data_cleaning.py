import pandas as pd
import numpy as np

# Veri setini yükleme
df = pd.read_csv('dataset.csv')

# p_id2'yi p_id olarak değiştir
df = df.rename(columns={'p_id2': 'p_id'})

# Temizleme öncesi durum raporu
with open('data_cleaning_report.txt', 'w', encoding='utf-8') as f:
    f.write("=== VERİ TEMİZLEME RAPORU ===\n\n")
    f.write("BAŞLANGIÇ DURUMU:\n")
    f.write(f"Satır Sayısı: {df.shape[0]}\n")
    f.write(f"Sütun Sayısı: {df.shape[1]}\n")
    f.write("\nEksik Değerler:\n")
    f.write(df.isnull().sum().to_string())
    
    # Önemli sütunları belirle
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
    
    # Önemli sütunlardaki eksik değerleri kontrol et
    f.write("\n\nÖNEMLİ SÜTUNLARDA EKSİK DEĞERLER:\n")
    for col in important_columns:
        missing_count = df[col].isnull().sum()
        missing_percentage = (missing_count / len(df)) * 100
        f.write(f"{col}: {missing_count} (%{missing_percentage:.2f})\n")
    
    # Bu sütunlarda eksik değer içeren satırları kaldır
    initial_rows = len(df)
    df = df.dropna(subset=important_columns)
    removed_rows = initial_rows - len(df)
    
    f.write(f"\nKALDIRILAN SATIRLAR:\n")
    f.write(f"Toplam kaldırılan satır sayısı: {removed_rows}\n")
    f.write(f"Kaldırılan satır yüzdesi: %{(removed_rows/initial_rows)*100:.2f}\n")
    
    f.write("\nTEMİZLEME SONRASI DURUM:\n")
    f.write(f"Kalan satır sayısı: {len(df)}\n")
    f.write(f"Sütun sayısı: {df.shape[1]}\n")
    f.write("\nKalan Eksik Değerler:\n")
    f.write(df.isnull().sum().to_string())
    
    # P_ID ve DOB analizi
    f.write("\n\n=== P_ID VE DOB TUTARSIZLIK ANALİZİ ===\n")
    
    # Her p_id için benzersiz dob sayısını kontrol et
    id_dob_counts = df.groupby('p_id')['dob'].nunique()
    inconsistent_ids = id_dob_counts[id_dob_counts > 1]
    
    if len(inconsistent_ids) > 0:
        f.write(f"\nFarklı doğum tarihi olan p_id sayısı: {len(inconsistent_ids)}")
        f.write("\n\nTutarsız kayıtların detayları:\n")
        
        for pid in inconsistent_ids.index:
            inconsistent_records = df[df['p_id'] == pid]
            f.write(f"\nP_ID: {pid}")
            f.write("\n" + inconsistent_records[['p_id', 'dob', 'nationality', 'position', 'age', 'fifa_rating']].to_string())
            
            # Bu p_id için benzersiz doğum tarihlerini kontrol et
            unique_dobs = inconsistent_records['dob'].unique()
            
            # Her benzersiz doğum tarihi için yeni p_id oluştur
            for dob in unique_dobs:
                birth_year = dob[:4]  # Sadece yılı al (ilk 4 karakter)
                mask = (df['p_id'] == pid) & (df['dob'] == dob)
                new_id = f"{pid}_{birth_year}"
                df.loc[mask, 'p_id'] = new_id
                f.write(f"\nYeni P_ID oluşturuldu (birth year): {new_id}")
    
    # Son durum kontrolü
    f.write("\n\n=== P_ID YENİDEN İSİMLENDİRME SONRASI DURUM ===\n")
    
    # DOB kontrolü
    last_check = df.groupby('p_id')['dob'].nunique()
    remaining_inconsistent = last_check[last_check > 1]
    
    if len(remaining_inconsistent) > 0:
        f.write(f"\nHala DOB tutarsızlığı olan p_id sayısı: {len(remaining_inconsistent)}")
        f.write("\nKalan tutarsız kayıtlar:\n")
        for pid in remaining_inconsistent.index:
            inconsistent_records = df[df['p_id'] == pid]
            f.write(f"\nP_ID: {pid}\n")
            f.write(inconsistent_records[['p_id', 'dob', 'nationality', 'position', 'age']].to_string())
    else:
        f.write("\nTüm P_ID ve DOB tutarsızlıkları düzeltildi.")
    
    # FIFA Rating kontrolü
    rating_check = df.groupby('p_id')['fifa_rating'].nunique()
    rating_inconsistent = rating_check[rating_check > 1]
    
    if len(rating_inconsistent) > 0:
        f.write(f"\n\nFIFA Rating tutarsızlığı olan p_id sayısı: {len(rating_inconsistent)}")
        f.write("\nTutarsız kayıtların detayları:\n")
        for pid in rating_inconsistent.index:
            inconsistent_records = df[df['p_id'] == pid]
            f.write(f"\nP_ID: {pid}")
            f.write("\n" + inconsistent_records[['p_id', 'dob', 'fifa_rating', 'start_year']].to_string())
    else:
        f.write("\nFIFA Rating tutarsızlığı bulunmamaktadır.")
    
    # Position kontrolü
    position_check = df.groupby('p_id')['position'].nunique()
    position_inconsistent = position_check[position_check > 1]
    
    if len(position_inconsistent) > 0:
        f.write(f"\n\nPosition tutarsızlığı olan p_id sayısı: {len(position_inconsistent)}")
        f.write("\nTutarsız kayıtların detayları:\n")
        for pid in position_inconsistent.index:
            inconsistent_records = df[df['p_id'] == pid]
            f.write(f"\nP_ID: {pid}")
            f.write("\n" + inconsistent_records[['p_id', 'dob', 'position', 'start_year']].to_string())
    else:
        f.write("\nPosition tutarsızlığı bulunmamaktadır.")
    
    # Position kontrolünden sonra belirli p_id-position eşleşmelerini koru
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

    # Position tutarsızlığı olan p_id'ler için
    for pid in position_inconsistent.index:
        if pid in correct_mappings:
            # Doğru position dışındaki kayıtları sil
            mask = (df['p_id'] == pid) & (df['position'] != correct_mappings[pid])
            df.drop(df[mask].index, inplace=True)
            
    f.write("\n\nBelirtilen p_id-position eşleşmeleri dışındaki kayıtlar silindi.")
    
    # Silme işlemi sonrası tekrar position kontrolü
    f.write("\n\n=== KAYIT SİLME SONRASI POSITION KONTROLÜ ===\n")
    position_check_after = df.groupby('p_id')['position'].nunique()
    position_inconsistent_after = position_check_after[position_check_after > 1]
    
    if len(position_inconsistent_after) > 0:
        f.write(f"\nHala position tutarsızlığı olan p_id sayısı: {len(position_inconsistent_after)}")
        f.write("\nKalan tutarsız kayıtların detayları:\n")
        for pid in position_inconsistent_after.index:
            inconsistent_records = df[df['p_id'] == pid]
            f.write(f"\nP_ID: {pid}")
            f.write("\n" + inconsistent_records[['p_id', 'dob', 'position', 'start_year']].to_string())
    else:
        f.write("\nTüm position tutarsızlıkları giderildi.")
    
    # PACE VE PHYSIC ANALİZİ
    f.write("\n\n=== PACE VE PHYSIC EKSİK DEĞER ANALİZİ ===\n")
    missing_both = df[df['pace'].isnull() & df['physic'].isnull()]
    f.write(f"\nHer iki sütunda da eksik olan satır sayısı: {len(missing_both)}")
    
    f.write("\n\nEksik değer olan satırların detayları:\n")
    columns_to_show = ['p_id', 'position', 'age', 'fifa_rating', 'pace', 'physic']
    f.write(missing_both[columns_to_show].to_string())
    
    # Aynı p_id'den pace ve physic değerlerini doldur
    f.write("\n\n=== AYNI P_ID'DEN PACE VE PHYSIC DOLDURMA ===\n")
    filled_count = 0
    
    for idx, row in missing_both.iterrows():
        # Aynı p_id'ye sahip diğer kayıtları bul ve start_year'a göre sırala
        same_id_records = df[(df['p_id'] == row['p_id']) & 
                           (df['pace'].notna()) & 
                           (df['physic'].notna())].sort_values('start_year', ascending=False)
        
        if len(same_id_records) > 0:
            # En yüksek start_year'a sahip kayıttaki değerleri al
            pace_value = same_id_records.iloc[0]['pace']
            physic_value = same_id_records.iloc[0]['physic']
            
            # Değerleri doldur
            df.loc[idx, 'pace'] = pace_value
            df.loc[idx, 'physic'] = physic_value
            filled_count += 1
            
            f.write(f"\nP_ID: {row['p_id']}")
            f.write(f"\nDoldurulan değerler - Pace: {pace_value}, Physic: {physic_value}")
    
    f.write(f"\n\nToplam {filled_count} kayıt dolduruldu.")
    
    # Costelpantilimon için position düzeltmesi
    df.loc[df['p_id'] == 'costelpantilimon', 'position'] = 'Goalkeeper'
    
    # Position numeric değerlerini güncelle
    position_mapping = {
        'Goalkeeper': 1,
        'Defender': 2,
        'Midfielder': 3,
        'Forward': 4
    }
    df['position_numeric'] = df['position'].map(position_mapping)
    
    # Benzer oyunculardan pace ve physic değerlerini doldur
    f.write("\n\n=== BENZER OYUNCULARDAN PACE VE PHYSIC DOLDURMA ===\n")
    similar_filled_count = 0
    
    # Hala eksik değerleri olan kayıtları bul
    still_missing = df[df['pace'].isnull() & df['physic'].isnull()]
    
    for idx, row in still_missing.iterrows():
        if pd.isna(row['position']):
            continue
            
        # Aynı pozisyondaki, değerleri olan oyuncuları bul
        same_position = df[
            (df['position'] == row['position']) & 
            (df['pace'].notna()) & 
            (df['physic'].notna())
        ].copy()
        
        if len(same_position) > 0:
            # Benzerlik skorları hesapla
            same_position.loc[:, 'age_diff'] = abs(same_position['age'] - row['age'])
            same_position.loc[:, 'bmi_diff'] = abs(same_position['bmi'] - row['bmi'])
            same_position.loc[:, 'rating_diff'] = abs(same_position['fifa_rating'] - row['fifa_rating'])
            
            # Normalize et
            same_position.loc[:, 'age_diff_norm'] = same_position['age_diff'] / same_position['age_diff'].max()
            same_position.loc[:, 'bmi_diff_norm'] = same_position['bmi_diff'] / same_position['bmi_diff'].max()
            same_position.loc[:, 'rating_diff_norm'] = same_position['rating_diff'] / same_position['rating_diff'].max()
            
            # Ağırlıklı benzerlik skoru (düşük = daha benzer)
            same_position.loc[:, 'similarity_score'] = (
                0.4 * same_position['age_diff_norm'] +    # Yaşa daha fazla önem
                0.2 * same_position['bmi_diff_norm'] +    # BMI'ya daha az önem
                0.4 * same_position['rating_diff_norm']   # Rating'e daha fazla önem
            )
            
            # En benzer oyuncuyu bul
            most_similar = same_position.nsmallest(1, 'similarity_score').iloc[0]
            
            # Değerleri doldur
            df.loc[idx, 'pace'] = most_similar['pace']
            df.loc[idx, 'physic'] = most_similar['physic']
            similar_filled_count += 1
            
            f.write(f"\nP_ID: {row['p_id']}")
            f.write(f"\nBenzer oyuncu P_ID: {most_similar['p_id']}")
            f.write(f"\nBenzerlik özellikleri:")
            f.write(f"\n  Yaş farkı: {most_similar['age_diff']:.2f}")
            f.write(f"\n  BMI farkı: {most_similar['bmi_diff']:.2f}")
            f.write(f"\n  Rating farkı: {most_similar['rating_diff']:.2f}")
            f.write(f"\nDoldurulan değerler - Pace: {most_similar['pace']}, Physic: {most_similar['physic']}")
    
    f.write(f"\n\nBenzer oyunculardan toplam {similar_filled_count} kayıt dolduruldu.")
    
    # Doldurma sonrası tekrar analiz
    f.write("\n\n=== DOLDURMA SONRASI PACE VE PHYSIC ANALİZİ ===\n")
    missing_both_after = df[df['pace'].isnull() & df['physic'].isnull()]
    f.write(f"\nHer iki sütunda da hala eksik olan satır sayısı: {len(missing_both_after)}")
    
    if len(missing_both_after) > 0:
        f.write("\n\nHala eksik değer olan satırların detayları:\n")
        f.write(missing_both_after[columns_to_show].to_string())
    
    # Tüm temizleme işlemlerinden sonra son durum analizi
    f.write("\n\n=== VERİ SETİ SON DURUM ANALİZİ ===\n")
    
    # Genel bilgiler
    f.write("\nGenel Bilgiler:")
    f.write(f"\nSatır Sayısı: {df.shape[0]}")
    f.write(f"\nSütun Sayısı: {df.shape[1]}")
    
    # Eksik değerler
    f.write("\n\nEksik Değerler:")
    f.write("\n" + df.isnull().sum().to_string())
    
    # Veri Temizleme Analizleri
    f.write("\n\n=== VERİ TEMİZLEME ANALİZLERİ ===\n")
    
    # 1. Aykırı Değer Analizi
    f.write("\n1. AYKIRI DEĞER ANALİZİ\n")
    numeric_cols = ['age', 'fifa_rating', 'bmi', 'pace', 'physic']
    
    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
        
        f.write(f"\n{col.upper()} Aykırı Değer Analizi:")
        f.write(f"\nAlt Sınır: {lower_bound:.2f}")
        f.write(f"\nÜst Sınır: {upper_bound:.2f}")
        f.write(f"\nAykırı Değer Sayısı: {len(outliers)}")
        if len(outliers) > 0:
            f.write("\nAykırı Değerler:")
            f.write("\n" + outliers[['p_id', col, 'position']].to_string())
    
    # 2. Tutarsızlık Analizi
    f.write("\n\n2. TUTARSIZLIK ANALİZİ\n")
    
    # Yaş-Rating Tutarsızlığı
    f.write("\nYaş-Rating Tutarsızlığı:")
    young_high_rated = df[(df['age'] < 20) & (df['fifa_rating'] > 85)]
    old_high_rated = df[(df['age'] > 35) & (df['fifa_rating'] > 85)]
    
    if len(young_high_rated) > 0:
        f.write("\n20 yaş altı yüksek ratingli oyuncular:")
        f.write("\n" + young_high_rated[['p_id', 'age', 'fifa_rating']].to_string())
    
    if len(old_high_rated) > 0:
        f.write("\n35 yaş üstü yüksek ratingli oyuncular:")
        f.write("\n" + old_high_rated[['p_id', 'age', 'fifa_rating']].to_string())
    
    # Position-Pace/Physic Tutarsızlığı
    f.write("\n\nPosition-Pace/Physic Tutarsızlığı:")
    slow_forwards = df[(df['position'] == 'Forward') & (df['pace'] < 60)]
    weak_defenders = df[(df['position'] == 'Defender') & (df['physic'] < 60)]
    
    if len(slow_forwards) > 0:
        f.write("\nDüşük pace'li forvetler:")
        f.write("\n" + slow_forwards[['p_id', 'position', 'pace']].to_string())
    
    if len(weak_defenders) > 0:
        f.write("\nDüşük physic'li defanslar:")
        f.write("\n" + weak_defenders[['p_id', 'position', 'physic']].to_string())
    
    # BMI Tutarsızlığı
    f.write("\n\nBMI Tutarsızlığı:")
    abnormal_bmi = df[(df['bmi'] < 18.5) | (df['bmi'] > 30)]
    if len(abnormal_bmi) > 0:
        f.write("\nAnormal BMI değerleri:")
        f.write("\n" + abnormal_bmi[['p_id', 'bmi']].to_string())
    
    # 3. Veri Tipi Kontrolleri
    f.write("\n\n3. VERİ TİPİ KONTROLLERİ\n")
    f.write("\nSütun Veri Tipleri:")
    f.write("\n" + df.dtypes.to_string())
    
    # 4. Korelasyon Analizi
    f.write("\n\n4. KORELASYON ANALİZİ\n")
    
    # Tüm sayısal sütunları seç
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
        f.write("\nYüksek Korelasyonlu Değişkenler (>0.9):")
        f.write("\n" + "\n".join(high_corr))
    
    # Yüksek korelasyonlu değişkenleri çıkart
    columns_to_drop = [
        'season_games_played',
        'total_games_played',
        'cumulative_games_played'
    ]
    
    df = df.drop(columns=columns_to_drop)
    f.write("\n\nÇıkartılan Yüksek Korelasyonlu Değişkenler:")
    f.write("\n" + ", ".join(columns_to_drop))
    
    # Veri Kalitesi Skoru
    f.write("\n\n=== VERİ KALİTESİ SKORU ===\n")
    
    # 1. Eksik Veri Skoru
    missing_score = (1 - df.isnull().sum().sum() / (df.shape[0] * df.shape[1])) * 100
    f.write(f"\n1. Eksik Veri Skoru: %{missing_score:.2f}")
    
    # 2. Tutarlılık Skoru
    consistency_issues = 0
    # Yaş-Rating tutarsızlığı
    consistency_issues += len(df[(df['age'] < 20) & (df['fifa_rating'] > 85)])
    consistency_issues += len(df[(df['age'] > 35) & (df['fifa_rating'] > 85)])
    # Position-Pace/Physic tutarsızlığı
    consistency_issues += len(df[(df['position'] == 'Forward') & (df['pace'] < 60)])
    consistency_issues += len(df[(df['position'] == 'Defender') & (df['physic'] < 60)])
    # BMI tutarsızlığı
    consistency_issues += len(df[(df['bmi'] < 18.5) | (df['bmi'] > 30)])
    
    consistency_score = (1 - consistency_issues / len(df)) * 100
    f.write(f"\n2. Tutarlılık Skoru: %{consistency_score:.2f}")
    
    # 3. Aykırı Değer Skoru
    outlier_count = 0
    numeric_cols = ['age', 'fifa_rating', 'bmi', 'pace', 'physic']
    
    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        outlier_count += len(df[(df[col] < Q1 - 1.5 * IQR) | (df[col] > Q3 + 1.5 * IQR)])
    
    outlier_score = (1 - outlier_count / (len(df) * len(numeric_cols))) * 100
    f.write(f"\n3. Aykırı Değer Skoru: %{outlier_score:.2f}")
    
    # 4. Veri Tipi Uygunluk Skoru
    # Güncellenmiş sayısal sütunlar listesi (silinen sütunlar çıkartıldı)
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
    
    # Tüm sayısal değişkenlerin gerçekten sayısal olup olmadığını kontrol et
    type_issues = 0
    for col in numeric_cols_corr:
        if not np.issubdtype(df[col].dtype, np.number):
            type_issues += 1
    
    type_score = (1 - type_issues / len(numeric_cols_corr)) * 100
    f.write(f"\n4. Veri Tipi Uygunluk Skoru: %{type_score:.2f}")
    
    # Genel Kalite Skoru (ağırlıklı ortalama)
    quality_score = (
        0.35 * missing_score +      # Eksik veriye daha fazla önem
        0.30 * consistency_score +   # Tutarlılığa önem
        0.20 * outlier_score +      # Aykırı değerlere orta önem
        0.15 * type_score          # Veri tipine daha az önem
    )
    
    f.write(f"\n\nGENEL VERİ KALİTESİ SKORU: %{quality_score:.2f}")

# Temizlenmiş veri setini kaydet
df.to_csv('cleaned_dataset.csv', index=False)
print("Temizlenmiş veri seti 'cleaned_dataset.csv' olarak kaydedildi.")