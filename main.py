import pandas as pd

# Veri setini yükleme
df = pd.read_csv('dataset.csv')

# Sütun bilgilerini görüntüleme
print("Sütun Sayısı:", len(df.columns))
print("\nSütun İsimleri ve Veri Tipleri:\n")
print(df.dtypes)

# Temel istatistiksel bilgiler
print("\nVeri Seti Özeti:\n")
print(df.describe())

# Eksik değer analizi
print("\nEksik Değer Sayıları:\n")
print(df.isnull().sum())