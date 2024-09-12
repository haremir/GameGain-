import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest, f_classif

# Veri yükleme ve işleme fonksiyonu
def load_and_preprocess_data(file_path):
    # Veriyi yükle, encoding ile Türkçe karakter sorununu çözelim
    df = pd.read_csv(file_path, encoding='utf-8')

    # Sütunları yazdır
    print("Yüklenen veri sütunları:", df.columns)

    # Tarih sütunlarını çıkarıyoruz
    date_columns = ['first_open_date', 'first_open_timestamp', 'local_first_open_timestamp']
    df = df.drop(columns=date_columns, errors='ignore')

    # Sadece sayısal sütunları alalım
    numeric_df = df.select_dtypes(include=[np.number])

    # 'TARGET' sütununun kontrolü
    if 'TARGET' not in numeric_df.columns:
        raise ValueError("'TARGET' sütunu veri setinde bulunamadı.")

    # İlk 1000 satırı alarak işlemleri hızlandırabiliriz (örnekleme)
    numeric_df = numeric_df.head(1000)  # Performans testleri için

    return numeric_df

# Özellik seçimi ve veri seti oluşturma fonksiyonu
def feature_selection(df, k=20):
    # 'TARGET' dışındaki sütunlar
    X = df.drop(columns=['TARGET'])
    y = df['TARGET']

    # Özellik seçimi işlemi
    selector = SelectKBest(score_func=f_classif, k=min(k, X.shape[1]))  # k değerini sütun sayısı ile sınırla
    X_selected = selector.fit_transform(X, y)

    # Seçilen sütun isimleri
    selected_columns = X.columns[selector.get_support()]

    # Yeni veri setini oluştur
    selected_df = pd.DataFrame(X_selected, columns=selected_columns)
    selected_df['TARGET'] = y.values

    print(f"Seçilen özellikler: {selected_columns.tolist()}")
    return selected_df

# Ana fonksiyon
def main():
    file_path = r'C:\Users\emirh\Desktop\DOSYALAR\veri_bilimi\gamegain\data\processed\train_data_processed.csv'  # İşlenmiş veri setinin yolu
    df = load_and_preprocess_data(file_path)  # Veriyi yükle ve ön işlemleri yap
    selected_df = feature_selection(df, k=20)  # Özellik seçimini yap

    # Seçilen özelliklerden oluşan veri setini kaydet
    selected_df.to_csv(r'C:\Users\emirh\Desktop\DOSYALAR\veri_bilimi\gamegain\data\processed\selected_features_dataset.csv', index=False)
    print("Seçilen özellikler veri seti oluşturuldu ve kaydedildi.")

# Çalıştır
if __name__ == "__main__":
    main()
