# data_preprocessing.py
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

# Verileri yükleme
def load_data():
    users = pd.read_csv(r"C:\Users\emirh\Desktop\DOSYALAR\veri_bilimi\gamegain\data\users_train.csv")
    user_features = pd.read_csv(r"C:\Users\emirh\Desktop\DOSYALAR\veri_bilimi\gamegain\data\user_features_train.csv")
    targets = pd.read_csv(r"C:\Users\emirh\Desktop\DOSYALAR\veri_bilimi\gamegain\data\targets_train.csv")

    # Verileri birleştirme
    data = users.merge(user_features, on='ID').merge(targets, on='ID')
    return data

# Eksik değerlerin temizlenmesi
def handle_missing_values(df):
    # Eksik değerlerin oranını kontrol edelim
    missing_report = df.isnull().mean() * 100
    print("Eksik Değer Raporu:\n", missing_report[missing_report > 0])

    # Belirli kolonlar için eksik değerlerin doldurulması
    df.fillna({
        'first_open_date': df['first_open_date'].mode()[0],  # Tarih verileri için mod değeri
        'ad_network': 'organic'  # Reklam ağı eksikse organik olarak doldur
    }, inplace=True)

    # Sayısal kolonlarda eksik değerleri ortalama ile doldurma
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())

    # Kategorik kolonlarda eksik değerlerin doldurulması
    categorical_cols = ['country', 'device_brand']
    for col in categorical_cols:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].mode()[0])

    # Çok fazla eksik değeri olan sütunları çıkarma
    if 'ad_network' in df.columns:
        df = df.drop(['ad_network'], axis=1)

    return df

# Kategorik ve Boolean verilerin encode edilmesi
def encode_categorical(df):
    le = LabelEncoder()

    # Boolean kolonları 1 ve 0'a dönüştürme
    boolean_cols = df.select_dtypes(include=['bool']).columns
    for col in boolean_cols:
        df[col] = df[col].astype(int)

    # Kategorik kolonları encode et
    categorical_cols = ['country', 'platform', 'device_category', 'device_brand', 'device_model']
    for col in categorical_cols:
        if col in df.columns:
            df[col] = le.fit_transform(df[col].astype(str))

    return df

# Özellik mühendisliği
def feature_engineering(df):
    # Tarih kolonlarından yıl, ay, gün çıkarılması
    df['first_open_year'] = pd.to_datetime(df['first_open_date']).dt.year
    df['first_open_month'] = pd.to_datetime(df['first_open_date']).dt.month
    df['first_open_day'] = pd.to_datetime(df['first_open_date']).dt.day

    # Kullanıcının ilk 15 gün davranışlarını toplayarak yeni özellikler yaratma
    revenue_cols = [col for col in df.columns if 'AdRevenueD' in col]
    df['total_ad_revenue'] = df[revenue_cols].sum(axis=1)

    return df

# Ana fonksiyon
def preprocess_data():
    # Veriyi yükle ve işle
    data = load_data()
    data = handle_missing_values(data)
    data = encode_categorical(data)
    data = feature_engineering(data)

    # İşlenmiş veriyi kaydetme
    data.to_csv(r'C:\Users\emirh\Desktop\DOSYALAR\veri_bilimi\gamegain\data\processed\train_data_processed.csv', index=False, encoding='utf-8-sig')
    print("Veri ön işleme tamamlandı ve kaydedildi.")

if __name__ == "__main__":
    preprocess_data()
