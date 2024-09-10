# feature_engineering.py
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np

def load_data():
    data = pd.read_csv(r'C:\Users\emirh\Desktop\DOSYALAR\veri_bilimi\gamegain\data\processed\train_data_processed.csv')
    return data

def feature_engineering(df):
    # Tarih kolonlarından yıl, ay, gün çıkarılması
    df['first_open_year'] = pd.to_datetime(df['first_open_date']).dt.year
    df['first_open_month'] = pd.to_datetime(df['first_open_date']).dt.month
    df['first_open_day'] = pd.to_datetime(df['first_open_date']).dt.day

    # Kullanıcının ilk 15 gün davranışlarını toplayarak yeni özellikler yaratma
    revenue_cols = [col for col in df.columns if 'RevenueD' in col]
    df['total_ad_revenue'] = df[revenue_cols].sum(axis=1)

    # Özellik mühendisliği için diğer işlemler
    df['total_level_count'] = df.filter(like='LevelAdvancedCountD').sum(axis=1)
    df['average_level_duration'] = df.filter(like='Level_Duration').mean(axis=1)

    return df

def save_data(df):
    df.to_csv(r'C:\Users\emirh\Desktop\DOSYALAR\veri_bilimi\gamegain\data\processed\train_data_features_engineered.csv', index=False)
    print("Özellik mühendisliği tamamlandı ve kaydedildi.")

if __name__ == "__main__":
    data = load_data()
    data = feature_engineering(data)
    save_data(data)
