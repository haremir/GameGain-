import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
import joblib

# Modeli yükleme
model = joblib.load(r'C:\Users\emirh\Desktop\DOSYALAR\veri_bilimi\gamegain\models\ridge_model.pkl')

# Test verilerini yükleme
def load_data(file_path):
    df = pd.read_csv(file_path, encoding='utf-8')
    return df

# Eğitim verileri ile tahmin yapma ve değerlendirme
def evaluate_model(df):
    # Özellikler ve hedef değişken
    X = df.drop(columns=['TARGET'])
    y = df['TARGET']
    
    # Tahmin yapma
    y_pred = model.predict(X)
    
    # Performans ölçümleri
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    
    # Çapraz doğrulama skorları
    cv_scores = np.array(pd.read_csv(r'C:\Users\emirh\Desktop\DOSYALAR\veri_bilimi\gamegain\data\processed\cross_validation_scores.csv'))  # Çapraz doğrulama skorlarını okuma
    cv_rmse = np.sqrt(-cv_scores.mean())

    print("Model Performans Değerlendirmesi:")
    print("RMSE:", rmse)
    print("Cross-Validation RMSE:", cv_rmse)

# Ana fonksiyon
def main():
    file_path = r'C:\Users\emirh\Desktop\DOSYALAR\veri_bilimi\gamegain\data\processed\selected_features_dataset.csv'
    df = load_data(file_path)
    evaluate_model(df)

# Çalıştır
if __name__ == "__main__":
    main()
