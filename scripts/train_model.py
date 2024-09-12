import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import mean_squared_error
import joblib

# Veri yükleme ve işleme fonksiyonu
def load_data(file_path):
    df = pd.read_csv(file_path, encoding='utf-8')
    return df

# Modeli eğitme ve tahmin sonuçlarını kaydetme fonksiyonu
def train_model(df, save_predictions=False):
    # Özellikler ve hedef değişken
    X = df.drop(columns=['TARGET'])
    y = df['TARGET']

    # Model ve hiperparametreler
    model = Ridge()
    param_grid = {'alpha': [0.01, 0.1, 1, 10, 100]}
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='neg_mean_squared_error')
    grid_search.fit(X, y)

    # En iyi model ve hiperparametreler
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_

    # Tahmin ve performans ölçümü
    y_pred = best_model.predict(X)
    rmse = np.sqrt(mean_squared_error(y, y_pred))

    # Çapraz doğrulama skorları
    cv_scores = cross_val_score(best_model, X, y, cv=5, scoring='neg_mean_squared_error')
    cv_rmse = np.sqrt(-cv_scores.mean())

    # Çapraz doğrulama skorlarını dosyaya yazma
    pd.DataFrame(cv_scores).to_csv(r'C:\Users\emirh\Desktop\DOSYALAR\veri_bilimi\gamegain\data\processed\cross_validation_scores.csv', index=False, header=False)

    # Tahmin sonuçlarını kaydetme (isteğe bağlı)
    if save_predictions:
        predictions_df = pd.DataFrame({'Actual': y, 'Predicted': y_pred})
        predictions_df.to_csv(r'C:\Users\emirh\Desktop\DOSYALAR\veri_bilimi\gamegain\submissions\predictions.csv', index=False)

    # Modeli kaydetme
    joblib.dump(best_model, r'C:\Users\emirh\Desktop\DOSYALAR\veri_bilimi\gamegain\models\ridge_model.pkl')

    return best_params, rmse, cv_rmse

# Ana fonksiyon
def main():
    file_path = r'C:\Users\emirh\Desktop\DOSYALAR\veri_bilimi\gamegain\data\processed\selected_features_dataset.csv'
    df = load_data(file_path)
    best_params, rmse, cv_rmse = train_model(df, save_predictions=True)

    print("En iyi Hiperparametreler:", best_params)
    print("RMSE:", rmse)
    print("Cross-Validation RMSE:", cv_rmse)

# Çalıştır
if __name__ == "__main__":
    main()
