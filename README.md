# GameGain: Oyun Kullanıcılarının Kazanç Tahmini

## Proje Özeti

GameGain, oyun kullanıcılarının ilk 15 günlük davranış verilerine dayanarak, kullanıcıların sonraki 90 gün boyunca ne kadar kazanç sağlayacaklarını tahmin eden bir model geliştirmeyi amaçlayan bir projedir. Proje, veri ön işleme, özellik mühendisliği, model eğitimi ve değerlendirmesi süreçlerini içerir.

## İçindekiler

- [GameGain: Oyun Kullanıcılarının Kazanç Tahmini](#gamegain-oyun-kullanıcılarının-kazanç-tahmini)
  - [Proje Özeti](#proje-özeti)
  - [İçindekiler](#i̇çindekiler)
  - [Veri Setleri](#veri-setleri)
  - [Proje Adımları](#proje-adımları)
  - [Kurulum ve Kullanım](#kurulum-ve-kullanım)
  - [Yazarlar](#yazarlar)

## Veri Setleri

Projede kullanılan veri setleri şunlardır:

1. **users_train.csv**: Kullanıcı metadata verileri (eğitim seti)
   - `user_id`: Kullanıcı kimliği
   - `age`: Kullanıcının yaşı
   - `gender`: Kullanıcının cinsiyeti
   - `country`: Kullanıcının ülkesi

2. **user_features_train.csv**: Kullanıcı davranış verileri (ilk 15 gün) (eğitim seti)
   - `user_id`: Kullanıcı kimliği
   - `feature_1`: Davranış özelliği 1
   - `feature_2`: Davranış özelliği 2
   - `feature_3`: Davranış özelliği 3

3. **targets_train.csv**: Kullanıcıların ilk 90 gün toplam gelir hedefleri (eğitim seti)
   - `user_id`: Kullanıcı kimliği
   - `total_revenue`: Toplam gelir

4. **users_test.csv**: Kullanıcı metadata verileri (test seti)

5. **user_features_test.csv**: Kullanıcı davranış verileri (ilk 15 gün) (test seti)

6. **processed/**: İşlenmiş veri dosyaları ve sonuçlar

## Proje Adımları

1. **Veri Ön İşleme:**
   - Verilerin birleştirilmesi ve temizlenmesi
   - Eksik verilerin işlenmesi

2. **Özellik Mühendisliği:**
   - Özelliklerin oluşturulması ve seçilmesi
   - Verilerin normalize edilmesi

3. **Model Eğitimi:**
   - Ridge regresyon modeli kullanılarak eğitim
   - Hiperparametre optimizasyonu

4. **Model Değerlendirmesi:**
   - Performans değerlendirmesi (RMSE, çapraz doğrulama)

5. **Tahmin:**
   - Test verileri üzerinde tahminler yapılması
   - Tahmin sonuçlarının kaydedilmesi ve gönderilmesi

## Kurulum ve Kullanım

1. **Gereksinimler:**
   - Python 3.x
   - Pandas
   - NumPy
   - Scikit-learn
   - Joblib

2. **Kurulum:**
   ```bash
   pip install pandas numpy scikit-learn joblib
   ```
3. **Projeyi Çalıştırma:**
    - Repo'yu klonlayın:
    git clone https://github.com/username/gamegain

    - Veri işleme ve model eğitimi için gerekli scriptsleri çalıştırın:

    ```bash
    python scripts/data_preprocessing.py
    python scripts/feature_selection.py
    python scripts/train_model.py
    python scripts/evaluate_model.py
    ```
## Yazarlar

- Harun Emirhan - [LinkedIn Profilim](https://www.linkedin.com/in/harun-emirhan-bostanci-24144726b/)
