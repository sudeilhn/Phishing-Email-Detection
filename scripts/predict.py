import tensorflow as tf
import pickle
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences

# === Yükleme ayarları ===
model_path = 'C:/Users/sudee/OneDrive/phishing_project/models/backup/phishing_model_lstm_v2.keras'
tokenizer_path = 'C:/Users/sudee/OneDrive/phishing_project/models/backup/tokenizer_v1.pkl'

# === Tokenizer'ı yükle ===
with open(tokenizer_path, 'rb') as f:
    tokenizer = pickle.load(f)

# === Modeli yükle ===
model = tf.keras.models.load_model(model_path)

# === Tahmin fonksiyonu ===
def predict_email(konu, gonderen, icerik):
    metin = f"{konu} {gonderen} {icerik}"
    sequence = tokenizer.texts_to_sequences([metin])
    padded = pad_sequences(sequence, maxlen=250, padding='post', truncating='post')
    prediction = model.predict(padded)[0][0]
    
    if prediction > 0.5:
        sonuc = "Oltalama "
    else:
        sonuc = "Güvenilir "
    
    print(f"\n--- Tahmin Sonucu ---")
    print(f"Model Skoru: {prediction:.4f}")
    print(f"Sınıf: {sonuc}")
    return sonuc

# === Ana çalışma bloğu ===
if __name__ == "__main__":
    print("=== E-Posta Tahmin Aracı ===\n")
    konu = input("Konu: ").strip()
    gonderen = input("Gönderen: ").strip()
    icerik = input("İçerik: ").strip()

    predict_email(konu, gonderen, icerik)






