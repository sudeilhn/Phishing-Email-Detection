import pickle
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# === Model ve Tokenizer yolları ===
model_path = 'C:/Users/sudee/OneDrive/phishing_project/models/backup/phishing_model_cnn_lstm_v2.keras'
tokenizer_path = 'C:/Users/sudee/OneDrive/phishing_project/models/backup/tokenizer_cnn_lstm.pkl'
max_length = 250

# === Model ve tokenizer yükle ===
model = load_model(model_path)

with open(tokenizer_path, 'rb') as f:
    tokenizer = pickle.load(f)

print("Model başarıyla yüklendi.")

# === Kullanıcıdan e-posta bilgilerini al ===
subject = input("Konu: ")
sender = input("Gönderen: ")
content = input("İçerik: ")

# === Metni birleştir ve dönüştür ===
text = subject + " " + sender + " " + content
sequence = tokenizer.texts_to_sequences([text])
padded = pad_sequences(sequence, maxlen=max_length, padding='post', truncating='post')

# === Tahmin yap ===
prediction = model.predict(padded)[0][0]
label = "Oltalama (Phishing)" if prediction > 0.5 else "Güvenilir"

print(f"\n Tahmin: {label}")
print(f" Güven Skoru: {prediction:.4f}")
