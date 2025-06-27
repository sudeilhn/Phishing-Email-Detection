import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences

# === Hiperparametreler ===
vocab_size = 5000
max_length = 250
oov_token = "<OOV>"
padding_type = 'post'
trunc_type = 'post'

# === Model ve tokenizer yolu ===
model_path = 'C:/Users/sudee/OneDrive/phishing_project/models/backup/phishing_model_gru_v2.keras'
tokenizer_path = 'C:/Users/sudee/OneDrive/phishing_project/models/backup/tokenizer_gru.pkl'

# === Tokenizer'ı yükle ===
with open(tokenizer_path, 'rb') as f:
    tokenizer = pickle.load(f)

# === Eğitilmiş modeli yükle ===
model = tf.keras.models.load_model(model_path)

# === Kullanıcıdan giriş al ===
print("Tahmin etmek istediğiniz e-posta bilgilerini giriniz:")
subject = input("Konu: ")
sender = input("Gönderen: ")
body = input("İçerik: ")

# === Metni birleştir ===
text = subject.strip() + " " + sender.strip() + " " + body.strip()

# === Tokenize ve pad işlemi ===
sequence = tokenizer.texts_to_sequences([text])
padded = pad_sequences(sequence, maxlen=max_length, padding=padding_type, truncating=trunc_type)

# === Tahmin yap ===
prediction = model.predict(padded)[0][0]

# === Sonucu yazdır ===
print("\n🔎 Tahmin Sonucu:")
if prediction > 0.5:
    print(f" Bu e-posta **Oltalama (Phishing)** olabilir. (Skor: {prediction:.4f})")
else:
    print(f" Bu e-posta **Güvenilir** görünüyor. (Skor: {prediction:.4f})")
