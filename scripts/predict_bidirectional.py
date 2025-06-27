import tensorflow as tf
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Model ve tokenizer dosya yolları
model_path = 'C:/Users/sudee/OneDrive/phishing_project/models/backup/phishing_model_bilstm_v2.keras'
tokenizer_path = 'C:/Users/sudee/OneDrive/phishing_project/models/backup/tokenizer_bilstm.pkl'

# Modeli yükle
model = tf.keras.models.load_model(model_path)

# Tokenizer'ı yükle
with open(tokenizer_path, 'rb') as f:
    tokenizer = pickle.load(f)

# Modelin eğitiminde kullandığın max_length ve padding/truncating tipleri
max_length = 250
padding_type = 'post'
trunc_type = 'post'

def predict_email(konu, gönderen, içerik):
    full_text = f"{konu} {gönderen} {içerik}"
    seq = tokenizer.texts_to_sequences([full_text])
    padded = pad_sequences(seq, maxlen=max_length, padding=padding_type, truncating=trunc_type)
    pred_prob = model.predict(padded)[0][0]
    label = 1 if pred_prob > 0.5 else 0
    return label, pred_prob

# Kullanıcıdan input alma
print("Phishing Email Tahmin Aracı\n")

konu = input("Konu: ")
gönderen = input("Gönderen: ")
içerik = input("İçerik: ")

label, prob = predict_email(konu, gönderen, içerik)

print("\nTahmin Sonucu:")
print(f"Sınıf: {label} ({'Oltalama' if label == 1 else 'Güvenilir'})")
print(f"Olasılık Skoru: {prob:.4f}")


