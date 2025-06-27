import pandas as pd
import numpy as np
import pickle
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# === Hiperparametreler ===
vocab_size = 5000
embedding_dim = 100
max_length = 250
oov_token = "<OOV>"
padding_type = 'post'
trunc_type = 'post'

# === Veri yolu ===
data_path = 'C:/Users/sudee/OneDrive/phishing_project/data/turkish_phishing_dataset.csv'

# === Veriyi oku ===
df = pd.read_csv(data_path)

# Metinleri birleştir
df['metin'] = df['Konu'].astype(str) + " " + df['Gönderen'].astype(str) + " " + df['İçerik'].astype(str)
texts = df['metin'].tolist()

# Etiketleri sayıya dönüştür
df['etiket'] = df['Kategori'].map({'Oltalama': 1, 'Güvenilir': 0})

# === Tokenizer ve örnek metin dönüşümü ===
tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_token)
tokenizer.fit_on_texts(texts)

# === Tüm veriyi tokenize et ve pad et ===
sequences = tokenizer.texts_to_sequences(texts)
padded = pad_sequences(sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)

# Veriyi numpy array'e dönüştür
padded = np.array(padded)
labels = np.array(df['etiket'])

# === Eğitim ve doğrulama verilerini ayır ===
X_train, X_val, y_train, y_val = train_test_split(padded, labels, test_size=0.2, random_state=42)

# Tokenizer'ı kaydet
save_dir = 'C:/Users/sudee/OneDrive/phishing_project/models/backup'
with open(f'{save_dir}/tokenizer_cnn_lstm.pkl', 'wb') as f:
    pickle.dump(tokenizer, f)

# === CNN-LSTM Modelini oluştur ===
def create_cnn_lstm_model(vocab_size, embedding_dim, max_length):
    model = tf.keras.Sequential([
        # Embedding katmanı
        layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
        
        # Convolutional katman
        layers.Conv1D(filters=128, kernel_size=5, activation='relu'),
        layers.MaxPooling1D(pool_size=2),
        
        # LSTM katmanı
        layers.LSTM(64),
        
        # Dropout katmanı
        layers.Dropout(0.5),
        
        # Çıkış katmanı
        layers.Dense(1, activation='sigmoid')
    ])
    
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

# Modeli oluştur
model = create_cnn_lstm_model(vocab_size, embedding_dim, max_length)

# === Modeli eğit ===
history = model.fit(X_train, y_train, epochs=15, batch_size=32, validation_data=(X_val, y_val))

# === Modeli kaydet ===
model.save(f'{save_dir}/phishing_model_cnn_lstm_v1.keras')

print("Model başarıyla kaydedildi -> models/backup/")

# === Toplam ortalama doğruluğu hesapla ===
avg_train_accuracy = np.mean(history.history['accuracy'])
avg_val_accuracy = np.mean(history.history['val_accuracy'])

print(f'\nEğitim Ortalama Doğruluğu: {avg_train_accuracy:.4f}')
print(f'Doğrulama Ortalama Doğruluğu: {avg_val_accuracy:.4f}')

# === Eğitimi görselleştir ===
# Accuracy ve Loss grafiklerini çiz
plt.figure(figsize=(12, 6))

# Accuracy grafiği
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Eğitim Doğruluğu')
plt.plot(history.history['val_accuracy'], label='Doğrulama Doğruluğu')
plt.title('Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Doğruluk')
plt.legend()

# Loss grafiği
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Eğitim Kaybı')
plt.plot(history.history['val_loss'], label='Doğrulama Kaybı')
plt.title('Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.show()

