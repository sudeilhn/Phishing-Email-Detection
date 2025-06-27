import pandas as pd
import numpy as np
import pickle
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

# === Hiperparametreler ===
vocab_size = 5000
embedding_dim = 100
max_length = 250
oov_token = "<OOV>"
padding_type = 'post'
trunc_type = 'post'
batch_size = 32
epochs = 15
learning_rate = 0.0001  # Düşük öğrenme oranı

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
with open(f'{save_dir}/tokenizer_gru.pkl', 'wb') as f:
    pickle.dump(tokenizer, f)

# === Farklı GRU Modelini oluştur ===
def create_gru_model_v2(vocab_size, embedding_dim, max_length):
    model = tf.keras.Sequential([
        # Embedding katmanı
        layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
        
        # GRU katmanı
        layers.GRU(128, return_sequences=True),
        layers.Dropout(0.5),
        
        # GRU katmanı
        layers.GRU(64),
        layers.Dropout(0.5),
        
        # Çıkış katmanı
        layers.Dense(1, activation='sigmoid')
    ])
    
    # Modeli derle
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model

# Modeli oluştur
model_v2 = create_gru_model_v2(vocab_size, embedding_dim, max_length)

# === Modeli eğit ===
history_v2 = model_v2.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_val, y_val))

# === Modeli kaydet ===
model_v2.save(f'{save_dir}/phishing_model_gru_v2.keras')

print("Model başarıyla kaydedildi -> models/backup/")

# === Toplam ortalama doğruluğu hesapla ===
avg_train_accuracy_v2 = np.mean(history_v2.history['accuracy'])
avg_val_accuracy_v2 = np.mean(history_v2.history['val_accuracy'])

# === Doğrulama verisi için tahminleri al ===
y_pred_probs = model_v2.predict(X_val)
y_pred = (y_pred_probs > 0.5).astype(int)

# === Confusion Matrix hesapla ===
cm = confusion_matrix(y_val, y_pred)
labels = ['Güvenilir', 'Oltalama']

# === Confusion Matrix'i görselleştir ===
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
plt.xlabel('Tahmin')
plt.ylabel('Gerçek')
plt.title('Karışıklık Matrisi (Confusion Matrix)')
plt.show()

# === Sınıflandırma Raporu ===
print("\nSınıflandırma Raporu:\n")
print(classification_report(y_val, y_pred, target_names=labels))


print(f'\nEğitim Ortalama Doğruluğu: {avg_train_accuracy_v2:.4f}')
print(f'Doğrulama Ortalama Doğruluğu: {avg_val_accuracy_v2:.4f}')

# === Eğitimi görselleştir ===
# Accuracy ve Loss grafiklerini çiz
plt.figure(figsize=(12, 6))

# Accuracy grafiği
plt.subplot(1, 2, 1)
plt.plot(history_v2.history['accuracy'], label='Eğitim Doğruluğu')
plt.plot(history_v2.history['val_accuracy'], label='Doğrulama Doğruluğu')
plt.title('Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Doğruluk')
plt.legend()

# Loss grafiği
plt.subplot(1, 2, 2)
plt.plot(history_v2.history['loss'], label='Eğitim Kaybı')
plt.plot(history_v2.history['val_loss'], label='Doğrulama Kaybı')
plt.title('Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.show()
