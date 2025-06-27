import pandas as pd
import numpy as np
import pickle
import tensorflow as tf
from tensorflow.keras import layers, regularizers
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

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

# === Tokenizer ve metin dönüşümü ===
tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_token)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)
padded = pad_sequences(sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)
padded = np.array(padded)
labels = np.array(df['etiket'])

# === Eğitim ve doğrulama seti ===
X_train, X_val, y_train, y_val = train_test_split(padded, labels, test_size=0.2, random_state=42)

# Tokenizer'ı kaydet
save_dir = 'C:/Users/sudee/OneDrive/phishing_project/models/backup'
with open(f'{save_dir}/tokenizer_cnn_lstm.pkl', 'wb') as f:
    pickle.dump(tokenizer, f)

# === Model tanımı ===
def create_cnn_lstm_model(vocab_size, embedding_dim, max_length):
    model = tf.keras.Sequential([
        layers.Embedding(vocab_size, embedding_dim),
        layers.Conv1D(128, 5, activation='relu'),
        layers.MaxPooling1D(2),
        layers.Dropout(0.8),  # Dropout oranını artırdık
        layers.Bidirectional(layers.LSTM(32, dropout=0.5, recurrent_dropout=0.5, kernel_regularizer=regularizers.l2(0.02))),
        layers.Dense(32, activation='relu', kernel_regularizer=regularizers.l2(0.02)),
        layers.Dense(1, activation='sigmoid')
    ])
    model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])  # Öğrenme oranını artırdık
    return model

# Modeli oluştur
model = create_cnn_lstm_model(vocab_size, embedding_dim, max_length)

# === EarlyStopping ===
early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# === Eğit ===
history = model.fit(
    X_train, y_train,
    epochs=15,  # Eğitim süresi 15 epoch olacak
    batch_size=128,
    validation_data=(X_val, y_val),
    callbacks=[early_stop]
)

# === Modeli kaydet ===
model.save(f'{save_dir}/phishing_model_cnn_lstm_v2.keras')
print("Model başarıyla kaydedildi -> models/backup/")


# === Modelin validasyon seti üzerindeki tahminleri ===
y_pred_prob = model.predict(X_val)
y_pred = (y_pred_prob > 0.5).astype(int).reshape(-1)

# === Confusion Matrix ===
cm = confusion_matrix(y_val, y_pred)
print("\nConfusion Matrix:")
print(cm)

# Confusion Matrix'i tablo şeklinde göster
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Güvenilir (0)', 'Oltalama (1)'],
            yticklabels=['Güvenilir (0)', 'Oltalama (1)'])
plt.xlabel('Tahmin')
plt.ylabel('Gerçek')
plt.title('Confusion Matrix')
plt.show()

# === Classification Report ===
print("\nClassification Report:")
print(classification_report(y_val, y_pred, target_names=['Güvenilir', 'Oltalama']))

# === Doğruluk ===
val_accuracy = np.mean(y_pred == y_val)
print(f'\nValidation Accuracy: {val_accuracy:.4f}')


# === Ortalama doğruluk ===
avg_train_accuracy = np.mean(history.history['accuracy'])
avg_val_accuracy = np.mean(history.history['val_accuracy'])

print(f'\nEğitim Ortalama Doğruluğu: {avg_train_accuracy:.4f}')
print(f'Doğrulama Ortalama Doğruluğu: {avg_val_accuracy:.4f}')

# === Eğitim grafikleri ===
plt.figure(figsize=(12, 6))

# Doğruluk grafiği
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Eğitim Doğruluğu')
plt.plot(history.history['val_accuracy'], label='Doğrulama Doğruluğu')
plt.title('Doğruluk (Accuracy)')
plt.xlabel('Epoch')
plt.ylabel('Doğruluk')
plt.legend()

# Kayıp grafiği
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Eğitim Kaybı')
plt.plot(history.history['val_loss'], label='Doğrulama Kaybı')
plt.title('Kayıp (Loss)')
plt.xlabel('Epoch')
plt.ylabel('Kayıp')
plt.legend()

plt.show()



