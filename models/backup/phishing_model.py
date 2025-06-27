import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils import class_weight

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
if df['etiket'].isnull().any():
    print("Bilinmeyen kategori bulundu! Şunlar hatalı:", df[df['etiket'].isnull()]['Kategori'].unique())
    exit()
labels = df['etiket'].astype(int).tolist()

# Sınıf dağılımını yazdır
print("\nSınıf Dağılımı:\n", df['etiket'].value_counts())

# === Tokenizer ===
tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_token)
tokenizer.fit_on_texts(texts)

# Tokenize ve pad et
sequences = tokenizer.texts_to_sequences(texts)
padded = pad_sequences(sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)
labels = np.array(labels)

# === Train-test ayır ===
X_train, X_test, y_train, y_test = train_test_split(padded, labels, test_size=0.2, random_state=42, stratify=labels)

# === Class weights hesapla ===
class_weights = class_weight.compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
class_weights = dict(enumerate(class_weights))
print("\nClass Weights:", class_weights)

# === Modeli oluştur ===
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
    tf.keras.layers.LSTM(64),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# === Callback'ler ===
early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True)
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2, min_lr=1e-5)

# === Eğit ===
history = model.fit(
    X_train, y_train,
    epochs=15,
    batch_size=32,
    validation_split=0.2,
    callbacks=[early_stop, reduce_lr],
    class_weight=class_weights
)

# === Test setinde değerlendir ===
y_pred_prob = model.predict(X_test)
y_pred = (y_pred_prob > 0.5).astype(int)

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n")
print(confusion_matrix(y_test, y_pred))

# === Kaydet ===
save_dir = 'C:/Users/sudee/OneDrive/phishing_project/models/backup'
os.makedirs(save_dir, exist_ok=True)

# Tokenizer'ı kaydet
with open(f'{save_dir}/tokenizer_v1.pkl', 'wb') as f:
    pickle.dump(tokenizer, f)

# Modeli kaydet
model.save(f'{save_dir}/phishing_model_lstm_v1.keras')

print("\nModel ve tokenizer başarıyla kaydedildi -> models/backup/")

# === Eğitim grafiği ===
plt.plot(history.history['accuracy'], label='train accuracy')
plt.plot(history.history['val_accuracy'], label='val accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Eğitim ve Doğrulama Doğruluğu')
plt.legend()
plt.show()
