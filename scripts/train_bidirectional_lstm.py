import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import layers, regularizers
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

# === Veri Yolu ===
data_path = 'C:/Users/sudee/OneDrive/phishing_project/data/turkish_phishing_dataset.csv'

# === Veriyi oku ===
df = pd.read_csv(data_path)
df['metin'] = df['Konu'].astype(str) + " " + df['Gönderen'].astype(str) + " " + df['İçerik'].astype(str)
texts = df['metin'].tolist()
df['etiket'] = df['Kategori'].map({'Oltalama': 1, 'Güvenilir': 0})

if df['etiket'].isnull().any():
    print("Bilinmeyen kategori bulundu!", df[df['etiket'].isnull()])
    exit()

labels = df['etiket'].astype(int).tolist()

# === Tokenizer ve Padding ===
tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_token)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)
padded = pad_sequences(sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)
labels = np.array(labels)

# === Train-Test Ayırma ===
X_train, X_test, y_train, y_test = train_test_split(padded, labels, test_size=0.2, random_state=42, stratify=labels)

# === Class Weights Hesapla ===
class_weights = class_weight.compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
class_weights = dict(enumerate(class_weights))

# === Bidirectional LSTM Modeli ===
def create_bidirectional_lstm_model(vocab_size, embedding_dim, max_length):
    model = tf.keras.Sequential([
        layers.Embedding(vocab_size, embedding_dim),
        layers.SpatialDropout1D(0.4),  # dropout biraz artırıldı

        layers.Bidirectional(layers.LSTM(
            16,
            return_sequences=True,
            dropout=0.5,              # dropout artırıldı
            recurrent_dropout=0.3,
            kernel_regularizer=regularizers.l2(0.01)  # L2 artırıldı
        )),
        layers.Bidirectional(layers.LSTM(
            8,
            dropout=0.5,             # dropout artırıldı
            recurrent_dropout=0.3,
            kernel_regularizer=regularizers.l2(0.01)  # L2 artırıldı
        )),
        
        layers.Dense(16, activation='relu', kernel_regularizer=regularizers.l2(0.01)),
        layers.Dropout(0.5),
        layers.Dense(1, activation='sigmoid')
    ])

    model.compile(
        loss='binary_crossentropy',
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),  # lr azaltıldı
        metrics=['accuracy']
    )
    return model


# === Modeli Oluştur ===
model = create_bidirectional_lstm_model(vocab_size, embedding_dim, max_length)

# === Callback'ler ===
early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=4, restore_best_weights=True)
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-6)


# === Modeli Eğit ===
history = model.fit(
    X_train, y_train,
    epochs=15,
    batch_size=32,
    validation_split=0.3,
    callbacks=[early_stop, reduce_lr],
    class_weight=class_weights,
    verbose=1
)

# === Test Performansı ===
y_pred_prob = model.predict(X_test)
y_pred = (y_pred_prob > 0.5).astype(int)

print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

# === Test Performansı ===
y_pred_prob = model.predict(X_test)
y_pred = (y_pred_prob > 0.5).astype(int)

print("\nClassification Report:\n", classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:\n", cm)

TN, FP, FN, TP = cm.ravel()

print(f"True Positive (TP): {TP}")
print(f"True Negative (TN): {TN}")
print(f"False Positive (FP): {FP}")
print(f"False Negative (FN): {FN}")

cm_df = pd.DataFrame(
    [[TN, FP],
     [FN, TP]],
    index=['Actual Negative', 'Actual Positive'],
    columns=['Predicted Negative', 'Predicted Positive']
)
print("\nConfusion Matrix (Tablo):")
print(cm_df)

# Test doğruluğunu hesapla ve yazdır
accuracy = (TP + TN) / (TP + TN + FP + FN)
print(f"\nTest Doğruluğu (Accuracy): {accuracy:.4f}")

# === Kaydet ===
save_dir = 'C:/Users/sudee/OneDrive/phishing_project/models/backup'
os.makedirs(save_dir, exist_ok=True)

with open(f'{save_dir}/tokenizer_bilstm.pkl', 'wb') as f:
    pickle.dump(tokenizer, f)

model.save(f'{save_dir}/phishing_model_bilstm_v2.keras')
print("Model ve tokenizer başarıyla kaydedildi!")

# === Grafik ===
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Doğruluk Grafiği')
plt.show()
