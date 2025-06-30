# Phishing Email Detection with Deep Learning 🧠📧

Bu proje, oltalama (phishing) e-postalarını tespit etmek amacıyla geliştirilmiş bir makine öğrenmesi uygulamasıdır. LSTM, GRU, CNN-LSTM ve Bidirectional LSTM gibi derin öğrenme modelleri karşılaştırılmıştır.

## 📁 Klasörler
- `data/`: Veri seti
- `models/`: Eğitilmiş modeller
- `scripts/`: Eğitim ve tahmin scriptleri

## 📊 Kullanılan Teknolojiler
- Python, TensorFlow, Keras
- NLP, Embedding, LSTM

## 📌 Amaç
Gerçek ve oltalama e-postaları ayırt edebilen bir sistem geliştirmek.

⚠️ Not:
Bu proje geliştirme sürecinde yerel bir ortamda (örneğin Windows işletim sistemi altında Jupyter Notebook ya da Python script dosyaları ile) çalıştırılmıştır. Dosya yolları (örn. veri seti, model dosyaları, ekran görüntüleri vb.) tam yol (absolute path) şeklinde yazılmıştır:

Örnek:
C:/Users/kullanici_adi/Desktop/phishing_project/dataset.csv

Bu nedenle, projeyi farklı bir bilgisayarda çalıştırmak isteyen kullanıcıların:

Dosya yollarını kendi bilgisayarlarının dizin yapısına göre güncellemesi,

Tercihen dosya yollarını göreli (relative path) yaparak taşınabilirliği artırması gerekmektedir.

Önerilen çözüm:

python
Kopyala
Düzenle
import os
base_path = os.path.dirname(__file__)
data_path = os.path.join(base_path, "dataset", "emails.csv")
Bu yöntemle kod, farklı ortamlarda da çalışabilir hale gelecektir.


⚠️ Note:
This project was developed and tested in a local environment (e.g., Windows OS using Jupyter Notebook or Python scripts).
Therefore, many file paths (such as dataset, model files, screenshots, etc.) are written using absolute paths like:

Example:
C:/Users/username/Desktop/phishing_project/dataset.csv

As a result, running this project on a different machine may cause file path errors unless these paths are updated.

🛠 How to Fix:
Update the file paths according to your own directory structure.

Alternatively, replace absolute paths with relative paths to make the project portable.

✅ Recommended approach:

python
Kopyala
Düzenle
import os
base_path = os.path.dirname(__file__)
data_path = os.path.join(base_path, "dataset", "emails.csv")
Using this method, the code will be more flexible and compatible across different systems.

