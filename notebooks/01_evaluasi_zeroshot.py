# %% [markdown]
# # Evaluasi Model Zero-Shot Classification
# Notebook ini berfungsi untuk mengevaluasi performa model `mDeBERTa-v3-base-mnli-xnli` 
# menggunakan dataset sintetik yang telah kita generate.

# %%
import os
import sys
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report

# Tambahkan root path agar bisa import dari src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.classifier import DocumentClassifier

# %% [markdown]
# ## 1. Load Dataset Sintetik
# %%
dataset_path = "../data/synthetic/dataset.json"

if not os.path.exists(dataset_path):
    print("Dataset sintetik belum ada. Jalankan python src/utils/data_generator.py terlebih dahulu.")
else:
    with open(dataset_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    df = pd.DataFrame(data)
    print(f"Total data: {len(df)}")
    print(df['category'].value_counts())

# %% [markdown]
# ## 2. Inisialisasi Model
# %%
classifier = DocumentClassifier(model_type="zero-shot")

# %% [markdown]
# ## 3. Prediksi Seluruh Data
# Proses ini akan memakan waktu sesuai spesifikasi perangkat keras (CPU/GPU).
# %%
y_true = []
y_pred = []

print("Memulai prediksi...")
for i, row in df.iterrows():
    text = row['content']
    true_label = row['category']
    
    # Lakukan prediksi
    pred_label = classifier.classify(text)
    
    y_true.append(true_label)
    y_pred.append(pred_label)
    
    if (i+1) % 10 == 0:
        print(f"Memproses {i+1}/{len(df)}")

print("Selesai!")

# %% [markdown]
# ## 4. Evaluasi Metrik
# Menghitung Accuracy dan F1-Score
# %%
acc = accuracy_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred, average='weighted')

print(f"Accuracy : {acc:.4f} ({acc*100:.2f}%)")
print(f"F1-Score : {f1:.4f} ({f1*100:.2f}%)")

print("\nDetail Classification Report:")
print(classification_report(y_true, y_pred))

# %% [markdown]
# ## 5. Visualisasi Confusion Matrix
# %%
cm = confusion_matrix(y_true, y_pred)
labels = sorted(list(set(y_true + y_pred)))

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
plt.title('Confusion Matrix - Zero-Shot Classification')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.show()

# %%
# Anda dapat menganalisis letak kesalahan model dari plot di atas.
