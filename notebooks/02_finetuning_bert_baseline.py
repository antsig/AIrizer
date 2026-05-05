# %% [markdown]
# # Template Fine-Tuning IndoBERT Baseline
# Jika Akurasi dari model Zero-Shot belum mencukupi untuk kebutuhan spesifik perusahaan,
# kita dapat melatih ulang (fine-tune) pre-trained model bahasa Indonesia (IndoBERT) 
# menggunakan dataset klasifikasi mandiri.

# %%
import os
import json
import torch
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import Trainer, TrainingArguments
from datasets import Dataset

# Pastikan menggunakan GPU jika tersedia
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(f"Menggunakan perangkat: {device}")

# %% [markdown]
# ## 1. Persiapan Data
# Misalkan kita memuat dataset dari synthetic, namun untuk fine-tuning idealnya
# kita membutuhkan minimal ratusan hingga ribuan data agar hasilnya stabil.
# %%
dataset_path = "../data/synthetic/dataset.json"

with open(dataset_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

df = pd.DataFrame(data)

# Buat mapping kategori ke ID angka (0, 1, 2, dst)
unique_labels = df['category'].unique().tolist()
label2id = {label: i for i, label in enumerate(unique_labels)}
id2label = {i: label for label, i in label2id.items()}

# Ubah target kolom
df['label'] = df['category'].map(label2id)

# Bagi dataset Train dan Test (80% / 20%)
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# Konversi DataFrame pandas menjadi HuggingFace Dataset
train_dataset = Dataset.from_pandas(train_df[['content', 'label']])
test_dataset = Dataset.from_pandas(test_df[['content', 'label']])

print(f"Jumlah Data Latih: {len(train_dataset)}")
print(f"Jumlah Data Uji: {len(test_dataset)}")

# %% [markdown]
# ## 2. Inisialisasi Tokenizer & Model
# Kita menggunakan IndoBERT dari IndoBenchmark
# %%
model_name = "indobenchmark/indobert-base-p1"

try:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, 
        num_labels=len(unique_labels),
        id2label=id2label,
        label2id=label2id
    ).to(device)
except Exception as e:
    print(f"Gagal memuat model. Pastikan koneksi internet stabil. Error: {e}")

# %% [markdown]
# ## 3. Fungsi Tokenisasi (Preprocessing)
# %%
def tokenize_function(examples):
    return tokenizer(examples['content'], padding="max_length", truncation=True, max_length=256)

tokenized_train = train_dataset.map(tokenize_function, batched=True)
tokenized_test = test_dataset.map(tokenize_function, batched=True)

# %% [markdown]
# ## 4. Konfigurasi Pelatihan (Training Arguments)
# %%
training_args = TrainingArguments(
    output_dir="../models/indobert_finetuned",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,  # Sesuaikan dengan VRAM GPU
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    save_strategy="epoch",
    logging_dir="../models/logs",
    load_best_model_at_end=True,
)

from sklearn.metrics import accuracy_score, f1_score

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average='weighted')
    return {"accuracy": acc, "f1": f1}

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_test,
    compute_metrics=compute_metrics
)

# %% [markdown]
# ## 5. Mulai Pelatihan
# (Uncomment baris di bawah untuk memulai proses pelatihan sungguhan)
# Peringatan: Sangat disarankan menjalankan proses ini di GPU/Google Colab.
# %%
# trainer.train()

# %% [markdown]
# ## 6. Simpan Model Terlatih
# %%
# trainer.save_model("../models/indobert_finetuned_final")
# tokenizer.save_pretrained("../models/indobert_finetuned_final")
# print("Model berhasil disimpan!")
