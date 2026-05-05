import os
from typing import List, Dict, Union
from transformers import pipeline
import json

class DocumentClassifier:
    def __init__(self, model_type: str = "zero-shot"):
        """
        Inisialisasi klasifikator dokumen.
        Untuk baseline capstone, kita gunakan Zero-Shot Classification berbasis model multilingual
        agar langsung bekerja tanpa perlu training pada tahap awal.
        Nantinya, kita bisa membuat model fine-tuned BERT atau BiLSTM di sini.
        """
        self.model_type = model_type
        self.categories = ["Laporan", "Surat Resmi", "Berita", "Invoice", "Pendidikan", "Keuangan", "IT"]
        
        if self.model_type == "zero-shot":
            # Using a multilingual zero-shot model that works well for Indonesian
            model_name = "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli"
            print(f"Loading zero-shot classification model: {model_name}...")
            try:
                self.classifier = pipeline("zero-shot-classification", model=model_name)
            except Exception as e:
                print(f"Error loading model: {e}")
                self.classifier = None
        else:
            # Placeholder for future fine-tuned BERT or BiLSTM
            print(f"Model type {model_type} selected. Ensure weights are available in models/ dir.")
            self.classifier = None

    def set_categories(self, categories: List[str]):
        """Mengubah daftar kategori dinamis untuk zero-shot classification"""
        self.categories = categories

    def classify(self, text: str, return_all_scores: bool = False) -> Union[str, Dict]:
        """
        Mengklasifikasikan dokumen ke dalam salah satu kategori.
        """
        if not text or len(text.strip()) < 10:
            return "Tidak Diketahui"
            
        if self.model_type == "zero-shot" and self.classifier:
            try:
                # Limit text length to prevent memory/token errors
                chunk = text[:1500] 
                
                result = self.classifier(chunk, self.categories)
                
                if return_all_scores:
                    return {
                        "labels": result["labels"],
                        "scores": result["scores"]
                    }
                else:
                    # Return the top predicted label
                    return result["labels"][0]
            except Exception as e:
                print(f"Classification error: {e}")
                return "Error"
        else:
            return "Model not initialized"

# Example usage:
# classifier = DocumentClassifier()
# label = classifier.classify("Tagihan pembelian server bulan Maret Rp 10.000.000")
# print(label) # Likely "Invoice"
