import os
from typing import List
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, EncoderDecoderModel
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class DocumentSummarizer:
    def __init__(self):
        """
        Inisialisasi Document Summarizer menggunakan HuggingFace transformers secara langsung
        untuk menghindari isu pipeline 'Unknown task summarization' di versi transformers terbaru.
        """
        model_name = os.getenv("SUMMARIZATION_MODEL", "csebuetnlp/mT5_multilingual_XLSum")
        print(f"Loading summarization model: {model_name}...")
        
        try:
            # use_fast=False sering disarankan untuk model mT5 (SentencePiece)
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
            self.is_loaded = True
        except Exception as e:
            print(f"Error loading model {model_name}: {e}")
            self.is_loaded = False

    def summarize(self, text: str, max_length: int = 150, min_length: int = 40) -> str:
        """
        Menghasilkan ringkasan dari teks dokumen.
        """
        if not text or len(text.strip()) < 50:
            return text
            
        if not self.is_loaded:
            return "Error: Model peringkas tidak dapat dimuat."
            
        try:
            # Tokenize input
            input_ids = self.tokenizer.encode(text, return_tensors="pt", max_length=512, truncation=True)
            
            # Generate summary
            output_ids = self.model.generate(
                input_ids,
                max_length=max_length,
                min_length=min_length,
                no_repeat_ngram_size=3,
                num_beams=4,
                early_stopping=True
            )
            
            # Decode output
            summary = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
            return summary
            
        except Exception as e:
            print(f"Summarization error: {e}")
            return "Error generating summary."

    def generate_bullet_points(self, text: str, num_points: int = 3) -> List[str]:
        """
        Mengonversi teks menjadi bullet points.
        """
        summary = self.summarize(text, max_length=200, min_length=50)
        
        import re
        sentences = re.split(r'(?<=[.!?]) +', summary)
        
        points = [s.strip() for s in sentences if len(s.strip()) > 10][:num_points]
        
        if not points:
            points = [summary]
            
        return points
