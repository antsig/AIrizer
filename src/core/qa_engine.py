import requests
import json
import torch
from transformers import AutoTokenizer, AutoModelForQuestionAnswering

class DocumentQA:
    """
    Extractive Question Answering Engine for Indonesian Documents.
    Using IndoBERT-QA model to extract the exact answer span from the context.
    """
    def __init__(self, model_name="Rifky/Indobert-QA"):
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForQuestionAnswering.from_pretrained(model_name)
            self.is_ready = True
        except Exception as e:
            print(f"Error loading QA model: {e}")
            self.is_ready = False

    def answer_question(self, question: str, context: str) -> str:
        """
        Answers a question based on the provided document context.
        """
        if not self.is_ready:
            return "Sistem QA (Tanya Jawab) belum diinisialisasi dengan benar. Terjadi kesalahan saat memuat model."
        
        if not question.strip() or not context.strip():
            return "Pertanyaan atau dokumen tidak boleh kosong."
            
        try:
            inputs = self.tokenizer(question, context, return_tensors="pt", max_length=512, truncation=True)
            with torch.no_grad():
                outputs = self.model(**inputs)
            
            answer_start_index = outputs.start_logits.argmax()
            answer_end_index = outputs.end_logits.argmax()
            
            # Confidence check fallback (we can use logits max values as a proxy, but for simplicity we just return the span)
            # If start index > end index, it means it couldn't find a valid answer span
            if answer_start_index > answer_end_index:
                return "Maaf, saya tidak dapat menemukan informasi yang secara spesifik menjawab pertanyaan tersebut di dalam dokumen ini."
                
            predict_answer_tokens = inputs.input_ids[0, answer_start_index : answer_end_index + 1]
            answer = self.tokenizer.decode(predict_answer_tokens, skip_special_tokens=True)
            
            if not answer.strip() or answer == "[CLS]" or answer == "[SEP]":
                return "Maaf, jawaban tidak dapat dipastikan dari konteks teks."
                
            return answer
            
        except Exception as e:
            print(f"Error during QA inference: {e}")
            return "Terjadi kesalahan saat mencari jawaban."

    def answer_question_generative(self, question: str, context: str, api_key: str, provider: str = "openai") -> str:
        """
        Answers a question using an external Generative LLM API (OpenAI or HuggingFace).
        """
        if not api_key:
            return "API Key tidak ditemukan. Silakan tambahkan API Key Anda di file .env"
            
        if not question.strip() or not context.strip():
            return "Pertanyaan atau dokumen tidak boleh kosong."

        try:
            if provider.lower() == "openai":
                url = "https://api.openai.com/v1/chat/completions"
                headers = {
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {api_key}"
                }
                
                # Truncate context to avoid token limits (very basic truncation)
                truncated_context = context[:10000] 
                
                prompt = f"Berdasarkan dokumen berikut, jawablah pertanyaan pengguna dengan jelas dan ringkas. Jika jawaban tidak ada di dalam dokumen, katakan 'Informasi tidak tersedia di dalam dokumen'.\n\nDokumen:\n{truncated_context}"
                
                data = {
                    "model": "gpt-3.5-turbo",
                    "messages": [
                        {"role": "system", "content": prompt},
                        {"role": "user", "content": question}
                    ],
                    "temperature": 0.3
                }
                response = requests.post(url, headers=headers, json=data)
                
                if response.status_code == 200:
                    return response.json()['choices'][0]['message']['content']
                else:
                    return f"Error API OpenAI: {response.status_code} - {response.text}"
            else:
                return f"Provider {provider} belum didukung."
                
        except Exception as e:
            print(f"Error during Generative API call: {e}")
            return "Terjadi kesalahan saat menghubungi API eksternal."
