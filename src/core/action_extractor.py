import re
from typing import List, Dict

class ActionItemExtractor:
    def __init__(self):
        """
        Inisialisasi Action Item Extractor.
        Untuk baseline, menggunakan pendekatan berbasis aturan (Rule-based / Regex)
        yang difokuskan pada pola kalimat bahasa Indonesia untuk mengidentifikasi
        tugas, PIC (Person in Charge), dan deadline.
        
        Dalam versi Advanced, modul ini sebaiknya diganti dengan panggilan ke LLM 
        (OpenAI/Gemini) menggunakan prompt spesifik.
        """
        # Pola-pola umum yang mengindikasikan action item
        self.task_indicators = [
            r"harus\s+([a-zA-Z0-9\s]+)",
            r"bertanggung\s+jawab\s+(?:untuk|pada)?\s*([a-zA-Z0-9\s]+)",
            r"ditugaskan\s+(?:untuk)?\s*([a-zA-Z0-9\s]+)",
            r"wajib\s+([a-zA-Z0-9\s]+)"
        ]
        
        # Pola untuk mencari entitas penanggung jawab (huruf kapital di awal kalimat atau sebelum indikator)
        # Sederhananya, mencari Kata Ganti atau Nama Orang di sekitar kata tugas
        self.pic_pattern = r"([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)"
        
        # Pola untuk mencari deadline
        self.deadline_indicators = [
            r"paling\s+lambat\s+([a-zA-Z0-9\s]+)",
            r"sebelum\s+(?:hari|tanggal)?\s*([a-zA-Z0-9\s]+)",
            r"maksimal\s+(?:tanggal)?\s*([a-zA-Z0-9\s]+)"
        ]

    def extract_action_items(self, text: str) -> List[str]:
        """
        Mengekstrak action items dari teks.
        Format output yang diharapkan: "[PIC] -> [Tugas] -> [Deadline]"
        """
        if not text:
            return []
            
        action_items = []
        
        # Split teks menjadi kalimat-kalimat
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
                
            task = None
            pic = "Unknown"
            deadline = "No Deadline"
            
            # 1. Cari indikator tugas
            for pattern in self.task_indicators:
                match = re.search(pattern, sentence, re.IGNORECASE)
                if match:
                    # Ambil teks setelah indikator tugas sampai koma atau titik
                    task_raw = match.group(1).split(',')[0].split('.')[0].strip()
                    # Buang kata "sebelum", "paling lambat", dll dari task
                    task = re.split(r'\b(sebelum|paling lambat|maksimal)\b', task_raw, flags=re.IGNORECASE)[0].strip()
                    break
            
            # Jika ada tugas yang ditemukan
            if task:
                # 2. Cari PIC (siapa yang harus melakukan?)
                # Asumsi sederhana: PIC adalah subjek sebelum kata indikator, biasanya diawali huruf kapital
                words_before_task = sentence[:sentence.lower().find("harus") if "harus" in sentence.lower() else len(sentence)]
                pic_match = re.findall(self.pic_pattern, words_before_task)
                if pic_match:
                    pic = pic_match[-1] # Ambil entitas terakhir sebelum indikator tugas
                
                # 3. Cari Deadline
                for d_pattern in self.deadline_indicators:
                    d_match = re.search(d_pattern, sentence, re.IGNORECASE)
                    if d_match:
                        deadline = d_match.group(1).strip().split('.')[0].split(',')[0]
                        break
                
                # Format output
                action_items.append(f"{pic} -> {task} -> {deadline}")
                
        return action_items

# Example usage:
# extractor = ActionItemExtractor()
# items = extractor.extract_action_items("Budi harus menyusun laporan keuangan paling lambat Jumat depan.")
# print(items) # Budi -> menyusun laporan keuangan -> Jumat depan
