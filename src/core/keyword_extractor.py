from typing import List, Tuple
from keybert import KeyBERT

class KeywordExtractor:
    def __init__(self):
        """
        Inisialisasi Keyword Extractor menggunakan KeyBERT.
        Secara default menggunakan model multilingual yang ringan.
        """
        # "paraphrase-multilingual-MiniLM-L12-v2" is good for Indonesian
        model_name = "paraphrase-multilingual-MiniLM-L12-v2"
        print(f"Loading KeyBERT model: {model_name}...")
        try:
            self.kw_model = KeyBERT(model=model_name)
        except Exception as e:
            print(f"Error loading KeyBERT: {e}")
            self.kw_model = None

    def extract_keywords(self, text: str, top_n: int = 5) -> List[str]:
        """
        Mengekstrak kata kunci dari teks menggunakan KeyBERT.
        Mengembalikan daftar string (kata kunci).
        """
        if not text or not self.kw_model:
            return []
            
        try:
            # We can extract n-grams (1 to 2 words)
            keywords_with_scores = self.kw_model.extract_keywords(
                text, 
                keyphrase_ngram_range=(1, 2), 
                stop_words=None, 
                top_n=top_n
            )
            
            # KeyBERT returns a list of tuples: (keyword, score)
            # We just want the keywords sorted by score (descending)
            keywords = [kw[0] for kw in keywords_with_scores]
            return keywords
        except Exception as e:
            print(f"Keyword extraction error: {e}")
            return []

    def extract_with_scores(self, text: str, top_n: int = 5) -> List[Tuple[str, float]]:
        """
        Mengekstrak kata kunci beserta skor kepentingannya.
        """
        if not text or not self.kw_model:
            return []
            
        return self.kw_model.extract_keywords(
            text, 
            keyphrase_ngram_range=(1, 2), 
            stop_words=None, 
            top_n=top_n
        )

# Example usage:
# extractor = KeywordExtractor()
# keywords = extractor.extract_keywords("Pemerintah mengalokasikan dana pendidikan sebesar 20 triliun rupiah.")
# print(keywords)
