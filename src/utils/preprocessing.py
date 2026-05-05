import re
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

class TextPreprocessor:
    def __init__(self):
        # Initialize Sastrawi components
        self.stemmer = StemmerFactory().create_stemmer()
        self.stopword_remover = StopWordRemoverFactory().create_stop_word_remover()

    def clean_text(self, text: str) -> str:
        """
        Membersihkan teks dari karakter yang tidak diinginkan (HTML tags, URL, simbol).
        """
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', ' ', text)
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        # Remove multiple spaces
        text = re.sub(r'\s+', ' ', text)
        # Remove special characters (keep only alphanumeric and spaces)
        text = re.sub(r'[^a-zA-Z0-9\s.,!?]', '', text)
        return text.strip()

    def preprocess_for_nlp(self, text: str, do_stemming: bool = False, remove_stopwords: bool = True) -> str:
        """
        Preprocessing teks penuh untuk NLP task seperti klasifikasi atau semantic search.
        """
        text = self.clean_text(text)
        
        # Lowercasing
        text = text.lower()
        
        # Stopword removal (opsional tapi dianjurkan untuk task tertentu)
        if remove_stopwords:
            text = self.stopword_remover.remove(text)
            
        # Stemming (opsional, terkadang menurunkan performa semantic search/summarization)
        if do_stemming:
            text = self.stemmer.stem(text)
            
        return text

# Example usage:
# preprocessor = TextPreprocessor()
# clean_txt = preprocessor.clean_text("Teks <b>kotor</b> dengan URL http://example.com !!!")
