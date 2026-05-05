import os
from io import BytesIO
from typing import Union
import PyPDF2
import docx

class DocumentParser:
    """
    Kelas utilitas untuk mengekstrak teks dari berbagai format dokumen (PDF, DOCX, TXT).
    Mendukung baik input path file (lokal) maupun file-like objects (dari Streamlit upload).
    """
    
    @staticmethod
    def extract_text(file_input: Union[str, BytesIO], file_type: str = None) -> str:
        """
        Mengekstrak teks berdasarkan format file.
        :param file_input: Path (string) atau file-like object (BytesIO)
        :param file_type: Ekstensi file (misal: 'pdf', 'docx', 'txt'). Wajib jika input adalah BytesIO.
        :return: String teks dari dokumen.
        """
        # Tentukan tipe file jika input adalah string (path)
        if isinstance(file_input, str):
            if not os.path.exists(file_input):
                return ""
            file_type = file_input.split('.')[-1].lower()
            
        if not file_type:
            return ""

        file_type = file_type.lower()
        
        try:
            if file_type == 'pdf':
                return DocumentParser._parse_pdf(file_input)
            elif file_type in ['docx', 'doc']:
                return DocumentParser._parse_docx(file_input)
            elif file_type == 'txt':
                return DocumentParser._parse_txt(file_input)
            else:
                print(f"Format tidak didukung: {file_type}")
                return ""
        except Exception as e:
            print(f"Error saat membaca file {file_type}: {e}")
            return ""

    @staticmethod
    def _parse_pdf(file_input: Union[str, BytesIO]) -> str:
        text = ""
        is_path = isinstance(file_input, str)
        
        # Buka file dalam mode binary (jika path)
        f = open(file_input, 'rb') if is_path else file_input
        
        try:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages:
                extracted = page.extract_text()
                if extracted:
                    text += extracted + "\n"
        finally:
            if is_path:
                f.close()
            elif isinstance(file_input, BytesIO):
                file_input.seek(0) # Reset pointer
                
        return text.strip()

    @staticmethod
    def _parse_docx(file_input: Union[str, BytesIO]) -> str:
        # docx.Document bisa menerima path maupun file-like object
        doc = docx.Document(file_input)
        text = "\n".join([para.text for para in doc.paragraphs])
        
        # Reset pointer jika BytesIO
        if isinstance(file_input, BytesIO):
            file_input.seek(0)
            
        return text.strip()

    @staticmethod
    def _parse_txt(file_input: Union[str, BytesIO]) -> str:
        if isinstance(file_input, str):
            with open(file_input, 'r', encoding='utf-8', errors='ignore') as f:
                return f.read().strip()
        else:
            text = file_input.read().decode('utf-8', errors='ignore')
            file_input.seek(0)
            return text.strip()
