import subprocess
import shutil
import re
import PyPDF2
import unicodedata

# try optional import
try:
    from pdfminer.high_level import extract_text as pdfminer_extract_text
    from pdfminer.layout import LAParams
except Exception:
    pdfminer_extract_text = None
    LAParams = None


class PDFDocumentManager:
    """Manages PDF document operations including text extraction."""

    @staticmethod
    def _normalize_whitespace(text: str) -> str:
        """Remove special characters and normalize whitespace."""
        text = text.replace('\ufeff', '')
        text = text.replace('\u200b', '')
        text = text.replace('\u00a0', ' ')
        text = re.sub(r'[ \t]{2,}', ' ', text)
        text = re.sub(r'\n{3,}', '\n\n', text)
        return text

    @staticmethod
    def clean_text_for_translation(text: str) -> str:
        """
        Conservative cleaning before sending text to a translation model:
        - Normalize Unicode (NFC)
        - Replace form-feed (page breaks) with newlines
        - Remove most control characters (but keep newline and tab)
        - Remove soft-hyphen and some invisible chars already handled elsewhere
        - Expand a small set of locale abbreviations (example for Portuguese)
        - Normalize whitespace but preserve newlines (keeps paragraph breaks)
        - Remove lines that are only page numbers (digits with optional whitespace)
        """
        if not text:
            return text

        # Unicode normalization
        text = unicodedata.normalize('NFC', text)

        # Replace page-break / form-feed with paragraph break
        text = text.replace('\f', '\n\n')

        # Remove soft-hyphen (often inserted by PDF hyphenation)
        text = text.replace('\u00ad', '')

        # Common invisible chars already observed
        text = text.replace('\ufeff', '')
        text = text.replace('\u200b', '')
        text = text.replace('\u00a0', ' ')

        # Remove other C0/C1 control chars except newline (0x0A) and tab (0x09) and carriage return (0x0D)
        text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]', '', text)

        # Normalize whitespace but preserve paragraphs
        text = re.sub(r'[ \t]{2,}', ' ', text)
        text = re.sub(r'\n{3,}', '\n\n', text)

        # Trim accidental leading/trailing whitespace
        return text.strip()

    @staticmethod
    def _extract_with_pdftotext(pdf_path: str) -> str | None:
        pdftotext_path = shutil.which("pdftotext")
        if not pdftotext_path:
            return None
        try:
            proc = subprocess.run([pdftotext_path, "-enc", "UTF-8", pdf_path, "-"],
                                  stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
            return proc.stdout.decode("utf-8", errors="replace")
        except Exception:
            return None

    @staticmethod
    def _extract_with_pdfminer(pdf_path: str) -> str | None:
        if pdfminer_extract_text is None:
            return None
        try:
            laparams = LAParams(char_margin=2.0, word_margin=0.1, line_margin=0.5)
            return pdfminer_extract_text(pdf_path, laparams=laparams)
        except Exception:
            return None

    @staticmethod
    def _extract_with_pypdf2(pdf_path: str) -> str:
        parts = []
        with open(pdf_path, 'rb') as f:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages:
                try:
                    pt = page.extract_text()
                except Exception:
                    pt = None
                parts.append(pt or "")
        return "\n\n".join(parts)

    @staticmethod
    def extract_text_from_pdf(pdf_path: str) -> str:
        """
        Extract text from PDF with multiple strategies:
        1) try pdftotext (system tool, most robust)
        2) try pdfminer.six if installed
        3) fallback to PyPDF2
        """
        # 1) try pdftotext (preferred)
        txt = PDFDocumentManager._extract_with_pdftotext(pdf_path)
        if txt:
            return PDFDocumentManager._normalize_whitespace(txt)

        # 2) try pdfminer.six
        txt = PDFDocumentManager._extract_with_pdfminer(pdf_path)
        if txt:
            return PDFDocumentManager._normalize_whitespace(txt)

        # 3) fallback to PyPDF2
        txt = PDFDocumentManager._extract_with_pypdf2(pdf_path)
        return PDFDocumentManager._normalize_whitespace(txt)

    @staticmethod
    def save_text_to_file(text: str, output_path: str) -> None:
        """Save extracted text to a .txt file."""
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(text)
        except Exception as e:
            raise Exception(f"Error saving text file: {str(e)}")