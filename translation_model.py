import re
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

class TranslationModel:
    """M2M100 1.2B translation with natural formatting."""

    LANGUAGE_CODES = {
        'portuguese': 'pt', 'pt': 'pt',
        'english': 'en', 'en': 'en',
        'spanish': 'es', 'es': 'es',
        'french': 'fr', 'fr': 'fr',
        'german': 'de', 'de': 'de',
        'chinese': 'zh', 'zh': 'zh',
        'arabic': 'ar', 'ar': 'ar',
        'russian': 'ru', 'ru': 'ru',
        'japanese': 'ja', 'ja': 'ja',
        'korean': 'ko', 'ko': 'ko',
    }

    def __init__(self, model_name: str = "facebook/m2m100_1.2B", progress_callback=None):
        self.model_name = model_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading M2M100 model '{model_name}'... (using {self.device})")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(self.device)
        self.model.eval()
        print("Model loaded.")
        
        if progress_callback:
            self.progress = progress_callback()
        else:
            self.progress = {"current_batch": 0, "total_batches": 0, "current_text": ""}

    @staticmethod
    def _normalize_language_code(lang: str) -> str:
        key = (lang or "").lower().strip()
        if key in TranslationModel.LANGUAGE_CODES:
            return TranslationModel.LANGUAGE_CODES[key]
        raise ValueError(f"Unsupported language: {lang}")

    # Formatting detection removed to prioritize translation quality over output formatting

    def _chunk_by_tokens(self, text: str, src: str, max_tokens: int = 250) -> list[str]:
        """Split text into chunks at natural boundaries, targeting ~250 tokens per chunk.
        
        Chunking priority (never splits mid-word):
        1. Paragraph boundaries (\n\n)
        2. Sentence boundaries (. ! ?)
        3. Phrase boundaries (, ; :)
        4. Whitespace (as last resort)
        
        Ensures every chunk starts/ends at clean boundaries for better translation quality.
        """
        if not text:
            return []
        
        # Quick check: if entire text fits, return as-is
        self.tokenizer.src_lang = src
        test_tokens = self.tokenizer(text, return_tensors="pt", truncation=False)
        if test_tokens['input_ids'].shape[1] <= max_tokens:
            return [text]
        
        # Split by paragraphs first (best natural boundary)
        paragraphs = text.split('\n\n')
        chunks = []
        current_chunk = []
        current_tokens = 0
        
        for para in paragraphs:
            if not para.strip():
                continue
                
            # Tokenize paragraph to get actual token count
            para_tokens = self.tokenizer(para, return_tensors="pt", truncation=False)
            para_token_count = para_tokens['input_ids'].shape[1]
            
            # If single paragraph exceeds limit, split at sentence boundaries
            if para_token_count > max_tokens:
                # Save current chunk if any
                if current_chunk:
                    chunks.append('\n\n'.join(current_chunk))
                    current_chunk = []
                    current_tokens = 0
                
                # Split by sentences (. ! ?) - but not abbreviations like "Art." or "Dr."
                # Look for sentence-ending punctuation followed by space and capital letter
                sentences = re.split(r'(?<=[.!?])\s+(?=[A-ZÁÉÍÓÚÀÂÊÔÃÕÇ])', para)
                if len(sentences) == 1:
                    # Fallback: split on any . ! ? if no capital letters detected
                    sentences = re.split(r'(?<=[.!?])\s+', para)
                for sentence in sentences:
                    if not sentence.strip():
                        continue
                    
                    sent_tokens = self.tokenizer(sentence, return_tensors="pt", truncation=False)
                    sent_token_count = sent_tokens['input_ids'].shape[1]
                    
                    # If single sentence exceeds limit, split at phrase boundaries
                    if sent_token_count > max_tokens:
                        # Save current chunk if any
                        if current_chunk:
                            chunks.append(' '.join(current_chunk))
                            current_chunk = []
                            current_tokens = 0
                        
                        # Split by phrases (, ; :)
                        phrases = re.split(r'(?<=[,;:])\s+', sentence)
                        for phrase in phrases:
                            if not phrase.strip():
                                continue
                            
                            phrase_tokens = self.tokenizer(phrase, return_tensors="pt", truncation=False)
                            phrase_token_count = phrase_tokens['input_ids'].shape[1]
                            
                            # If single phrase still exceeds limit, split at whitespace (last resort)
                            if phrase_token_count > max_tokens:
                                # Save current chunk if any
                                if current_chunk:
                                    chunks.append(' '.join(current_chunk))
                                    current_chunk = []
                                    current_tokens = 0
                                
                                # Split by words (whitespace boundary - never mid-word!)
                                words = phrase.split()
                                for word in words:
                                    word_tokens = self.tokenizer(word, return_tensors="pt", truncation=False)
                                    word_token_count = word_tokens['input_ids'].shape[1]
                                    
                                    if current_tokens + word_token_count > max_tokens:
                                        if current_chunk:
                                            chunks.append(' '.join(current_chunk))
                                        current_chunk = [word]
                                        current_tokens = word_token_count
                                    else:
                                        current_chunk.append(word)
                                        current_tokens += word_token_count
                            
                            # Phrase fits, try to add to current chunk
                            elif current_tokens + phrase_token_count > max_tokens:
                                if current_chunk:
                                    chunks.append(' '.join(current_chunk))
                                current_chunk = [phrase]
                                current_tokens = phrase_token_count
                            else:
                                current_chunk.append(phrase)
                                current_tokens += phrase_token_count
                    
                    # Sentence fits, try to add to current chunk
                    elif current_tokens + sent_token_count > max_tokens:
                        if current_chunk:
                            chunks.append(' '.join(current_chunk))
                        current_chunk = [sentence]
                        current_tokens = sent_token_count
                    else:
                        current_chunk.append(sentence)
                        current_tokens += sent_token_count
            
            # Paragraph fits within limit, try to add to current chunk
            elif current_tokens + para_token_count > max_tokens:
                # Current chunk would overflow, save it and start new
                if current_chunk:
                    chunks.append('\n\n'.join(current_chunk))
                current_chunk = [para]
                current_tokens = para_token_count
            else:
                # Add paragraph to current chunk
                current_chunk.append(para)
                current_tokens += para_token_count
        
        # Don't forget last chunk
        if current_chunk:
            chunks.append('\n\n'.join(current_chunk) if len(current_chunk) > 1 else ' '.join(current_chunk))
        
        return chunks if chunks else [text]



    def _translate_batch(self, texts: list[str], src: str, tgt_lang_id: int, max_len: int = 768) -> list[str]:
        """Translate a batch with conservative settings."""
        if not texts:
            return []
        
        self.tokenizer.src_lang = src
        inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_len
        ).to(self.device)

        with torch.no_grad():
            gen = self.model.generate(
                **inputs,
                forced_bos_token_id=tgt_lang_id,
                max_length=max_len,
                num_beams=5,
                length_penalty=1.0,  # Neutral to avoid cutting or padding
                no_repeat_ngram_size=3,
                early_stopping=True,
            )

        return self.tokenizer.batch_decode(gen, skip_special_tokens=True)

    def translate(self, text: str, source_lang: str, target_lang: str) -> str:
        """Translate text using token-aware chunking to maximize context per chunk.
        
        Priority: Translation quality over output formatting.
        Uses up to 700 tokens (~3500 chars) per chunk for maximum context.
        """
        if not text:
            return text

        src = self._normalize_language_code(source_lang)
        tgt = self._normalize_language_code(target_lang)
        if src == tgt:
            return text

        self.tokenizer.src_lang = src
        tgt_lang_id = self.tokenizer.get_lang_id(tgt)

        # Split text into optimal-size chunks that fit within token limits
        chunks = self._chunk_by_tokens(text, src, max_tokens=250)
        
        self.progress["total_batches"] = len(chunks)
        self.progress["current_batch"] = 0
        print(f"Translating {len(chunks)} chunks (max context per chunk)...")

        results = []

        for idx, chunk in enumerate(chunks):
            self.progress["current_text"] = chunk[:80] + ("..." if len(chunk) > 80 else "")

            # Translate with full token budget
            translated = self._translate_batch([chunk], src, tgt_lang_id, max_len=768)[0]
            results.append(translated)
            
            self.progress["current_batch"] += 1
            print(f"Chunk {self.progress['current_batch']}/{self.progress['total_batches']} done")

        # Join with double newline to preserve paragraph separation
        return "\n\n".join(results)

