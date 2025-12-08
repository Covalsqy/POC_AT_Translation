import re
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

class TranslationModel:
    """NLLB-200 1.3B translation optimized for quality."""

    LANGUAGE_CODES = {
        'portuguese': 'por_Latn', 'pt': 'por_Latn',
        'english': 'eng_Latn', 'en': 'eng_Latn',
        'spanish': 'spa_Latn', 'es': 'spa_Latn',
        'french': 'fra_Latn', 'fr': 'fra_Latn',
        'german': 'deu_Latn', 'de': 'deu_Latn',
        'chinese': 'zho_Hans', 'zh': 'zho_Hans',
        'arabic': 'arb_Arab', 'ar': 'arb_Arab',
        'russian': 'rus_Cyrl', 'ru': 'rus_Cyrl',
        'japanese': 'jpn_Jpan', 'ja': 'jpn_Jpan',
        'korean': 'kor_Hang', 'ko': 'kor_Hang',
    }

    def __init__(self, model_name: str = "facebook/nllb-200-distilled-1.3B", progress_callback=None):
        self.model_name = model_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading NLLB-200 model '{model_name}'... (using {self.device})")
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

    def _chunk_by_tokens(self, text: str, src: str, max_tokens: int = 512) -> list[str]:
        """Split text into chunks of up to 512 tokens (NLLB training limit).
        
        Splits at word boundaries to avoid mid-word cuts.
        Priority: Maximize context for translation quality.
        """
        if not text:
            return []
        
        # Set source language for tokenizer
        self.tokenizer.src_lang = src
        
        # Quick check: if entire text fits, return as-is
        test_tokens = self.tokenizer(text, return_tensors="pt", truncation=False, add_special_tokens=True)
        if test_tokens['input_ids'].shape[1] <= max_tokens:
            return [text]
        
        # Split by whitespace to get words
        words = text.split()
        chunks = []
        current_chunk = []
        current_text = ""
        
        for word in words:
            # Try adding this word to current chunk
            test_text = current_text + (" " if current_text else "") + word
            test_tokens = self.tokenizer(test_text, return_tensors="pt", truncation=False, add_special_tokens=True)
            token_count = test_tokens['input_ids'].shape[1]
            
            if token_count <= max_tokens:
                # Word fits, add it
                current_chunk.append(word)
                current_text = test_text
            else:
                # Word doesn't fit, save current chunk and start new one
                if current_chunk:
                    chunks.append(" ".join(current_chunk))
                current_chunk = [word]
                current_text = word
        
        # Don't forget last chunk
        if current_chunk:
            chunks.append(" ".join(current_chunk))
        
        return chunks if chunks else [text]



    def _translate_batch(self, texts: list[str], src: str, tgt: str, max_input_len: int = 512, max_output_len: int = 1024, num_beams: int = 12) -> list[str]:
        """Translate a batch with NLLB-200 optimized settings.
        
        Args:
            max_input_len: Max tokens for input (for truncation check)
            max_output_len: Max total length for input+output combined during generation
            num_beams: Number of beams for beam search (higher = better quality, slower)
        """
        if not texts:
            return []
        
        self.tokenizer.src_lang = src
        
        # Detect actual input truncation by comparing with/without truncation
        for i, text in enumerate(texts):
            tokens_no_trunc = self.tokenizer(text, return_tensors="pt", truncation=False, add_special_tokens=True)
            tokens_with_trunc = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=max_input_len, add_special_tokens=True)
            
            actual_length = tokens_no_trunc['input_ids'].shape[1]
            truncated_length = tokens_with_trunc['input_ids'].shape[1]
            
            if actual_length != truncated_length:
                print(f"⚠️  TRUNCATION DETECTED: Input was truncated! Chunk {i+1}: {actual_length} tokens -> {truncated_length} tokens (LOST {actual_length - truncated_length} tokens)")
        
        inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=False,
            max_length=max_input_len
        ).to(self.device)

        with torch.no_grad():
            gen = self.model.generate(
                **inputs,
                forced_bos_token_id=self.tokenizer.convert_tokens_to_ids(tgt),
                max_length=max_output_len,
                num_beams=num_beams,
                length_penalty=1.0,  # Neutral - don't force longer or shorter
                no_repeat_ngram_size=3,
                early_stopping=True,  # Stop when all beams generate EOS (prevents hallucination)
                repetition_penalty=1.2,  # Discourage repetitive output
            )
        
        # Detect actual output truncation: check if EOS token is missing at the end
        # Note: max_length in generate() is input+output combined length
        eos_token_id = self.tokenizer.eos_token_id
        for i, output_ids in enumerate(gen):
            # For 1D tensor (single sequence)
            if output_ids.dim() == 1:
                output_length = output_ids.shape[0]
                last_token = output_ids[-1].item()
            else:  # Should not happen but handle it
                output_length = output_ids.shape[-1]
                last_token = output_ids[0, -1].item() if output_ids.shape[0] > 0 else None
            
            # If last token is NOT EOS, generation was cut off (incomplete translation)
            if last_token is not None and last_token != eos_token_id:
                print(f"⚠️  TRUNCATION DETECTED: Output was truncated! Chunk {i+1}: Generation stopped without EOS token (translation incomplete, {output_length} tokens generated)")

        return self.tokenizer.batch_decode(gen, skip_special_tokens=True)

    def translate(self, text: str, source_lang: str, target_lang: str, chunk_size: int = 400, num_beams: int = 12) -> str:
        """Translate text using NLLB-200 with optimized chunking strategy.
        
        Priority: Translation quality over output formatting.
        
        Args:
            text: Text to translate
            source_lang: Source language code
            target_lang: Target language code
            chunk_size: Max tokens per chunk (reduced from 512 to provide more context overlap)
            num_beams: Beam search width (12 recommended for quality, 8 for speed, 16 for max quality)
        
        Returns:
            Translated text
        """
        if not text:
            return text

        src = self._normalize_language_code(source_lang)
        tgt = self._normalize_language_code(target_lang)
        if src == tgt:
            return text

        self.tokenizer.src_lang = src

        # Split text into smaller chunks (400 tokens instead of 512)
        # This leaves more room for better translation without hitting limits
        chunks = self._chunk_by_tokens(text, src, max_tokens=chunk_size)
        
        self.progress["total_batches"] = len(chunks)
        self.progress["current_batch"] = 0
        print(f"Translating {len(chunks)} chunks ({chunk_size} tokens max per chunk, {num_beams} beams)...")

        results = []

        for idx, chunk in enumerate(chunks):
            self.progress["current_text"] = chunk[:80] + ("..." if len(chunk) > 80 else "")

            # Translate with increased beam search and higher output limit
            # max_output_len=1536 allows for expansion (translations can be longer than source)
            translated = self._translate_batch(
                [chunk], 
                src, 
                tgt, 
                max_input_len=chunk_size,
                max_output_len=1024,
                num_beams=num_beams
            )[0]
            results.append(translated)
            
            self.progress["current_batch"] += 1
            print(f"Chunk {self.progress['current_batch']}/{self.progress['total_batches']} done")

        # Join with space to maintain readability
        return " ".join(results)

