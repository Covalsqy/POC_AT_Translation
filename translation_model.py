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

    @staticmethod
    def _is_header_line(line: str) -> bool:
        """Detect headers - lines ending with colon or very short labels or placeholders."""
        s = line.strip()
        if not s:
            return False
        return (s.endswith(':') or 
                (len(s) <= 50 and s[0].isupper()) or
                bool(re.match(r'^\[.*\]$', s)))

    @staticmethod
    def _is_bullet_line(line: str) -> bool:
        """Detect list items including Roman numerals."""
        return bool(re.match(r'^\s*([–\-•]|[IVX]+\s*[–\-])\s+', line))

    @staticmethod
    def _split_into_sentences(text: str) -> list[str]:
        """Split paragraph into sentences for better translation of long texts."""
        sentences = re.split(r'(?<=[.;])\s+(?=[A-ZÀÁÂÃÄÅÆÇÈÉÊËÌÍÎÏÐÑÒÓÔÕÖØÙÚÛÜÝ])', text)
        return [s.strip() for s in sentences if s.strip()]

    @staticmethod
    def _split_into_blocks(text: str):
        """Split text into blocks, keeping chunks smaller to reduce hallucinations."""
        lines = text.split('\n')
        blocks = []
        
        i = 0
        while i < len(lines):
            line = lines[i]
            
            if not line.strip():
                blocks.append({"type": "blank", "text": ""})
                i += 1
                continue
            
            if TranslationModel._is_header_line(line):
                blocks.append({"type": "header", "text": line.strip()})
                i += 1
                continue
            
            if TranslationModel._is_bullet_line(line):
                blocks.append({"type": "bullet", "text": line.strip()})
                i += 1
                continue
            
            # Regular paragraph
            para_lines = [line]
            i += 1
            while i < len(lines):
                next_line = lines[i]
                if (not next_line.strip() or 
                    TranslationModel._is_header_line(next_line) or
                    TranslationModel._is_bullet_line(next_line)):
                    break
                para_lines.append(next_line)
                i += 1
            
            full_para = " ".join(para_lines).strip()
            
            # Split long paragraphs into smaller chunks (reduced from 400 to 250)
            if len(full_para) > 250:
                sentences = TranslationModel._split_into_sentences(full_para)
                current_group = []
                current_length = 0
                
                for sentence in sentences:
                    sentence_len = len(sentence)
                    # Reduced from 350 to 200 chars per group
                    if current_length + sentence_len > 200 and current_group:
                        blocks.append({"type": "paragraph", "text": " ".join(current_group)})
                        current_group = [sentence]
                        current_length = sentence_len
                    else:
                        current_group.append(sentence)
                        current_length += sentence_len + 1
                
                if current_group:
                    blocks.append({"type": "paragraph", "text": " ".join(current_group)})
            else:
                blocks.append({"type": "paragraph", "text": full_para})
        
        return blocks

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
        """Block-level translation with smaller chunks to reduce hallucinations."""
        if not text:
            return text

        src = self._normalize_language_code(source_lang)
        tgt = self._normalize_language_code(target_lang)
        if src == tgt:
            return text

        self.tokenizer.src_lang = src
        tgt_lang_id = self.tokenizer.get_lang_id(tgt)

        blocks = self._split_into_blocks(text)
        non_blank_blocks = [b for b in blocks if b["type"] != "blank"]
        
        self.progress["total_batches"] = len(non_blank_blocks)
        self.progress["current_batch"] = 0
        print(f"Translating {len(non_blank_blocks)} blocks...")

        result_lines = []

        for idx, block in enumerate(blocks):
            block_type = block["type"]
            block_text = block["text"]

            if block_type == "blank":
                result_lines.append("")
                continue

            self.progress["current_text"] = block_text[:80] + ("..." if len(block_text) > 80 else "")

            # Conservative max_length - use 768 as default
            translated = self._translate_batch([block_text], src, tgt_lang_id, max_len=768)[0]
            
            if block_type == "header":
                result_lines.append(translated.strip())
            elif block_type == "bullet":
                result_lines.append(translated.strip())
            else:  # paragraph
                wrapped = self._wrap_text(translated.strip(), width=80)
                result_lines.append(wrapped)
            
            self.progress["current_batch"] += 1
            print(f"Block {self.progress['current_batch']}/{self.progress['total_batches']} done")

        return "\n".join(result_lines)

    @staticmethod
    def _wrap_text(text: str, width: int = 80) -> str:
        """Wrap text to specified width, preserving whole words."""
        if not text:
            return ""
        
        words = text.split()
        lines = []
        current_line = []
        current_length = 0
        
        for word in words:
            word_length = len(word)
            needed_length = word_length + (1 if current_line else 0)
            
            if current_length + needed_length <= width:
                current_line.append(word)
                current_length += needed_length
            else:
                if current_line:
                    lines.append(" ".join(current_line))
                current_line = [word]
                current_length = word_length
        
        if current_line:
            lines.append(" ".join(current_line))
        
        return "\n".join(lines)