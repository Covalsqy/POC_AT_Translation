# PDF Translation Proof of Concept

A Flask-based web application for translating legal documents from PDFs using Meta's NLLB-200 (No Language Left Behind) multilingual translation model with automated quality assessment via Unbabel's COMET-QE model.

## ⚠️ This is a Proof of Concept

This is an experimental system not intended for production use. Key limitations:

- **Translation Quality**: Target accuracy is around 70% for legal/technical documents, though this has not yet been verified by professional translators
- **Document Size**: Limited to 512 tokens per chunk for both translation and quality estimation models - exceeding this results in truncation and significant quality degradation
- **Formatting**: Original document formatting (paragraphs, layout, tables) is not preserved in the output due to text cleaning required for optimal translation quality

## Features

- **PDF Text Extraction**: Multi-strategy extraction using pdftotext (preferred), pdfminer.six, or PyPDF2 as fallback
- **Multi-language Translation**: Supports 11+ languages via Meta's NLLB-200 1.3B distilled model
- **Quality Assessment**: Automated reference-free quality estimation using COMET-QE (0-100 scale with interpretations)
- **Performance Metrics**: Terminal output showing translation time, quality estimation time, and total processing time
- **Side-by-side Comparison**: Original and translated text displayed together for review
- **Text Export**: Download translated text as .txt file

## Supported Languages

Portuguese, English, Spanish, French, German, Italian, Chinese, Arabic, Russian, Japanese, Korean

## Installation

### Requirements

- Python 3.8+
- Linux (tested on Ubuntu)
- CUDA-capable GPU recommended (will fallback to CPU with slower performance)
- 6GB+ RAM for model loading

### Quick Setup (Recommended)

```bash
# Clone repository
git clone https://github.com/Covalsqy/POC_AT_Translation.git
cd POC_AT_Translation

# Run setup script (creates venv and installs dependencies)
chmod +x setup.sh
./setup.sh

# Activate virtual environment and run the application
source venv/bin/activate
./run.sh
```

### Manual Setup

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
sudo apt-get install poppler-utils

# Run application
python3 app.py
```

### HuggingFace Authentication (Required for COMET-QE)

The COMET-QE quality estimation model is gated and requires HuggingFace authentication:

1. Create a HuggingFace account at https://huggingface.co/join
2. Accept the model terms at https://huggingface.co/Unbabel/wmt22-cometkiwi-da
3. Generate an access token at https://huggingface.co/settings/tokens (read access is sufficient)
4. Login via terminal:
   ```bash
   source venv/bin/activate
   huggingface-cli login
   # Paste your access token when prompted
   ```

**Note**: Without authentication, the quality estimation feature will fail. The translation will still work.

## Usage

After setup, the application will be available at `http://127.0.0.1:5000`

1. Upload a PDF file
2. Select source and target languages
3. Wait for translation
4. View quality assessment score and interpretation
5. Download or copy the translated text

**To run manually:**
```bash
source venv/bin/activate
python app.py
```

Translation and quality estimation timing metrics will be displayed in the terminal.

## Project Structure

```
POC_AT_Translation/
├── app.py                      # Flask web application and routing
├── translation_model.py        # NLLB-200 translation with chunking logic
├── quality_estimator.py        # COMET-QE quality assessment
├── pdf_document_management.py  # PDF extraction and text cleaning
├── requirements.txt            # Python dependencies
├── setup.sh                    # Automated setup script
├── run.sh                      # Quick run script
├── templates/                  # HTML templates
│   ├── index.html             # Upload form
│   ├── progress.html          # Real-time progress view
│   └── result.html            # Results display (legacy)
└── docs/                       # Example documents
```

## Acknowledgments

- [NLLB-200 Model](https://huggingface.co/facebook/nllb-200-distilled-1.3B) by Meta AI
- [COMET-QE Model](https://huggingface.co/Unbabel/wmt22-cometkiwi-da) by Unbabel
- Built with Flask, PyTorch, Transformers
- PDF extraction via pdfminer.six, PyPDF2, and poppler-utils

---

**Note**: This is a proof of concept for research/evaluation purposes. For production translation needs, consider commercial services (DeepL, Google Translate) which offer significantly higher quality for specialized content.