# PDF Translation Proof of Concept

A Flask-based web application for translating PDF documents using the M2M100 multilingual translation model.

## ⚠️ Current Status: Proof of Concept

This is an experimental system with known limitations:

- **Translation Quality**: ~70% accuracy on legal/technical documents
- **Known Issues**: Terminology inconsistencies, occasional hallucinations, struggles with specialized vocabulary
- **Best Use Case**: General documents with standard language, not legal/official translations

## Features

- PDF text extraction using multiple methods (pdftotext, pdfminer, PyPDF2)
- Multi-language support (10+ languages via M2M100)
- Real-time translation progress tracking
- Side-by-side comparison view
- Text file download of results

## Supported Languages

Portuguese, English, Spanish, French, German, Chinese, Arabic, Russian, Japanese, Korean

## Installation

### Requirements

- Python 3.8+
- CUDA-capable GPU recommended (model will use CPU otherwise)
- 4GB+ RAM for model loading

### Quick Setup (Recommended)

**Linux/macOS:**
```bash
# Clone repository
git clone https://github.com/yourusername/POC_AT_Translation.git
cd POC_AT_Translation

# Run setup script (creates venv and installs dependencies)
chmod +x setup.sh
./setup.sh

# Run the application
./run.sh
```

**Windows (PowerShell):**
```powershell
# Clone repository
git clone https://github.com/yourusername/POC_AT_Translation.git
cd POC_AT_Translation

# Run setup script (creates venv and installs dependencies)
.\setup.ps1

# Run the application
.\run.ps1
```

### Manual Setup

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Optional: Install pdftotext for better PDF extraction
sudo apt-get install poppler-utils  # Ubuntu/Debian
brew install poppler  # macOS
```

## Usage

After setup, the application will be available at `http://127.0.0.1:5000`

1. Upload a PDF file
2. Select source and target languages
3. Wait for translation (progress shown in real-time)
4. Download or copy the translated text

**To run manually:**
```bash
source venv/bin/activate  # On Windows: venv\Scripts\activate
python app.py
```

## Project Structure

```
POC_AT_Translation/
├── app.py                      # Flask application
├── translation_model.py        # M2M100 translation logic
├── pdf_document_management.py  # PDF extraction utilities
├── requirements.txt            # Python dependencies
├── setup.sh / setup.ps1        # Automated setup scripts
├── run.sh / run.ps1            # Quick run scripts
├── templates/                  # HTML templates
│   ├── index.html
│   ├── progress.html
│   └── result.html
└── docs/                       # Example documents
```

## Known Limitations

### Translation Quality Issues

1. **Specialized Terminology**: Legal, medical, and technical terms often mistranslated
2. **Short Phrases**: Struggles with context-free placeholders like "[City, Date]"
3. **Hallucinations**: Occasionally generates unrelated text (~5% of content)
4. **Inconsistency**: Same term may be translated differently across document

### Technical Limitations

- Model size: 1.2B parameters (smaller than commercial solutions)
- Memory usage: ~5GB RAM + model
- Translation speed: ~10-30 seconds per page depending on complexity
- No GPU means 3-5x slower processing

## Improvement Opportunities

We welcome contributions! Priority areas:

1. **Post-processing filters** to catch hallucinations
2. **Terminology glossaries** for consistent specialized terms
3. **Alternative models** (NLLB, mBART, commercial APIs)
4. **Fine-tuning** on domain-specific datasets
5. **Better text extraction** for complex PDF layouts

## Contributing

Please feel free to:
- Report translation errors with input/output examples
- Suggest model improvements or alternatives
- Submit PRs for bug fixes or features
- Share test documents for evaluation

See `docs/` folder for example translations.

## License

MIT License

## Acknowledgments

- [M2M100 Model](https://huggingface.co/facebook/m2m100_1.2B) by Meta AI
- Built with Flask, PyTorch, Transformers
- PDF extraction via pdfminer.six, PyPDF2

---

**Note**: This is a proof of concept for research/evaluation purposes. For production translation needs, consider commercial services (DeepL, Google Translate) which offer significantly higher quality for specialized content.