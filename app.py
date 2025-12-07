from pathlib import Path
import tempfile
import threading
from flask import Flask, render_template, request, redirect, url_for, send_file, flash, jsonify
from werkzeug.utils import secure_filename

from pdf_document_management import PDFDocumentManager
from translation_model import TranslationModel
from quality_estimator import QualityEstimator

ALLOWED_EXTENSIONS = {'pdf'}

app = Flask(__name__)
app.secret_key = "dev"
UPLOAD_DIR = Path(tempfile.gettempdir()) / "poc_at_uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

translation_state = {
    "active": False,
    "progress": {"current_batch": 0, "total_batches": 0, "current_text": ""},
    "result": None,
    "original": None,
    "download_url": None,
    "error": None,
    "quality_score": None
}


def allowed_file(filename: str) -> bool:
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/", methods=["GET"])
def index():
    # Only show full language names (filter out 2-letter codes)
    lang_keys = sorted([lang for lang in TranslationModel.LANGUAGE_CODES.keys() if len(lang) > 2])
    return render_template("index.html", languages=lang_keys)


@app.route("/progress", methods=["GET"])
def progress():
    """Return current translation progress as JSON."""
    return jsonify(translation_state)


def _translate_background(uploaded_path: Path, src_lang: str, tgt_lang: str, out_name: str):
    """Background thread function for translation."""
    global translation_state
    try:
        translation_state["error"] = None
        
        # Extract & clean
        raw_text = PDFDocumentManager.extract_text_from_pdf(str(uploaded_path))
        cleaned = PDFDocumentManager.clean_text_for_translation(raw_text)

        # Store original for comparison
        translation_state["original"] = cleaned

        # Create translator and pass reference to shared progress state
        translator = TranslationModel(progress_callback=lambda: translation_state["progress"])
        
        # Translate - translator will update translation_state["progress"] directly
        translated = translator.translate(cleaned, src_lang, tgt_lang)

        # Save translated text to temporary file
        out_path = UPLOAD_DIR / out_name
        PDFDocumentManager.save_text_to_file(translated, str(out_path))

        # Estimate translation quality using COMET-QE
        print("Evaluating translation quality...")
        try:
            estimator = QualityEstimator()
            quality_result = estimator.evaluate_with_interpretation(cleaned, translated)
            translation_state["quality_score"] = quality_result
            print(f"Quality Score: {quality_result['score']:.1f}/100 ({quality_result['level']})")
        except Exception as qe_error:
            print(f"Warning: Quality estimation failed: {qe_error}")
            translation_state["quality_score"] = None

        translation_state["result"] = translated
        translation_state["download_url"] = f"/download/{out_name}"

    except Exception as e:
        translation_state["error"] = str(e)
        print(f"Translation error: {e}")
    finally:
        translation_state["active"] = False
        try:
            uploaded_path.unlink(missing_ok=True)
        except Exception:
            pass


@app.route("/translate", methods=["POST"])
def translate():
    global translation_state

    if "pdf_file" not in request.files:
        flash("No file part")
        return redirect(url_for("index"))

    file = request.files["pdf_file"]
    if file.filename == "":
        flash("No selected file")
        return redirect(url_for("index"))

    if not allowed_file(file.filename):
        flash("Only PDF files are allowed")
        return redirect(url_for("index"))

    src_lang = request.form.get("source_lang", "portuguese")
    tgt_lang = request.form.get("target_lang", "english")
    filename = secure_filename(file.filename)
    uploaded_path = UPLOAD_DIR / filename
    file.save(uploaded_path)

    # reset state
    translation_state.update({
        "active": True,
        "progress": {"current_batch": 0, "total_batches": 0, "current_text": ""},
        "result": None,
        "original": None,
        "download_url": None,
        "error": None,
        "quality_score": None
    })

    out_name = uploaded_path.stem + f"_{tgt_lang}.txt"
    thread = threading.Thread(
        target=_translate_background,
        args=(uploaded_path, src_lang, tgt_lang, out_name),
        daemon=True
    )
    thread.start()

    return render_template("progress.html")


@app.route("/download/<path:filename>", methods=["GET"])
def download_file(filename):
    fpath = UPLOAD_DIR / filename
    if not fpath.exists():
        flash("File not found")
        return redirect(url_for("index"))
    return send_file(str(fpath), as_attachment=True, download_name=filename)


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)