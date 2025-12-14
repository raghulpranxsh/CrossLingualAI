import os
import tempfile
import subprocess
import whisper
import json
import faiss
import numpy as np
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from werkzeug.utils import secure_filename
import time

app = Flask(__name__)
CORS(app)

# Configuration
UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'outputs'
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv', 'wav'}
MAX_FILE_SIZE = 500 * 1024 * 1024  # 500MB

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Language mappings
WHISPER_TO_NLLB = {
    "en": "eng_Latn",
    "es": "spa_Latn",
    "fr": "fra_Latn",
    "de": "deu_Latn",
    "ja": "jpn_Jpan",
    "hi": "hin_Deva",
    "ta": "tam_Taml",
    "te": "tel_Telu",
    "zh": "zho_Hans",
    "ko": "kor_Hang",
    "ar": "arb_Arab",
    "ru": "rus_Cyrl"
}

# Global models (loaded on startup)
whisper_model = None
translation_tokenizer = None
translation_model = None
rag_model = None
faiss_index = None
rag_documents = None

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def load_models():
    """Load all ML models on startup"""
    global whisper_model, translation_tokenizer, translation_model, rag_model, faiss_index, rag_documents
    
    print("Loading Whisper model...")
    whisper_model = whisper.load_model("base")
    
    print("Loading translation model...")
    model_name = "facebook/nllb-200-distilled-600M"
    translation_tokenizer = AutoTokenizer.from_pretrained(model_name)
    translation_model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    
    print("Loading RAG model...")
    rag_model = SentenceTransformer("all-MiniLM-L6-v2")
    
    print("Loading FAISS index...")
    load_or_create_faiss_index()

def load_or_create_faiss_index():
    """Load or create FAISS index from news corpus"""
    global faiss_index, rag_documents
    
    index_path = "faiss_index.index"
    docs_path = "docs.json"
    
    # Find news corpus JSON file
    news_files = [f for f in os.listdir('.') if f.startswith('news_corpus') and f.endswith('.json')]
    
    if not news_files:
        print("Warning: No news corpus JSON files found. RAG will not work.")
        faiss_index = None
        rag_documents = []
        return
    
    news_file = news_files[0]  # Use the first one found
    
    if os.path.exists(index_path) and os.path.exists(docs_path):
        print("Loading existing FAISS index...")
        faiss_index = faiss.read_index(index_path)
        with open(docs_path, "r", encoding="utf-8") as f:
            rag_documents = json.load(f)
        print(f"Loaded {len(rag_documents)} documents from index")
    else:
        print("Creating new FAISS index...")
        documents = []
        
        # Load news articles
        with open(news_file, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, list):
                articles = data
            else:
                articles = data.get("articles", [])
            
            for article in articles:
                content = article.get("content") or article.get("description") or article.get("title", "")
                if content:
                    documents.append(content)
        
        if not documents:
            print("Warning: No valid content found in news corpus.")
            faiss_index = None
            rag_documents = []
            return
        
        # Generate embeddings
        print(f"Generating embeddings for {len(documents)} documents...")
        doc_embeddings = rag_model.encode(documents, convert_to_numpy=True, show_progress_bar=True)
        
        # Build FAISS index
        dimension = doc_embeddings.shape[1]
        faiss_index = faiss.IndexFlatL2(dimension)
        faiss_index.add(doc_embeddings.astype('float32'))
        
        # Save index and documents
        faiss.write_index(faiss_index, index_path)
        with open(docs_path, "w", encoding="utf-8") as f:
            json.dump(documents, f)
        
        rag_documents = documents
        print(f"Created FAISS index with {len(documents)} documents")

def extract_audio_from_video(video_path):
    """Extract audio from video file"""
    output_audio_path = tempfile.mktemp(suffix='.wav')
    cmd = [
        'ffmpeg', '-i', video_path,
        '-vn',  # No video
        '-acodec', 'pcm_s16le',
        '-ar', '16000',
        '-ac', '1',  # Mono
        '-y',  # Overwrite output file
        output_audio_path
    ]
    subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
    return output_audio_path

def transcribe_audio(audio_path):
    """Transcribe audio using Whisper"""
    result = whisper_model.transcribe(audio_path)
    return result["text"], result["language"], result.get("segments", [])

def translate_text(text, source_lang_code, target_lang_code="en"):
    """Translate text using NLLB model"""
    if source_lang_code == target_lang_code:
        return text
    
    src_nllb = WHISPER_TO_NLLB.get(source_lang_code)
    tgt_nllb = WHISPER_TO_NLLB.get(target_lang_code, "eng_Latn")
    
    if not src_nllb:
        return text  # Return original if language not supported
    
    translation_tokenizer.src_lang = src_nllb
    translation_tokenizer.tgt_lang = tgt_nllb
    forced_bos_token_id = translation_tokenizer.convert_tokens_to_ids(translation_tokenizer.tgt_lang)
    
    # Split text into sentences for better translation
    sentences = text.strip().split('. ')
    if not sentences[-1]:
        sentences = sentences[:-1]
    
    translated_parts = []
    for sentence in sentences:
        if not sentence.strip():
            continue
        try:
            inputs = translation_tokenizer(
                sentence,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            )
            translated_tokens = translation_model.generate(
                **inputs,
                forced_bos_token_id=forced_bos_token_id,
                max_length=512,
                num_beams=4,
                early_stopping=True
            )
            translated_text = translation_tokenizer.decode(
                translated_tokens[0],
                skip_special_tokens=True
            )
            translated_parts.append(translated_text)
        except Exception as e:
            print(f"Translation error: {e}")
            translated_parts.append(sentence)  # Fallback to original
    
    return ". ".join(translated_parts)

def check_relevance(text, threshold=0.2):
    """Check if text is relevant to news corpus using RAG"""
    if faiss_index is None or not rag_documents:
        return True, None, 1.0  # If no RAG index, assume relevant
    
    query_embedding = rag_model.encode([text], convert_to_numpy=True)
    D, I = faiss_index.search(query_embedding.astype('float32'), k=1)
    
    matched_doc = rag_documents[I[0][0]]
    matched_embedding = rag_model.encode([matched_doc], convert_to_numpy=True)
    score = cosine_similarity(query_embedding, matched_embedding)[0][0]
    
    # Convert numpy float32 to Python float for JSON serialization
    score = float(score)
    
    # Return relevance status, matched doc, and score
    return score >= threshold, matched_doc, score

def generate_srt_from_segments(segments, filename):
    """Generate SRT file from Whisper segments"""
    with open(filename, "w", encoding="utf-8") as f:
        for i, segment in enumerate(segments, 1):
            start = format_timestamp(segment["start"])
            end = format_timestamp(segment["end"])
            text = segment["text"].strip()
            f.write(f"{i}\n{start} --> {end}\n{text}\n\n")
    return filename

def generate_srt_from_text(text, filename, duration_seconds=60):
    """Generate SRT file from plain text (fallback)"""
    sentences = [s.strip() for s in text.split('.') if s.strip()]
    if not sentences:
        sentences = [text]
    
    time_per_sentence = duration_seconds / len(sentences) if sentences else 5
    
    with open(filename, "w", encoding="utf-8") as f:
        for i, sentence in enumerate(sentences):
            start_time = i * time_per_sentence
            end_time = (i + 1) * time_per_sentence
            start = format_timestamp(start_time)
            end = format_timestamp(end_time)
            f.write(f"{i+1}\n{start} --> {end}\n{sentence}\n\n")
    return filename

def format_timestamp(seconds):
    """Format seconds to SRT timestamp format"""
    hrs = int(seconds // 3600)
    mins = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds - int(seconds)) * 1000)
    return f"{hrs:02d}:{mins:02d}:{secs:02d},{millis:03d}"

def embed_subtitles(video_path, srt_path, output_path):
    """Embed subtitles into video using ffmpeg"""
    # Escape the SRT path for ffmpeg
    srt_path_escaped = srt_path.replace('\\', '\\\\').replace(':', '\\:')
    
    cmd = [
        'ffmpeg', '-i', video_path,
        '-vf', f"subtitles={srt_path_escaped}",
        '-c:a', 'copy',
        '-y',  # Overwrite output
        output_path
    ]
    result = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    
    if result.returncode != 0:
        print(f"FFmpeg error: {result.stderr}")
        raise Exception(f"FFmpeg failed: {result.stderr}")
    
    return output_path

@app.route('/')
def index():
    """Serve the main HTML page"""
    return send_file('index.html')

@app.route('/api/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'models_loaded': whisper_model is not None
    })

@app.route('/api/upload', methods=['POST'])
def upload_video():
    """Handle video upload and processing"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        source_lang = request.form.get('sourceLanguage', 'auto')
        target_lang = request.form.get('targetLanguage', 'en')
        
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type'}), 400
        
        # Save uploaded file
        filename = secure_filename(file.filename)
        video_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(video_path)
        
        # Check file size
        file_size = os.path.getsize(video_path)
        if file_size > MAX_FILE_SIZE:
            os.remove(video_path)
            return jsonify({'error': 'File too large (max 500MB)'}), 400
        
        # Step 1: Extract audio
        print("Extracting audio...")
        audio_path = extract_audio_from_video(video_path)
        
        # Step 2: Transcribe
        print("Transcribing...")
        native_text, detected_lang, segments = transcribe_audio(audio_path)
        
        # Use detected language if auto was selected
        if source_lang == 'auto':
            source_lang = detected_lang
        
        # Step 3: Translate
        print("Translating...")
        translated_text = translate_text(native_text, source_lang, target_lang)
        
        # Step 4: RAG relevance check (informational only - don't block processing)
        print("Checking relevance...")
        relevant, matched_doc, relevance_score = check_relevance(translated_text)
        
        if not relevant:
            print(f"Warning: Low relevance score ({relevance_score:.3f}), but proceeding anyway...")
            # Don't block - just log a warning. Users should be able to translate any video.
        
        # Step 5: Generate SRT with translated text
        print("Generating subtitles...")
        srt_filename = os.path.join(OUTPUT_FOLDER, f"{filename}_subtitles.srt")
        if segments and source_lang != target_lang:
            # Translate segments individually to preserve timestamps
            translated_segments = []
            for segment in segments:
                translated_seg_text = translate_text(segment["text"], source_lang, target_lang)
                translated_segments.append({
                    "start": segment["start"],
                    "end": segment["end"],
                    "text": translated_seg_text
                })
            generate_srt_from_segments(translated_segments, srt_filename)
        elif segments:
            # No translation needed, use original segments
            generate_srt_from_segments(segments, srt_filename)
        else:
            # Fallback: estimate duration from video
            duration = 60  # Default, could extract from video metadata
            generate_srt_from_text(translated_text, srt_filename, duration)
        
        # Step 6: Embed subtitles
        print("Embedding subtitles...")
        output_filename = f"{filename.rsplit('.', 1)[0]}_translated.mp4"
        output_path = os.path.join(OUTPUT_FOLDER, output_filename)
        embed_subtitles(video_path, srt_filename, output_path)
        
        # Cleanup
        os.remove(audio_path)
        os.remove(video_path)
        
        return jsonify({
            'success': True,
            'output_file': output_filename,
            'transcription': native_text,
            'translation': translated_text,
            'detected_language': detected_lang,
            'relevance_score': round(float(relevance_score), 3),
            'download_url': f'/api/download/{output_filename}'
        })
        
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/api/download/<filename>', methods=['GET'])
def download_file(filename):
    """Download processed video file"""
    file_path = os.path.join(OUTPUT_FOLDER, secure_filename(filename))
    if os.path.exists(file_path):
        return send_file(file_path, as_attachment=True)
    return jsonify({'error': 'File not found'}), 404

if __name__ == '__main__':
    print("Starting SecureNews server...")
    print("Loading models (this may take a few minutes)...")
    load_models()
    print("Models loaded! Starting server...")
    app.run(debug=True, host='0.0.0.0', port=5001)

