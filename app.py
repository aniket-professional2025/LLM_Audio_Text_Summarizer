from flask import Flask, render_template, request, jsonify
from main import audio_to_english_text_whisper, summarize_with_transformers
import os

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process_audio', methods=['POST'])
def process_audio():
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file provided'}), 400

    audio = request.files['audio']
    audio_path = os.path.join(UPLOAD_FOLDER, audio.filename)
    audio.save(audio_path)

    transcribed_text = audio_to_english_text_whisper(audio_path)
    summarized_text = summarize_with_transformers(transcribed_text, min_length = 20, max_length = 80, do_sample = False)

    return jsonify({
        'transcription': transcribed_text,
        'summary': summarized_text
    })

if __name__ == '__main__':
    app.run(debug=True)