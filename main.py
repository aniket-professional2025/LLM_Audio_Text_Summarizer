# Import packages
import os
import whisper
from pydub import AudioSegment
import speech_recognition as sr
from transformers import pipeline
import warnings
warnings.filterwarnings('ignore')

# Setting the API key
openai_api_key = "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"
os.environ['OPENAI_API_KEY'] = openai_api_key

# Create the model using a python pipeline
def audio_to_english_text_whisper(audio_path):
    # Convert to WAV if needed (Whisper supports several formats)
    audio = AudioSegment.from_file(audio_path)
    audio.export("temp.wav", format = "wav")
    # Load Whisper model
    model = whisper.load_model("medium", device = "cuda")  # or "small", "medium", "large" for higher accuracy
    # Use 'translate' task for direct translation to English
    result = model.transcribe("temp.wav", task = "translate")
    # Return the audio transcribe as the result
    text = result["text"]
    return text

# Define the pipeline for Hugging Face based summarization
def summarize_with_transformers(text:str, min_length:int, max_length:int, do_sample:bool):
    summarizer = pipeline('summarization', model = "facebook/bart-large-cnn")
    summary = summarizer(text, min_length = min_length, max_length = max_length, do_sample = do_sample)
    summary_text = summary[0]['summary_text']
    return summary_text