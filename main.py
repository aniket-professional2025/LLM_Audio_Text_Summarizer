# Importing Required packages
import os
import whisper
from pydub import AudioSegment
import speech_recognition as sr
from torch.xpu import device
from transformers import pipeline
#from transformers import TFBartForConditionalGeneration, BartTokenizer
import warnings
warnings.filterwarnings('ignore')
from summarizers import Summarizers
from transformers import T5Tokenizer, T5ForConditionalGeneration

# Setting the API key
openai_api_key = ""
os.environ['OPENAI_API_KEY'] = openai_api_key

# Create a python pipeline to get the transcript text from a audio file
def audio_to_english_text_whisper(audio_path):
    if audio_path.lower().endswith('.mp3'):
        audio = AudioSegment.from_mp3(audio_path)
        audio.export("temp.wav", format = "wav")
        file_to_transcribe = "temp.wav"
    elif audio_path.lower().endswith('.wav'):
        file_to_transcribe = audio_path  # Use original WAV directly
    else:
        # For other formats, pydub can often handle them, but we'll convert to WAV explicitly
        audio = AudioSegment.from_file(audio_path)
        audio.export(temp_wav_path, format="wav")
        file_to_transcribe = temp_wav_path
    # Load the whisper model
    model = whisper.load_model("medium", device = "cuda")
    # Use Translation task to translate it into english
    result = model.transcribe(file_to_transcribe, task = 'translate')
    # Return audio transcript as result
    text = result["text"]
    return text

# Summarize text using gemini apis (free tier available) but paid
import google.generativeai as genai
import os

genai.configure(api_key = "")

def summarize_text_gemini_flash(text: str, max_words: int = 250) -> str:
    try:
        model = genai.GenerativeModel('gemini-1.5-flash')
        max_output_tokens = int(max_words * 1.5)
        # Construct the prompt for summarization.
        prompt = (
            f"Summarize the following document concisely within approximately {max_words} words. "
            "Focusing on the paint industry domain"
            "Ensure the summary is coherent and stands alone.\n\n"
            f"Document:\n{text}\n\nSummary:")
        # Set generation configuration, especially max_output_tokens to control summary length.
        generation_config = {"max_output_tokens": max_output_tokens,"temperature": 0.7,"top_p": 0.9,"top_k": 40,}
        # Make the API call to generate content
        print(f"Sending request to Gemini 1.5 Flash with max_output_tokens={max_output_tokens}...")
        response = model.generate_content(contents=[prompt],generation_config=generation_config)
        # Check if the response contains generated text
        if response and response.candidates and response.candidates[0].content and response.candidates[0].content.parts:
            summary = response.candidates[0].content.parts[0].text
            return summary
        else:
            return "Error: No summary generated or unexpected response structure."
    except Exception as e:

        return f"An error occurred during summarization: {e}"

