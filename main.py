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

#################################################################################################################
# # Define the pipeline for Hugging face based Summarization
# def summarize_with_transformers(text:str, min_length:int, max_length:int, do_sample:bool):
#     summarizer = pipeline('summarization', model = "facebook/bart-large-cnn")
#     summary = summarizer(text, min_length = min_length, max_length = max_length, do_sample = do_sample)
#     summary_text = summary[0]['summary_text']
#     return summary_text

##################################################################################################################
# # Define a function to get the summarized text from google pegasus model
# def summarize_with_pegasus(text:str, min_length:int, max_length:int, do_sample:bool):
#     summarizer = pipeline('summarization', model = "google/pegasus-xsum")
#     summary = summarizer(text, min_length = min_length, max_length = max_length, do_sample = do_sample)
#     summary_text = summary[0]['summary_text']
#     return summary_text

####################################################################################################################
# # Define a function to summarize with conditional prompts using Facebook's Bart
# def summarize_with_bart_conditional(text, focus_points, max_length = 150):
#     # Create the conditional prompt
#     prompt = f"You are a business analyst expert working on paint industry focusing on: {focus_points}. Summarize the following text accordingly:"
#     full_input = f"{prompt}\n\n{text}"
#     # Load BART model
#     summarizer = pipeline("summarization", model = "facebook/bart-large-cnn")
#     # Generate summary
#     summary = summarizer(full_input, max_length = max_length, min_length = 30, do_sample = False)
#     return summary[0]['summary_text']

#######################################################################################################
# # Summarized a text using Summarizers module
# summ = Summarizers(type = 'normal', device = 'cuda')
#
# # Define a function to summarized text with Summarizers module
# def python_ctrlsum_summarizer(text):
#     summarized_text = summ(text, query = "challenges, manpower, activity, putty, competition")
#     return summarized_text

#########################################################################################################
# # Summarization using the Meta-Llama-3.1-8B-Instruct-Summarizer
# def summarizer_metallama(text:str, min_length:int, max_length:int, do_sample:bool):
#     summarizer = pipeline('summarization', model= "raaec/Meta-Llama-3.1-8B-Instruct-Summarizer", device = 'cuda')
#     summary = summarizer(text, min_length = min_length, max_length = max_length, do_sample = do_sample)
#     summary_text = summary[0]['summary_text']
#     return summary_text

###########################################################################################################
# from transformers import T5Tokenizer, T5ForConditionalGeneration
# def flant5_summarizer(text, max_length=150, min_length=100):
#     # Load model and tokenizer
#     model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-large")
#     tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-large")
#     # Preprocess the input text (no prompt added)
#     inputs = tokenizer("summarize: " + text,  return_tensors="pt", truncation=True, max_length=512)
#     # Generate summary
#     outputs = model.generate(**inputs,max_length=max_length,min_length=min_length,num_beams=4,early_stopping=True,
#                              repetition_penalty = 2.5)
#     # Decode and return
#     summarized_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
#     return summarized_text

#######################################################################################################
# # Define a function for Conditional prompt summarization using Google's FLAN T5
# from transformers import T5ForConditionalGeneration, T5Tokenizer
# model_name = "t5-small"
# tokenizer = T5Tokenizer.from_pretrained(model_name)
# model = T5ForConditionalGeneration.from_pretrained(model_name)
# def conditional_t5(text, prompt):
#     prompt = prompt
#     input_text = prompt + " " + text
#     inputs = tokenizer.encode(input_text, return_tensors="pt", max_length=512, truncation=True)
#     summary_ids = model.generate(inputs, max_length=250, min_length=150, num_beams=4, early_stopping=True, repetition_penalty = 2.5)
#     summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
#     return summary

#############################################################################################################
# from transformers import T5Tokenizer, T5ForConditionalGeneration
# tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-large")
# model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-large", device_map="auto")
# # Define a function to generate summary using prompts from google_t5_small model
# def google_t5small_summarizer(text):
#     input_text = f"summarize document in 5 sentences focusing on challenges, solutions, innovations :{text}"
#     input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to("cuda")
#     outputs = model.generate(input_ids)
#     summary = tokenizer.decode(outputs[0])
#     return summary

###############################################################################################################

# Summarize text using gemini apis (free tier available) but paid
import google.generativeai as genai
import os

genai.configure(api_key = "AIzaSyCxdHELZeZykp4jGMNp8JGJKGFexRjc-i8aiml")

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
