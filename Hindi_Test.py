# In this script, we run the functions that are created in the main.py. This process will make the complete
# Process very easy and the process will take less time.

# Import the function from the main.py file
from main import audio_to_english_text_whisper,summarize_with_transformers

# Check the result on different audio files
result = audio_to_english_text_whisper(r"C:\Users\Webbies\Jupyter_Notebooks\Audio_Text_Summarizer\Hindi.mp3")
print("The audio Transcribe is:\n", result)

print("--------------------------------------------------------------------------------------------------------------------")

# Get the summary of the result
summary = summarize_with_transformers(result, min_length = 20, max_length = 50, do_sample = False)
print("The summarized text is:\n", summary)