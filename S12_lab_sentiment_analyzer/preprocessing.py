# preprocessing.py
# In this module we handle all the text cleaning steps
# before passing the input to our sentiment model.

import re
from langdetect import detect, LangDetectException


# We define a function to remove noise from the raw input text
def clean_text(text):
    # We remove URLs that would add noise to the analysis
    text = re.sub(r'http\S+|www\S+', '', text)

    # We remove characters that are not useful for sentiment analysis
    text = re.sub(r'[^a-zA-ZáéíóúÁÉÍÓÚüÜñÑ0-9\s\.\,\!\?\'\"-]', '', text)

    # We collapse multiple spaces into a single one
    text = re.sub(r'\s+', ' ', text).strip()

    return text


# We define a function to detect the language of the input text
def detect_language(text):
    try:
        lang = detect(text)
        return lang
    except LangDetectException:
        # If detection fails we default to English
        return 'en'


# We run the full preprocessing pipeline and return a dictionary
# with all the information we need for the next steps
def preprocess(text):
    cleaned = clean_text(text)
    lang = detect_language(cleaned)

    return {
        'original': text,
        'cleaned': cleaned,
        'language': lang
    }