# sentiment_model.py
# In this module we load and run our local HuggingFace sentiment classifier.
# We keep the model loading separate so we only load it once
# when the app starts, not on every request.

from transformers import pipeline

# We use a multilingual model so we can handle both English and Spanish without needing to switch models depending on the language
MODEL_NAME = 'nlptown/bert-base-multilingual-uncased-sentiment'

# We declare the classifier as None so we can apply lazy loading
_classifier = None


# We use lazy loading so the model is only downloaded and loaded the first time this function is called
def get_classifier():
    global _classifier
    if _classifier is None:
        print(f'[INFO] We are loading the sentiment model for the first time...')
        _classifier = pipeline(
            'text-classification',
            model=MODEL_NAME,
            top_k=None
        )
        print('[INFO] Sentiment model loaded successfully.')
    return _classifier


# We convert the raw star label into a human-readable sentiment category
def stars_to_label(star_str):
    mapping = {
        '1 star':  'Very Negative',
        '2 stars': 'Negative',
        '3 stars': 'Neutral',
        '4 stars': 'Positive',
        '5 stars': 'Very Positive'
    }
    return mapping.get(star_str, star_str)


# We run the classifier on the cleaned text and return the top prediction along with all confidence scores
def analyze_sentiment(cleaned_text):
    classifier = get_classifier()

    # We truncate the text to 512 characters to respect the model's limit
    results = classifier(cleaned_text[:512])[0]

    # We sort all scores from highest to lowest confidence
    results_sorted = sorted(results, key=lambda x: x['score'], reverse=True)
    top = results_sorted[0]

    return {
        'label': stars_to_label(top['label']),
        'raw_label': top['label'],
        'confidence': round(top['score'] * 100, 2),
        'all_scores': {
            stars_to_label(r['label']): round(r['score'] * 100, 2)
            for r in results_sorted
        }
    }