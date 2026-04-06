# pipeline.py
# In this module we orchestrate the full end-to-end pipeline:
# preprocessing, sentiment classification, embedding-based retrieval and LLM explanation with RAG context.
# We now include a retrieval step that enriches the LLM prompt with semantically similar reviews from previous analyses.

from preprocessing import preprocess
from sentiment_model import analyze_sentiment
from llm_explainer import explain_sentiment
from embeddings_store import add_to_store, retrieve_similar, store_size


# We define the full pipeline function that the UI will call
def run_pipeline(raw_text):

    # Step 1: We clean the text and detect its language
    preprocessed = preprocess(raw_text)
    cleaned_text = preprocessed['cleaned']
    language = preprocessed['language']

    # We stop early if the cleaned text is empty
    if not cleaned_text:
        return {'error': 'The text is empty after cleaning. Please write a valid review.'}

    # Step 2: We classify the sentiment using the local BERT model
    sentiment = analyze_sentiment(cleaned_text)

    # Step 3: We retrieve semantically similar reviews from the vector store
    # We skip retrieval on the very first analysis since the store is empty
    similar_reviews = []
    if store_size() > 0:
        similar_reviews = retrieve_similar(cleaned_text, top_k=2)

    # Step 4: We generate an explanation using Mistral with RAG context
    explanation = explain_sentiment(
        review=cleaned_text,
        sentiment_label=sentiment['label'],
        language=language,
        similar_reviews=similar_reviews
    )

    # Step 5: We store the current review in the vector store for future retrievals
    add_to_store(
        text=cleaned_text,
        sentiment_label=sentiment['label'],
        confidence=sentiment['confidence']
    )

    # We return all results including retrieval info for the UI
    return {
        'original_text': raw_text,
        'cleaned_text': cleaned_text,
        'language': language,
        'sentiment_label': sentiment['label'],
        'confidence': sentiment['confidence'],
        'all_scores': sentiment['all_scores'],
        'explanation': explanation,
        'similar_reviews': similar_reviews
    }