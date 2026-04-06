# llm_explainer.py
# In this module we use Ollama with Mistral to generate a natural language
# explanation of the detected sentiment using few-shot prompt engineering.
# This is the step that takes our system beyond simple classification.

import requests
import json

# We point to the local Ollama API that runs on our machine
OLLAMA_URL = 'http://localhost:11434/api/generate'

# We use Mistral 7B as our local LLM because it offers a good balance between quality and performance on consumer hardware
OLLAMA_MODEL = 'mistral'


# We build a few-shot prompt so the model understands exactly what kind of explanation we expect from it
def build_few_shot_prompt(review, sentiment_label, language):

    # We define few-shot examples in English to guide the model's output format
    few_shot_en = """You are an expert sentiment analyst. Given a review and its sentiment label, you must write a short explanation (2-3 sentences) of why the text has that sentiment. Focus on specific words and phrases from the review.

Example 1:
Review: "The battery lasts all day and the screen is gorgeous."
Sentiment: Very Positive
Explanation: The reviewer highlights two strong positive features: excellent battery life and display quality. Words like "gorgeous" convey high satisfaction and enthusiasm about the product.

Example 2:
Review: "It broke after one week. Total waste of money."
Sentiment: Very Negative
Explanation: The reviewer expresses strong disappointment due to a product failure within a very short time. Phrases like "total waste of money" indicate frustration and a feeling of being deceived.

Example 3:
Review: "It works fine but nothing special."
Sentiment: Neutral
Explanation: The reviewer acknowledges basic functionality but finds no standout qualities. The phrase "nothing special" suggests the product met expectations without exceeding them.

Now analyze this one:
Review: "{review}"
Sentiment: {sentiment}
Explanation:"""

    # We define few-shot examples in Spanish for Spanish input reviews
    few_shot_es = """Eres un analista experto en sentimientos. Dada una reseña y su etiqueta de sentimiento, escribe una explicación breve (2-3 frases) de por qué el texto tiene ese sentimiento. Céntrate en palabras y frases concretas de la reseña.

Ejemplo 1:
Reseña: "La batería dura todo el día y la pantalla es preciosa."
Sentimiento: Muy Positivo
Explicación: El usuario destaca dos características muy positivas: la duración de la batería y la calidad de pantalla. La palabra "preciosa" transmite un alto nivel de satisfacción.

Ejemplo 2:
Reseña: "Se rompió en una semana. Una pérdida de dinero total."
Sentimiento: Muy Negativo
Explicación: El usuario expresa una fuerte decepción por un fallo del producto en muy poco tiempo. La frase "pérdida de dinero total" indica frustración e indignación.

Ahora analiza esta:
Reseña: "{review}"
Sentimiento: {sentiment}
Explicación:"""

    # We choose the prompt template based on the detected language
    if language == 'es':
        return few_shot_es.format(review=review, sentiment=sentiment_label)
    else:
        return few_shot_en.format(review=review, sentiment=sentiment_label)


# We call the local Ollama API and stream the response back
def explain_sentiment(review, sentiment_label, language):
    prompt = build_few_shot_prompt(review, sentiment_label, language)

    # We send the request to Ollama with streaming disabled for simplicity
    try:
        response = requests.post(
            OLLAMA_URL,
            json={
                'model': OLLAMA_MODEL,
                'prompt': prompt,
                'stream': False,
                # We set a low temperature so the explanation is focused
                # and not overly creative or random
                'options': {
                    'temperature': 0.3,
                    'num_predict': 150
                }
            },
            timeout=60
        )

        data = response.json()
        explanation = data.get('response', '').strip()

        # We keep only the first 3 sentences to ensure conciseness
        sentences = explanation.split('.')
        short = '. '.join(s.strip() for s in sentences[:3] if s.strip())
        if short and not short.endswith('.'):
            short += '.'

        return short if short else 'No explanation could be generated.'

    except requests.exceptions.ConnectionError:
        # We inform the user clearly if Ollama is not running
        return '[ERROR] Ollama is not running. Please start it with: ollama serve'
    except Exception as e:
        return f'[ERROR] Unexpected error: {str(e)}'