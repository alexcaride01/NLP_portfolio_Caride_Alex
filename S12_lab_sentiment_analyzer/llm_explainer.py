# llm_explainer.py
# In this module we use Ollama with Mistral to generate a natural language explanation of the detected sentiment.
# We combine few-shot prompt engineering with RAG: we retrieve semantically similar past reviews and inject them as additional
# context for the LLM.
# This makes the explanations more grounded and consistent.

import requests

# We point to the local Ollama API running on our machine
OLLAMA_URL = 'http://localhost:11434/api/generate'
OLLAMA_MODEL = 'mistral'


# We build the prompt combining fixed few-shot examples with dynamically retrieved similar reviews from our vector store
def build_rag_prompt(review, sentiment_label, language, similar_reviews):

    # We format the retrieved similar reviews as dynamic examples so the LLM can use real past cases as additional context
    dynamic_examples = ''
    if similar_reviews:
        dynamic_examples = '\nHere are some similar reviews that were analyzed before:\n'
        for i, s in enumerate(similar_reviews, 1):
            dynamic_examples += (
                f'\nSimilar review {i} (similarity: {s["similarity"]}%):\n'
                f'Text: "{s["text"]}"\n'
                f'Sentiment: {s["sentiment"]}\n'
            )
        dynamic_examples += '\nUse these as additional context if relevant.\n'

    # We define the base few-shot prompt in English with static examples
    prompt_en = f"""You are an expert sentiment analyst. Given a review and its sentiment label, write a short explanation (2-3 sentences) of why the text has that sentiment. Focus on specific words and phrases from the review.

Example 1:
Review: "The battery lasts all day and the screen is gorgeous."
Sentiment: Very Positive
Explanation: The reviewer highlights two strong positive features: battery life and display quality. The word "gorgeous" conveys high enthusiasm and satisfaction.

Example 2:
Review: "It broke after one week. Total waste of money."
Sentiment: Very Negative
Explanation: The reviewer expresses strong disappointment due to an early product failure. The phrase "total waste of money" signals frustration and regret.
{dynamic_examples}
Now analyze this one:
Review: "{review}"
Sentiment: {sentiment_label}
Explanation:"""

    # We define the base few-shot prompt in Spanish for Spanish reviews
    prompt_es = f"""Eres un analista experto en sentimientos. Dada una reseña y su etiqueta, escribe una explicación breve (2-3 frases) de por qué el texto tiene ese sentimiento. Céntrate en palabras y frases concretas.

Ejemplo 1:
Reseña: "La batería dura todo el día y la pantalla es preciosa."
Sentimiento: Muy Positivo
Explicación: El usuario destaca la duración de la batería y la calidad de pantalla. La palabra "preciosa" transmite un alto nivel de satisfacción.

Ejemplo 2:
Reseña: "Se rompió en una semana. Una pérdida de dinero total."
Sentimiento: Muy Negativo
Explicación: El usuario expresa fuerte decepción por un fallo temprano del producto. La frase "pérdida de dinero" indica frustración e indignación.
{dynamic_examples}
Ahora analiza esta:
Reseña: "{review}"
Sentimiento: {sentiment_label}
Explicación:"""

    # We select the prompt language based on what we detected earlier
    if language == 'es':
        return prompt_es
    return prompt_en


# We call the Ollama API and return the generated explanation
def explain_sentiment(review, sentiment_label, language, similar_reviews=None):
    if similar_reviews is None:
        similar_reviews = []

    prompt = build_rag_prompt(review, sentiment_label, language, similar_reviews)

    try:
        response = requests.post(
            OLLAMA_URL,
            json={
                'model': OLLAMA_MODEL,
                'prompt': prompt,
                'stream': False,
                # We keep temperature low for focused and consistent explanations
                'options': {
                    'temperature': 0.3,
                    'num_predict': 150
                }
            },
            timeout=60
        )

        data = response.json()
        explanation = data.get('response', '').strip()

        # We trim the output to 3 sentences maximum for readability
        sentences = explanation.split('.')
        short = '. '.join(s.strip() for s in sentences[:3] if s.strip())
        if short and not short.endswith('.'):
            short += '.'

        return short if short else 'No explanation could be generated.'

    except requests.exceptions.ConnectionError:
        return '[ERROR] Ollama is not running. Please start it with: ollama serve'
    except Exception as e:
        return f'[ERROR] Unexpected error: {str(e)}'