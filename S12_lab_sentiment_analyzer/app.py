# app.py
# In this module we build the Gradio user interface.
# We now display the retrieved similar reviews alongside the explanation
# so the user can see the RAG system working in real time.

import gradio as gr
from pipeline import run_pipeline

# We keep the session history in memory during the app's lifetime
history = []


# We define the main analysis function called by the Gradio interface
def analyze(text):
    if not text.strip():
        return ' Please enter a review.', 0, {}, '', '', '', ''

    result = run_pipeline(text)

    if 'error' in result:
        return result['error'], 0, {}, '', '', '', ''

    # We format the retrieved similar reviews for display in the UI
    similar_text = ''
    if result['similar_reviews']:
        similar_text = '**Similar reviews retrieved from history:**\n\n'
        for i, s in enumerate(result['similar_reviews'], 1):
            similar_text += (
                f"**{i}.** *(similarity: {s['similarity']}%)* "
                f"→ {s['sentiment']}\n"
                f"> {s['text'][:120]}...\n\n"
            )
    else:
        similar_text = '*No similar reviews yet. Analyze more reviews to enable retrieval.*'

    # We save the result to the session history
    history.append({
        'review': result['original_text'][:60] + '...',
        'sentiment': result['sentiment_label'],
        'confidence': f"{result['confidence']}%"
    })

    return (
        result['sentiment_label'],
        result['confidence'],
        result['all_scores'],
        result['explanation'],
        result['cleaned_text'],
        f"Detected language: {result['language'].upper()}",
        similar_text
    )


# We format the session history as a list for the Gradio table
def get_history():
    if not history:
        return [['—', '—', '—']]
    return [[h['review'], h['sentiment'], h['confidence']] for h in history]


# We build the full Gradio Blocks interface
with gr.Blocks(title='Sentiment Analyzer', theme=gr.themes.Soft()) as demo:

    gr.Markdown('# 🔍 Sentiment Analyzer')
    gr.Markdown(
        'A local NLP pipeline using **HuggingFace BERT** for classification '
        'and **Mistral 7B via Ollama** for RAG-based explanation.'
    )

    with gr.Row():
        with gr.Column(scale=2):
            input_text = gr.Textbox(
                label='Review text',
                placeholder='Paste your review here in English or Spanish...',
                lines=5
            )
            submit_btn = gr.Button('Analyze', variant='primary')

        with gr.Column(scale=1):
            lang_output = gr.Textbox(
                label='Detected language',
                interactive=False
            )
            cleaned_output = gr.Textbox(
                label='Cleaned text (preprocessed)',
                interactive=False,
                lines=3
            )

    with gr.Row():
        sentiment_output = gr.Textbox(
            label='Detected sentiment',
            interactive=False
        )
        confidence_output = gr.Slider(
            label='Confidence (%)',
            minimum=0,
            maximum=100,
            interactive=False,
            value=0
        )

    scores_output = gr.Label(label='All sentiment scores (%)')

    # We display the RAG retrieval results so the user can see
    # which past reviews influenced the current explanation
    similar_output = gr.Markdown(label='Retrieved similar reviews (RAG)')

    explanation_output = gr.Textbox(
        label='LLM Explanation (Mistral + RAG)',
        interactive=False,
        lines=4
    )

    gr.Markdown('---')
    gr.Markdown('### Session history')

    history_btn = gr.Button('Refresh history')
    history_table = gr.Dataframe(
        headers=['Review (preview)', 'Sentiment', 'Confidence'],
        datatype=['str', 'str', 'str'],
        interactive=False
    )

    # We wire up the analyze button to the main pipeline function
    submit_btn.click(
        fn=analyze,
        inputs=[input_text],
        outputs=[
            sentiment_output,
            confidence_output,
            scores_output,
            explanation_output,
            cleaned_output,
            lang_output,
            similar_output
        ]
    )

    # We wire up the history refresh button
    history_btn.click(
        fn=get_history,
        inputs=[],
        outputs=[history_table]
    )


# We launch the app locally to comply with the local LLM requirement
if __name__ == '__main__':
    demo.launch(share=False)