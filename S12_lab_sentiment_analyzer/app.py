# app.py
# In this module we build the Gradio user interface.
# We keep the UI layer completely separate from the logic
# so the pipeline can be reused or tested independently.

import gradio as gr
from pipeline import run_pipeline

# We store the analysis history in memory for the current session
history = []


# We define the main function that the Gradio interface will call
def analyze(text):
    if not text.strip():
        return 'Please enter a review.', 0, {}, '', '', ''

    result = run_pipeline(text)

    # We handle pipeline errors gracefully in the UI
    if 'error' in result:
        return result['error'], 0, {}, '', '', ''

    # We add the result to our session history
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
        f"Detected language: {result['language'].upper()}"
    )


# We format the session history as a table for Gradio to display
def get_history():
    if not history:
        return [['—', '—', '—']]
    return [[h['review'], h['sentiment'], h['confidence']] for h in history]


# We build the interface using Gradio Blocks for maximum layout control
with gr.Blocks(title='Sentiment Analyzer', theme=gr.themes.Soft()) as demo:

    gr.Markdown('# Sentiment Analyzer')
    gr.Markdown('Analyze the sentiment of any product or movie review. Powered by a local NLP pipeline using HuggingFace + Mistral via Ollama.')

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

    explanation_output = gr.Textbox(
        label='LLM Explanation (Mistral via Ollama)',
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

    # We connect the analyze button to the main pipeline function
    submit_btn.click(
        fn=analyze,
        inputs=[input_text],
        outputs=[
            sentiment_output,
            confidence_output,
            scores_output,
            explanation_output,
            cleaned_output,
            lang_output
        ]
    )

    # We connect the history button to the history formatting function
    history_btn.click(
        fn=get_history,
        inputs=[],
        outputs=[history_table]
    )


# We launch the app locally; share=False ensures it stays on our machine
if __name__ == '__main__':
    demo.launch(share=False)