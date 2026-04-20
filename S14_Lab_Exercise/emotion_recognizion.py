# We perform emotion recognition from audio using a HuggingFace wav2vec2 model
import os
import torch
import torchaudio
from transformers import pipeline

# We cache the pipeline globally so we only load the model once
_emotion_pipeline = None


# We load the wav2vec2 emotion model from HuggingFace the first time it is needed
def get_pipeline():
    global _emotion_pipeline
    if _emotion_pipeline is None:
        print("[Emotion] Loading emotion recognition model (first run downloads weights)...")
        # We use a wav2vec2 model fine-tuned on IEMOCAP for 4-class emotion recognition
        _emotion_pipeline = pipeline(
            "audio-classification",
            model="superb/wav2vec2-base-superb-er",
        )
        print("[Emotion] Model loaded successfully.")
    return _emotion_pipeline


# We load audio using soundfile and librosa to avoid the torchaudio/torchcodec issue on Windows
def load_audio_numpy(file_path: str, target_sr: int = 16000):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Audio file not found: {file_path}")

    import librosa
    # We use librosa to load and resample in one step since it handles MP3 and WAV on Windows
    audio_array, sr = librosa.load(file_path, sr=target_sr, mono=True)
    print(f"[Emotion] Audio loaded: {len(audio_array)} samples at {sr} Hz")
    return audio_array, sr

# We run the classifier and return a ranked list of emotions with confidence scores
def predict_emotion(file_path: str) -> list:
    pipe = get_pipeline()
    audio_array, sr = load_audio_numpy(file_path)

    print(f"[Emotion] Running emotion classification on: {file_path}")
    # We pass the audio as a dict with the sample rate so the pipeline preprocesses it correctly
    results = pipe({"array": audio_array, "sampling_rate": sr}, top_k=4)

    return results


# We print the ranked emotion predictions in a readable format
def display_emotions(results: list) -> None:
    print("\n--- Emotion Recognition Result ---")
    for i, entry in enumerate(results):
        bar = "" * int(entry["score"] * 30)
        print(f"  {i+1}. {entry['label']:<10} {entry['score']*100:5.1f}%  {bar}")
    print("----------------------------------\n")


if __name__ == "__main__":
    sample_path = "NLP_portfolio_Caride_Alex/S14_Lab_Exercise/local_tts_output_sad.wav"
    try:
        results = predict_emotion(sample_path)
        display_emotions(results)
    except FileNotFoundError as e:
        print(f"[Emotion] Demo skipped: {e}")
        print("[Emotion] Please provide a valid audio file to test this module.")