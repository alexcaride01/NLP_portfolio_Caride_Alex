# We use OpenAI Whisper locally to transcribe audio files without sending data to any external server
import whisper
import os


# We load the Whisper model once so we can reuse it across multiple transcriptions
def load_whisper_model(model_size: str = "base") -> whisper.Whisper:
    print(f"[Local STT] Loading Whisper model: {model_size}")
    model = whisper.load_model(model_size)
    return model


# We transcribe the given audio file and return both the text and the detected language
def transcribe_audio(audio_path: str, model: whisper.Whisper) -> dict:
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    print(f"[Local STT] Transcribing: {audio_path}")
    # We run the transcription with language detection enabled so we know what language was spoken
    result = model.transcribe(audio_path)

    language = result.get("language", "unknown")
    text = result.get("text", "").strip()

    # We extract confidence scores from segments when they are available
    segments = result.get("segments", [])
    avg_confidence = None
    if segments:
        probs = [s.get("no_speech_prob", 0) for s in segments]
        avg_confidence = round(1 - (sum(probs) / len(probs)), 4)

    return {
        "text": text,
        "language": language,
        "avg_speech_confidence": avg_confidence,
        "segments": [
            {"start": s["start"], "end": s["end"], "text": s["text"]}
            for s in segments
        ],
    }


# We print a structured summary of the transcription result so the user can review it easily
def display_transcription(result: dict) -> None:
    print("\n--- Local STT Result ---")
    print(f"Language detected : {result['language']}")
    print(f"Speech confidence : {result['avg_speech_confidence']}")
    print(f"Full text         : {result['text']}")
    if result["segments"]:
        print("Segments:")
        for seg in result["segments"]:
            print(f"  [{seg['start']:.1f}s → {seg['end']:.1f}s] {seg['text']}")
    print("------------------------\n")


if __name__ == "__main__":
    # We provide a simple demo that transcribes a sample file if it exists
    sample_path = "NLP_portfolio_Caride_Alex/S14_Lab_Exercise/rp_16000.wav"
    model = load_whisper_model("base")
    try:
        result = transcribe_audio(sample_path, model)
        display_transcription(result)
    except FileNotFoundError as e:
        print(f"[Local STT] Demo skipped: {e}")
        print("[Local STT] Please provide a 'sample_audio.wav' file to test it.")