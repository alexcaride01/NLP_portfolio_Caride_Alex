# We tie all five speech modules together into a single interactive demo pipeline
import argparse
import os

from local_stt import load_whisper_model, transcribe_audio, display_transcription
from local_tts import build_engine, speak_text, save_speech
from API_stt import transcribe_with_api
from API_tts import synthesize_speech
from emotion_recognizion import predict_emotion, display_emotions

# We define a fixed sample audio path so each module can find the same input file
DEFAULT_AUDIO = "NLP_portfolio_Caride_Alex/S14_Lab_Exercise/local_tts_output_angry.wav"


# We run the local STT module and return the transcribed text
def step_local_stt(audio_path: str) -> str:
    print("\n========== STEP 1: LOCAL STT (Whisper) ==========")
    model = load_whisper_model("base")
    result = transcribe_audio(audio_path, model)
    display_transcription(result)
    return result["text"]


# We run the local TTS module, speaking the text aloud and saving a WAV file
def step_local_tts(text: str) -> str:
    print("\n========== STEP 2: LOCAL TTS (pyttsx3) ==========")
    
    # We use a separate engine instance for speaking and saving to avoid the Windows freeze bug
    engine_speak = build_engine(rate=160)
    speak_text(text, engine_speak)
    del engine_speak

    engine_save = build_engine(rate=160)
    out_path = "local_tts_output.wav"
    save_speech(text, out_path, engine_save)
    del engine_save
    
    return out_path


# We run the external API STT module using AssemblyAI
def step_api_stt(audio_path: str) -> dict:
    print("\n========== STEP 3: API STT (AssemblyAI) ==========")
    result = transcribe_with_api(audio_path)
    return result


# We run the external API TTS module using ElevenLabs (or gTTS fallback)
def step_api_tts(text: str) -> str:
    print("\n========== STEP 4: API TTS (ElevenLabs / gTTS) ==========")
    out_path = "api_tts_output.mp3"
    synthesize_speech(text, out_path)
    return out_path


# We run the emotion recognition module on the original audio
def step_emotion(audio_path: str) -> list:
    print("\n========== STEP 5: EMOTION RECOGNITION (SpeechBrain) ==========")
    try:
        ranked = predict_emotion(audio_path)
        display_emotions(ranked)
        return ranked
    except Exception as e:
        print(f"[Pipeline] Emotion recognition skipped: {e}")
        return []


# We orchestrate all five steps and print a final summary of outputs
def run_pipeline(audio_path: str = DEFAULT_AUDIO) -> None:
    if not os.path.exists(audio_path):
        # We generate a synthetic audio file using local TTS so the rest of the pipeline has input
        print(f"[Pipeline] '{audio_path}' not found. Generating a synthetic sample with pyttsx3...")
        engine = build_engine()
        sample_text = (
            "Hello! We are testing our speech pipeline. "
            "This audio was generated locally to bootstrap the demo."
        )
        save_speech(sample_text, audio_path, engine)

    transcribed_text = step_local_stt(audio_path)

    if transcribed_text:
        step_local_tts(transcribed_text)
        step_api_stt(audio_path)
        step_api_tts(transcribed_text)
        step_emotion(audio_path)
    else:
        print("[Pipeline] Transcription returned empty text; skipping downstream steps.")

    print("\n========== PIPELINE COMPLETE ==========")
    print("Output files generated:")
    for f in ["local_tts_output.wav", "api_tts_output.mp3"]:
        if os.path.exists(f):
            print(f"  ✓ {f}")


if __name__ == "__main__":
    # We parse a single optional argument so the user can point to any audio file
    parser = argparse.ArgumentParser(description="Speech processing pipeline demo")
    parser.add_argument(
        "--audio",
        type=str,
        default=DEFAULT_AUDIO,
        help="Path to the input audio file (WAV format recommended)",
    )
    args = parser.parse_args()
    run_pipeline(args.audio)