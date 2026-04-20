# We use pyttsx3 to synthesize speech locally without any internet connection
import pyttsx3
import os


# We initialize the TTS engine and configure its default properties
def build_engine(rate: int = 165, volume: float = 1.0, voice_index: int = 0) -> pyttsx3.Engine:
    engine = pyttsx3.init()
    engine.setProperty("rate", rate)
    engine.setProperty("volume", volume)

    voices = engine.getProperty("voices")
    if voices:
        # We pick the requested voice index but fall back to the first one if the index is out of range
        chosen = voices[voice_index] if voice_index < len(voices) else voices[0]
        engine.setProperty("voice", chosen.id)
        print(f"[Local TTS] Voice selected: {chosen.name}")
    return engine


# We list all voices available on this machine so the user can choose one
def list_voices() -> list:
    engine = pyttsx3.init()
    voices = engine.getProperty("voices")
    print("[Local TTS] Available voices:")
    for i, v in enumerate(voices):
        print(f"  [{i}] id={v.id}  name={v.name}  lang={getattr(v, 'languages', 'N/A')}")
    return voices


# We speak text aloud in real time using the system's audio output
def speak_text(text: str, engine: pyttsx3.Engine) -> None:
    print(f"[Local TTS] Speaking: {text[:80]}{'...' if len(text) > 80 else ''}")
    engine.say(text)
    engine.runAndWait()


# We save the synthesized speech to a WAV file on disk for later use
def save_speech(text: str, output_path: str = "local_tts_output.wav", engine: pyttsx3.Engine = None) -> str:
    if engine is None:
        engine = build_engine()
    print(f"[Local TTS] Saving speech to: {output_path}")
    engine.save_to_file(text, output_path)
    engine.runAndWait()
    if os.path.exists(output_path):
        print(f"[Local TTS] File saved successfully ({os.path.getsize(output_path)} bytes)")
    return output_path


if __name__ == "__main__":
    # We demonstrate listing voices and then synthesizing a short sentence
    list_voices()
    # With voice_index=0 we get a Spanish voice
    eng = build_engine(rate=160,voice_index=0)
    sample_text = (
        "Hola! Esto es una demostración de texto a voz local. "
        "Estamos ejecutando todo localmente usando pyttsx3."
    )
    speak_text(sample_text, eng)
    save_speech(sample_text, "NLP_portfolio_Caride_Alex/S14_Lab_Exercise/local_tts_output_spanish.wav", eng)