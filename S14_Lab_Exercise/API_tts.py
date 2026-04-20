# We use Google Text-to-Speech (gTTS) as our external API TTS solution
import os
from gtts import gTTS


# We synthesize text using the Google TTS API and save the resulting MP3 to disk
def synthesize_speech(
    text: str,
    output_path: str = "api_tts_output.mp3",
    lang: str = "en",
    slow: bool = False,
) -> str:
    print(f"[API TTS] Requesting speech synthesis for: {text[:80]}...")
    # We create a gTTS object with the target language and speed settings
    tts = gTTS(text=text, lang=lang, slow=slow)
    tts.save(output_path)
    print(f"[API TTS] Audio saved to: {output_path} ({os.path.getsize(output_path)} bytes)")
    return output_path


# We list some of the most common languages supported by gTTS
def list_languages() -> None:
    from gtts.lang import tts_langs
    langs = tts_langs()
    print("[API TTS] Available languages (sample):")
    # We print only the first 10 so the output stays readable
    for code, name in list(langs.items())[:10]:
        print(f"  {code}: {name}")
    print(f"  ... and {len(langs) - 10} more")


if __name__ == "__main__":
    list_languages()
    sample_text = (
        "Welcome to our speech project. "
        "We are demonstrating external API text to speech using Google Text-to-Speech."
    )
    # We synthesize and save the audio to the project folder
    synthesize_speech(
        sample_text,
        "NLP_portfolio_Caride_Alex/S14_Lab_Exercise/api_tts_output.mp3",
        lang="en",
    )