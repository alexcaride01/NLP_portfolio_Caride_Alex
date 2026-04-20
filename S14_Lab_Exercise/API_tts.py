# We use the ElevenLabs API to generate high-quality, expressive speech from text
import os
import requests

ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY", "sk_b9fb47e1567746662b9d92101843f61f6059bcbeab576b3d")
BASE_URL = "https://api.elevenlabs.io/v1"

# We define a default voice ID that corresponds to the "Rachel" voice on ElevenLabs
DEFAULT_VOICE_ID = "21m00Tcm4TlvDq8ikWAM"

HEADERS = {
    "xi-api-key": ELEVENLABS_API_KEY,
    "Content-Type": "application/json",
}


# We retrieve all available voices from the user's ElevenLabs account
def list_voices() -> list:
    if not ELEVENLABS_API_KEY:
        print("[API TTS] WARNING: ELEVENLABS_API_KEY not set. Cannot list voices.")
        return []

    print("[API TTS] Fetching available voices from ElevenLabs...")
    response = requests.get(f"{BASE_URL}/voices", headers={"xi-api-key": ELEVENLABS_API_KEY})
    response.raise_for_status()
    voices = response.json().get("voices", [])
    print(f"[API TTS] Found {len(voices)} voice(s):")
    for v in voices:
        print(f"  name={v['name']}  id={v['voice_id']}")
    return voices


