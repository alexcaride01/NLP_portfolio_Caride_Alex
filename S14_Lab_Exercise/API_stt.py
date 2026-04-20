# We use AssemblyAI's REST API to transcribe audio files with high accuracy and speaker diarization
import os
import time
import requests
 
# We read the API key from the environment so we never hardcode credentials in the source code
ASSEMBLYAI_API_KEY = os.getenv("ASSEMBLYAI_API_KEY", "")
BASE_URL = "https://api.assemblyai.com/v2"
 
HEADERS = {
    "authorization": ASSEMBLYAI_API_KEY,
    "content-type": "application/json",
}
 
 
# We upload a local audio file to AssemblyAI's storage so the transcription job can access it
def upload_audio(file_path: str) -> str:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Audio file not found: {file_path}")
 
    print(f"[API STT] Uploading audio: {file_path}")
    upload_headers = {"authorization": ASSEMBLYAI_API_KEY}
 
    with open(file_path, "rb") as f:
        response = requests.post(f"{BASE_URL}/upload", headers=upload_headers, data=f)
 
    response.raise_for_status()
    audio_url = response.json()["upload_url"]
    print(f"[API STT] Upload complete. URL: {audio_url}")
    return audio_url
 
 
# We submit a transcription job with optional speaker diarization enabled
def submit_transcription(audio_url: str, speaker_labels: bool = True) -> str:
    payload = {
        "audio_url": audio_url,
        "speaker_labels": speaker_labels,
        # We also request sentiment analysis as an extra enrichment from the same call
        "sentiment_analysis": True,
    }
    print(f"[API STT] Submitting transcription job (speaker_labels={speaker_labels})")
    response = requests.post(f"{BASE_URL}/transcript", json=payload, headers=HEADERS)
    response.raise_for_status()
    job_id = response.json()["id"]
    print(f"[API STT] Job submitted. ID: {job_id}")
    return job_id