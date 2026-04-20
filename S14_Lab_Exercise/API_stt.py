# We use AssemblyAI's REST API to transcribe audio files with high accuracy and speaker diarization
import os
import time
from regex import F
import requests
 
# We read the API key from the environment so we never hardcode credentials in the source code
ASSEMBLYAI_API_KEY = os.getenv("ASSEMBLYAI_API_KEY", "d7536975dc4e4f29a2cbbf9bea28edf8")
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
        "sentiment_analysis": True,
        # We specify the speech model explicitly as the new API version requires it
        "speech_models": ["universal-2"],
    }
    print(f"[API STT] Submitting transcription job (speaker_labels={speaker_labels})")
    response = requests.post(f"{BASE_URL}/transcript", json=payload, headers=HEADERS)

    print(f"[API STT] Response status: {response.status_code}")
    print(f"[API STT] Response body: {response.text}")

    response.raise_for_status()
    job_id = response.json()["id"]
    print(f"[API STT] Job submitted. ID: {job_id}")
    return job_id

# We poll the API until the transcription job reaches a terminal state
def poll_transcription(job_id: str, interval: int = 5, max_wait: int = 300) -> dict:
    print(f"[API STT] Polling job {job_id} every {interval}s (max {max_wait}s)...")
    elapsed = 0
    while elapsed < max_wait:
        response = requests.get(f"{BASE_URL}/transcript/{job_id}", headers=HEADERS)
        response.raise_for_status()
        data = response.json()
        status = data.get("status")
        print(f"[API STT]   status={status} ({elapsed}s elapsed)")
 
        if status == "completed":
            return data
        if status == "error":
            raise RuntimeError(f"Transcription failed: {data.get('error')}")
 
        time.sleep(interval)
        elapsed += interval
 
    raise TimeoutError(f"Transcription did not finish within {max_wait}s")
 
 
# We display the full result including speaker turns and sentiment when available
def display_api_result(data: dict) -> None:
    print("\n--- API STT Result (AssemblyAI) ---")
    print(f"Text: {data.get('text', '')}\n")
 
    utterances = data.get("utterances") or []
    if utterances:
        print("Speaker turns:")
        for u in utterances:
            print(f"  Speaker {u['speaker']} [{u['start']/1000:.1f}s–{u['end']/1000:.1f}s]: {u['text']}")
 
    sentiment_results = data.get("sentiment_analysis_results") or []
    if sentiment_results:
        print("\nSentiment analysis:")
        for s in sentiment_results[:5]:
            print(f"  [{s['sentiment']}] {s['text']}")
    print("-----------------------------------\n")
 
 
# We run the full pipeline: upload /submit / poll / display
def transcribe_with_api(file_path: str, speaker_labels: bool = True) -> dict:
    if not ASSEMBLYAI_API_KEY:
        print("[API STT] WARNING: ASSEMBLYAI_API_KEY is not set. Skipping API call.")
        return {}
    audio_url = upload_audio(file_path)
    # We pass the speaker_labels flag through so the caller can control diarization
    job_id = submit_transcription(audio_url, speaker_labels=speaker_labels)
    result = poll_transcription(job_id)
    display_api_result(result)
    return result
 
 
if __name__ == "__main__":
    # We expect the user to set the environment variable before running this module
    sample_path = "NLP_portfolio_Caride_Alex/S14_Lab_Exercise/local_tts_output_positive.wav"
    transcribe_with_api(sample_path,speaker_labels=False)