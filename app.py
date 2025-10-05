import base64
import io
import os
from typing import Optional, Dict, Any

from fastapi import FastAPI, Request, Form
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from dotenv import load_dotenv
from elevenlabs.client import ElevenLabs

from chatbot import ChatBot

load_dotenv()

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

# Init services
print("="*60)
print("INITIALIZING BOT DE CONTINUONUS SERVER")
print("="*60)

print("[INIT] Loading ChatBot (default Danish)...")
bot = ChatBot(repo_id=os.getenv("LLM_REPO_ID"), language="da")
print(f"[INIT] ChatBot initialized with repo: {os.getenv('LLM_REPO_ID')}")

print("[INIT] Loading ElevenLabs client...")
eleven = ElevenLabs(api_key=os.getenv("ELEVENLABS_API_KEY"))
VOICE_ID = os.getenv("VOICE_ID", "4PzN60Ir6O2U6RzaQ5fm")
MODEL_ID = os.getenv("MODEL_ID", "eleven_multilingual_v2")
print(f"[INIT] ElevenLabs initialized - Voice: {VOICE_ID}, Model: {MODEL_ID}")

print("="*60)
print("SERVER READY - Listening for requests...")
print("="*60)

@app.get("/health")
def health():
    print("[HEALTH] Health check requested")
    return {"status": "ok", "service": "Bot de Continuonus"}

@app.get("/")
def root():
    print("[ROOT] Serving main interface")
    return FileResponse("static/index.html")

@app.post("/api/tts")
async def api_tts(
    scripted_text: str = Form(...),
    language: str = Form("da")
):
    """Generate TTS audio for scripted messages (welcome, stage transitions, etc.)"""
    import asyncio
    import time
    
    start = time.time()
    print(f"[TTS] Generating audio for scripted message: '{scripted_text[:50]}...'")
    
    def generate_audio():
        tts_start = time.time()
        print(f"[TTS] Starting ElevenLabs synthesis...")
        try:
            audio = eleven.text_to_speech.convert(
                text=scripted_text,
                voice_id=VOICE_ID,
                model_id=MODEL_ID,
                output_format="mp3_44100_128",
            )
            if not isinstance(audio, (bytes, bytearray)):
                audio = b"".join(audio)
            tts_time = time.time() - tts_start
            print(f"[TTS] Audio generated in {tts_time:.2f}s, size: {len(audio)} bytes")
            return base64.b64encode(audio).decode("utf-8")
        except Exception as e:
            print(f"[TTS] ERROR: {e}")
            return ""
    
    # Generate audio in background thread
    audio_b64 = await asyncio.to_thread(generate_audio)
    
    total_time = time.time() - start
    print(f"[TTS] Total request time: {total_time:.2f}s\n")
    
    return JSONResponse({
        "audio_b64": audio_b64
    })

@app.post("/api/chat")
async def api_chat(
    user_input: str = Form(...),
    session_id: str = Form(...),
    user_name: str = Form("User"),
    user_location: str = Form("Unknown"),
    chat_history: Optional[str] = Form(None),
    continuous_json: Optional[str] = Form(None),
    language: str = Form("da"),
    mode: str = Form("question")  # "memory" or "question"
):
    import asyncio
    import time
    
    start = time.time()
    print(f"[API] Starting chat request: '{user_input[:50]}...'")
    
    # call pipeline
    continuous_data: Optional[Dict[str, Any]] = None
    if continuous_json:
        import json
        try:
            continuous_data = json.loads(continuous_json)
        except Exception:
            continuous_data = {"raw": continuous_json}

    # If mode is "memory", just save to vectorstore without running full pipeline
    if mode == "memory":
        print(f"[API] Memory mode - saving without AI response")
        pipeline_start = time.time()
        
        # Simple acknowledgment based on language
        ai_text = "Tak for at dele din erindring. Den er nu en del af Carte de Continuonus." if language == "da" else "Thank you for sharing your memory. It is now part of Carte de Continuonus."
        
        # Save to vectorstore
        bot.upsert_vectorstore(
            user_input=user_input,
            ai_output=ai_text,
            user_name=user_name,
            user_location=user_location,
            session_id=session_id,
            continuous_data=continuous_data
        )
        
        pipeline_time = time.time() - pipeline_start
        print(f"[API] Memory saved in {pipeline_time:.2f}s")
        
        result = {
            "ai_output": ai_text,
            "session_history": bot.retrieve_session(session_id)
        }
    else:
        # Question mode - run full pipeline
        print(f"[API] Calling chatbot pipeline (language: {language})...")
        pipeline_start = time.time()
        result = bot.pipeline(
            user_input=user_input,
            user_name=user_name,
            session_id=session_id,
            user_location=user_location,
            chat_history=chat_history,
            continuous_data=continuous_data,
            language=language
        )
        pipeline_time = time.time() - pipeline_start
        print(f"[API] Pipeline completed in {pipeline_time:.2f}s")

    ai_text = result["ai_output"]
    print(f"[API] AI response length: {len(ai_text)} chars")

    # Run TTS in thread pool to not block
    def generate_audio():
        tts_start = time.time()
        print(f"[TTS] Starting ElevenLabs synthesis...")
        try:
            audio = eleven.text_to_speech.convert(
                text=ai_text,
                voice_id=VOICE_ID,
                model_id=MODEL_ID,
                output_format="mp3_44100_128",
            )
            if not isinstance(audio, (bytes, bytearray)):
                audio = b"".join(audio)
            tts_time = time.time() - tts_start
            print(f"[TTS] Audio generated in {tts_time:.2f}s, size: {len(audio)} bytes")
            return base64.b64encode(audio).decode("utf-8")
        except Exception as e:
            print(f"[TTS] ERROR: {e}")
            return ""
    
    # Generate audio in background thread
    audio_b64 = await asyncio.to_thread(generate_audio)
    
    total_time = time.time() - start
    print(f"[API] Total request time: {total_time:.2f}s\n")

    return JSONResponse({
        "text": ai_text,
        "audio_b64": audio_b64,
        "session_history": result["session_history"],  # structured, sorted
    })
