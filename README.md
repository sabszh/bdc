# Bot de ContinuOnus

AI-powered chatbot for the "Carte de Continuonus" art installation, featuring bilingual support (Danish/English), guided conversation flow, and cloned voice synthesis.

## 📁 Project Structure

```
bdc-2/
├── app.py                    # FastAPI server (main entry point)
├── chatbot.py                # Core RAG chatbot logic
├── wsgi.py                   # WSGI config for PythonAnywhere deployment
├── requirements.txt          # Python dependencies (~50-100MB)
├── .env                      # API keys (not in git)
├── .gitignore               # Git ignore rules
├── static/
│   ├── index.html           # React frontend (KITT-style interface)
│   └── audio/               # Pre-recorded stage messages (2.9MB)
│       ├── da_WELCOME.mp3   # Danish welcome (874KB)
│       ├── da_MEMORY_1.mp3  # Danish memory prompt 1
│       ├── da_MEMORY_2.mp3  # Danish memory prompt 2
│       ├── da_QUESTION_1.mp3 # Danish question prompt 1
│       ├── da_QUESTION_2.mp3 # Danish question prompt 2
│       ├── da_GOODBYE.mp3   # Danish goodbye
│       ├── en_WELCOME.mp3   # English welcome (805KB)
│       ├── en_MEMORY_1.mp3  # English memory prompt 1
│       ├── en_MEMORY_2.mp3  # English memory prompt 2
│       ├── en_QUESTION_1.mp3 # English question prompt 1
│       ├── en_QUESTION_2.mp3 # English question prompt 2
│       └── en_GOODBYE.mp3   # English goodbye
└── .venv/                   # Virtual environment (local only)
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
# Activate virtual environment
.venv\Scripts\Activate.ps1  # Windows PowerShell
# or
source .venv/bin/activate   # Linux/Mac

# Install packages
pip install -r requirements.txt
```

### 2. Set Up Environment Variables

Create a `.env` file with your API keys:

```env
ELEVENLABS_API_KEY=sk_...
HUGGINGFACE_API_KEY=hf_...
PINECONE_API_KEY=pcsk_...
VOICE_ID=4PzN60Ir6O2U6RzaQ5fm
MODEL_ID=eleven_multilingual_v2
INDEX_NAME_BOT=botcon
INDEX_NAME_CHAT=bdc-interaction-data
LLM_REPO_ID=meta-llama/Llama-3.1-8B-Instruct
```

### 3. Run the Server

```bash
python app.py
```

Visit: http://localhost:8000

## 🎭 Features

### Guided Conversation Flow (7 Stages)
1. **START** - Language selection & boot sequence
2. **WELCOME** - Introduction with pre-recorded audio
3. **MEMORY_1** - First memory sharing
4. **MEMORY_2** - Second memory sharing
5. **QUESTION_1** - First question to AI
6. **QUESTION_2** - Second question to AI
7. **GOODBYE** - Farewell message

### Bilingual Support
- **Danish (da)** - Default language
- **English (en)** - Full translation

### Audio Features
- **Pre-recorded stage messages** - Instant playback (no API delay)
- **ElevenLabs TTS** - Cloned voice for AI responses
- **Auto-play** - Seamless audio experience

### AI Stack
- **Embeddings**: HuggingFace cloud API (768 dims, no local models)
- **LLM**: Llama-3.1-8B-Instruct via Cerebras
- **Vector DB**: Pinecone serverless (AWS US-EAST-1)
- **TTS**: ElevenLabs eleven_multilingual_v2

## 📦 Dependencies

Total size: ~50-100MB (optimized for PythonAnywhere free tier)

### Core (25MB)
- FastAPI - Web framework
- Uvicorn - ASGI server
- python-dotenv - Environment variables
- python-multipart - Form data handling

### AI/ML APIs (15MB)
- huggingface-hub - HF API client
- elevenlabs - TTS synthesis

### LangChain (10MB)
- langchain-huggingface - Cloud embeddings

### Vector Database (10MB)
- pinecone - Vector store client

**Note**: No PyTorch or sentence-transformers (saves 900MB!)

## 🌐 Deployment

### PythonAnywhere (Free Tier Compatible)

1. Upload project files
2. Create virtual environment
3. Install requirements
4. Configure WSGI (use `wsgi.py`)
5. Set environment variables in dashboard
6. Start web app

Total disk usage: ~55MB (fits 512MB limit)

## 🔧 Architecture

### Memory Mode (Fast)
- User shares memory
- Save to vectorstore without AI
- Simple acknowledgment
- ~1 second response time

### Question Mode (Full RAG)
- User asks question
- Retrieve from both indexes (botcon + chat history)
- Two-stage LLM response (source + reflection)
- TTS audio generation
- ~5-8 second response time

### Pre-recorded Audio
All stage messages are pre-generated MP3 files for instant playback:
- Eliminates API delays for scripted content
- Better user experience
- Reduced API costs
- Works offline for stage transitions

## 📊 Performance

- **Memory saving**: < 1 second
- **Question answering**: 5-8 seconds
- **Stage transitions**: Instant (pre-recorded audio)
- **Total app size**: ~55MB

## 🎨 Frontend

React-based KITT-style interface with:
- Retro terminal aesthetics
- Stage progression indicators
- Audio controls (play/pause/skip)
- Voice input support
- Bilingual UI

## 🔐 Security

- API keys in `.env` (not committed to git)
- `.gitignore` configured
- Session-based conversation tracking
- No user data stored locally

## 📝 License

Part of the "Carte de Continuonus" art installation by Helene Nymann.
