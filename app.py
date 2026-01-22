import streamlit as st
import os
import requests 
import time
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from openai import OpenAI
from streamlit_mic_recorder import mic_recorder
from elevenlabs.client import ElevenLabs

# 1. SETUP & CONFIG
load_dotenv()
st.set_page_config(layout="wide", page_title="Sunmarke School AI Assistant")

def get_key(key_name):
    if key_name in os.environ:
        return os.environ[key_name]
    elif key_name in st.secrets:
        return st.secrets[key_name]
    return None

DEEPGRAM_KEY = get_key("DEEPGRAM_API_KEY")
OPENROUTER_KEY = get_key("OPENROUTER_API_KEY")
ELEVENLABS_KEY = get_key("ELEVENLABS_API_KEY")

# Initialize Session State for Audio Handling
if 'last_processed_audio' not in st.session_state:
    st.session_state.last_processed_audio = None

# 2. LOAD KNOWLEDGE BASE
@st.cache_resource
def load_vector_db():
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    return db

vector_db = load_vector_db()

# 3. CORE FUNCTIONS

def get_transcription(audio_bytes):
    """Transcribes audio using Deepgram's direct API."""
    try:
        url = "https://api.deepgram.com/v1/listen?model=nova-2&smart_format=true&language=en-US"
        headers = {
            "Authorization": f"Token {DEEPGRAM_KEY}",
            "Content-Type": "application/octet-stream"
        }
        response = requests.post(url, headers=headers, data=audio_bytes)
        response.raise_for_status() 
        data = response.json()
        return data['results']['channels'][0]['alternatives'][0]['transcript']
    except Exception as e:
        st.error(f"Transcription Failed: {e}")
        return ""

def generate_voice(text, provider_name):
    """Converts text to speech using ElevenLabs."""
    try:
        client = ElevenLabs(api_key=ELEVENLABS_KEY)
        audio_gen = client.text_to_speech.convert(
            text=text,
            voice_id="pNInz6obpgDQGcFmaJgB", 
            model_id="eleven_multilingual_v2",
            output_format="mp3_44100_128"
        )
        return b"".join(list(audio_gen))
    except Exception as e:
        # Silent fail in UI, log in terminal for character quota tracking
        print(f"ElevenLabs Quota Error for {provider_name}: {e}") 
        return None

def query_llm(prompt, provider="mistral"):
    """
    Unified OpenRouter Query: Using stable 2026 free models.
    """
    # STABLE 2026 MODEL IDS
    model_options = {
        "mistral": [
            "xiaomi/mimo-v2-flash:free",     # 309B MoE - Top performer in 2026
            "google/gemma-3-12b-it:free",    # Brand new, high stability
            "mistralai/mistral-small-24b-instruct:free" # Keep as last resort
        ],
        "deepseek": [
            "deepseek/deepseek-chat", 
            "deepseek/deepseek-r1:free"
        ],
        "llama": [
            "meta-llama/llama-3.3-70b-instruct:free", 
            "meta-llama/llama-3.1-8b-instruct:free"
        ]
    }

    client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=OPENROUTER_KEY)
    
    # Try the main model, then the backup
    for model_id in model_options.get(provider, []):
        try:
            # Add detail instruction to force longer responses
            detailed_prompt = f"{prompt}\n\nPlease provide a very detailed and thorough response based on the context."
            
            response = client.chat.completions.create(
                model=model_id,
                messages=[{"role": "user", "content": detailed_prompt}],
                extra_headers={
                    "HTTP-Referer": "http://localhost:8501", 
                    "X-Title": "Sunmarke School Assistant"
                }
            )
            if response.choices:
                return response.choices[0].message.content
        except Exception:
            continue # Try next model if one is rate-limited

    return f"⚠️ {provider.upper()} API is currently busy. Try again in 1 minute."


# 4. MAIN UI & MODERN STYLING
st.markdown("""
    <style>
    /* Main Background */
    .stApp {
        background-color: #f8f9fa;
    }
    
    /* Column Styling */
    [data-testid="column"] {
        background-color: white;
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        border: 1px solid #e9ecef;
    }
    
    /* Title Styling */
    h1 {
        color: #1e3a8a;
        font-family: 'Helvetica Neue', sans-serif;
        font-weight: 700;
    }
    
    /* Subheader Colors */
    h3 {
        color: #495057;
    }
    </style>
""", unsafe_allow_html=True)

st.title("Sunmarke School AI Assistant 🏫")
st.write("### Ask about admissions, fees, or curriculum.")

# --- DUAL INPUT SYSTEM ---
input_col1, input_col2 = st.columns([1, 4])

with input_col1:
    # Voice Recorder
    audio = mic_recorder(start_prompt="🎤 Voice", stop_prompt="⏹️ Stop", key='voice_input_final')

with input_col2:
    # Text Input
    text_input = st.chat_input("Type your question here...")

# --- SMART INPUT HANDLING ---
user_prompt = None

# 1. Check Text Input FIRST (Priority 1)
if text_input:
    user_prompt = text_input

# 2. Check Audio Input SECOND (Priority 2)
# We only process audio if:
# A) Text input is empty
# B) The audio is actually NEW (not the same stale bytes from 5 mins ago)
elif audio:
    if audio['bytes'] != st.session_state.last_processed_audio:
        # It's a new recording!
        with st.spinner("Transcribing voice..."):
            st.session_state.last_processed_audio = audio['bytes'] # Update state
            user_prompt = get_transcription(audio['bytes'])
    else:
        # It's the same old recording, ignore it.
        pass

# --- PROCESSING LOGIC ---
if user_prompt:
    with st.status("🤖 AI is processing your request...", expanded=True) as status:
        
        st.write(f"**Question:** {user_prompt}")
        
        # Step 1: Retrieve Context
        st.write("Searching Knowledge Base...")
        docs = vector_db.similarity_search(user_prompt, k=3)
        context_text = "\n\n".join([d.page_content for d in docs])
        rag_prompt = f"Context: {context_text}\n\nQuestion: {user_prompt}\n\nAnswer only based on the context."

        # Step 2: Display Side-by-Side Results
        col1, col2, col3 = st.columns(3, gap="medium")

        # --- MISTRAL COLUMN ---
        with col1:
            st.markdown("#### 🌊 Mistral AI")
            res = query_llm(rag_prompt, "mistral")
            if "⚠️" in res: st.error(res)
            else:
                st.info(res)
                v_bytes = generate_voice(res, "Mistral")
                if v_bytes: st.audio(v_bytes, format="audio/mp3")

        # --- DEEPSEEK COLUMN ---
        with col2:
            st.markdown("#### 🧠 DeepSeek")
            res = query_llm(rag_prompt, "deepseek")
            if "⚠️" in res: st.error(res)
            else:
                st.success(res)
                v_bytes = generate_voice(res, "DeepSeek")
                if v_bytes: st.audio(v_bytes, format="audio/mp3")

        # --- LLAMA 3.3 COLUMN ---
        with col3:
            st.markdown("#### 🦙 Llama 3.3")
            res = query_llm(rag_prompt, "llama")
            if "⚠️" in res: st.error(res)
            else:
                st.warning(res)
                v_bytes = generate_voice(res, "Llama")
                if v_bytes: st.audio(v_bytes, format="audio/mp3")
        
        status.update(label="✅ Responses generated!", state="complete")