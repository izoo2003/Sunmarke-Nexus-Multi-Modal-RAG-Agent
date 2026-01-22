# 🎙️ Sunmarke-Nexus: Multi-Modal RAG Agent

![Python](https://img.shields.io/badge/Python-3.10%2B-blue) ![Streamlit](https://img.shields.io/badge/Streamlit-1.32-FF4B4B) ![LangChain](https://img.shields.io/badge/LangChain-RAG-green) ![OpenRouter](https://img.shields.io/badge/OpenRouter-Multi--Model-purple)

**Sunmarke-Nexus** is a production-ready AI voice assistant designed for Sunmarke School. It leverages a **Multi-LLM Architecture** to deliver high-fidelity, hallucination-resistant answers regarding admissions, fees, and curriculum.

The system features a **Retrieval-Augmented Generation (RAG)** pipeline, real-time voice synthesis, and a fault-tolerant inference engine that orchestrates parallel queries across **DeepSeek, Llama 3.3, and Mistral/Xiaomi** models.

---

## 🚀 Key Features

* **🗣️ Voice-to-Voice Interface:** Ultra-low latency Speech-to-Text (STT) via **Deepgram Nova-2** and neural Text-to-Speech (TTS) via **ElevenLabs**.
* **🧠 Multi-Model Consensus:** Simulates a "Panel of Experts" by querying three SOTA open-source models simultaneously (DeepSeek-V3, Llama 3.3 70B, Mistral Small/Xiaomi MiMo).
* **🛡️ Robust Failover Logic:** Implements automated retry strategies and dynamic model switching to handle API rate limits (429) and provider downtime.
* **📚 RAG Knowledge Base:** Uses **LangChain** and **FAISS** to ground answers in verified school documents, eliminating generic hallucinations.
* **💎 Glassmorphic UI:** A modern, responsive Streamlit interface with dual input modes (Voice + Text) and state-preserving chat logic.

---

## 🛠️ Tech Stack

* **Frontend:** Streamlit (Custom CSS, Session State Management)
* **LLM Orchestration:** OpenRouter API (Unified interface for DeepSeek, Meta, Mistral)
* **Vector Database:** FAISS (Facebook AI Similarity Search)
* **Embeddings:** HuggingFace (`all-MiniLM-L6-v2`)
* **Audio Pipeline:** * *Input:* `streamlit-mic-recorder` + Deepgram API
    * *Output:* ElevenLabs Turbo V2
* **Frameworks:** LangChain Community, Python-Dotenv

---

## ⚙️ Architecture

The application follows a modular "Chained" pipeline:

1.  **Ingestion:** PDF/Text data is chunked and embedded into a local FAISS index (`ingest.py`).
2.  **Perception:** User voice is captured and transcribed via Deepgram.
3.  **Retrieval:** The query searches the Vector DB for the top 3 relevant context chunks.
4.  **Inference:** The context + query are sent to 3 independent LLMs in parallel.
5.  **Synthesis:** The selected response is converted to speech and played back.

---

## 📦 Installation

1.  **Clone the Repository**
    ```bash
    git clone [https://github.com/yourusername/sunmarke-nexus.git](https://github.com/yourusername/sunmarke-nexus.git)
    cd sunmarke-nexus
    ```

2.  **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Set Up Environment Variables**
    Create a `.env` file in the root directory and add your keys:
    ```env
    DEEPGRAM_API_KEY="your_deepgram_key"
    OPENROUTER_API_KEY="your_openrouter_key"
    ELEVENLABS_API_KEY="your_elevenlabs_key"
    ```

4.  **Run the Application**
    ```bash
    streamlit run app.py
    ```

---

## 📂 Project Structure

```text
├── faiss_index/             # Vector Database (The "Brain")
│   ├── index.faiss
│   └── index.pkl
├── app.py                   # Main Application Logic (UI + RAG + Audio)
├── ingest.py                # Script to create the Knowledge Base
├── requirements.txt         # Dependency list for deployment
└── README.md                # Documentation
