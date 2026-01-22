import os
import requests
import time 
import random
from bs4 import BeautifulSoup
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv

# Load environment variables (not strictly needed for local embeddings, but good practice)
load_dotenv()

# 1. DEFINE TARGET URLS
# We manually list key pages to ensure we get high-quality data (Admissions, Fees, etc.)
urls = [
    "https://www.sunmarke.com/",
    "https://www.sunmarke.com/admissions/scholarships/",
    "https://www.sunmarke.com/admissions/faqs/",
    "https://applynow.sunmarke.com/",
    "https://www.sunmarke.com/admissions/tuition-fees/",
    "https://www.sunmarke.com/learning/eyfs/early-years-curriculum/",
    "https://www.sunmarke.com/learning/primary/our-curriculum-primary/",
    "https://www.sunmarke.com/learning/secondary/our-curriculum-secondary/",
    "https://www.sunmarke.com/activities/sunmarke-free-ecas/",
    "https://www.sunmarke.com/activities/sunmarke-paid-ecas/",
    "https://www.sunmarke.com/activities/third-party-paid-ecas/",
    "https://www.sunmarke.com/about/our-campus/",
    "https://www.sunmarke.com/about/mission-vision-values/"
]

import time
import random

# Use a Session object for connection pooling and cookie handling
session = requests.Session()

def scrape_text_from_url(url):
    print(f"🕷️ Scraping: {url}")
    try:
        # 1. Add realistic headers to mimic a real browser
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
            "Referer": "https://www.google.com/",
            "DNT": "1",
            "Connection": "keep-alive",
            "Upgrade-Insecure-Requests": "1",
        }
        
        # 2. Add a random delay (1 to 3 seconds) to look human
        time.sleep(random.uniform(1, 3))
        
        response = session.get(url, headers=headers, timeout=15)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
            tag.decompose()
            
        text_content = [tag.get_text(strip=True) for tag in soup.find_all(["p", "h1", "h2", "h3", "li"]) if len(tag.get_text(strip=True)) > 20]
        return "\n".join(text_content)
        
    except Exception as e:
        print(f"❌ Failed to scrape {url}: {e}")
        return ""

# 2. MAIN EXECUTION
if __name__ == "__main__":
    print("🚀 Starting Ingestion Process...")
    
    all_text_data = ""
    
    # A. Scrape all URLs
    for url in urls:
        text = scrape_text_from_url(url)
        all_text_data += text + "\n\n"
        
    if not all_text_data.strip():
        print("⚠️ Critical Error: No text scraped. Check your internet or URLs.")
        exit()

    print(f"✅ Scraped {len(all_text_data)} characters of text.")

    # B. Chunk the text
    # We break text into 1000-character chunks with overlap to maintain context
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    chunks = text_splitter.create_documents([all_text_data])
    print(f"🧩 Split into {len(chunks)} chunks.")

    # C. Create Embeddings & Vector DB
    # using 'all-MiniLM-L6-v2' (Small, Fast, Free, runs locally on CPU)
    print("🧠 Generating Embeddings (this might take a minute)...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    vector_db = FAISS.from_documents(chunks, embeddings)

    # D. Save to Local Disk
    vector_db.save_local("faiss_index")
    print("🎉 Success! Vector Database saved to folder 'faiss_index'")