import os
import json
import time
from openai import OpenAI
from dotenv import load_dotenv

# --- CONFIGURATION ---
TARGET_ROWS = 1700           # Start small today
BATCH_SIZE = 5             # Ask for 5 at a time
OUTPUT_FILE = "dataset/hindi_train.json"

# --- SETUP ---
load_dotenv()
client = OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=os.environ.get("GROQ_API_KEY"),
)

# Ensure output folder exists
os.makedirs("dataset", exist_ok=True)

def fetch_batch():
    """Asks AI for a batch of Hindi data."""
    prompt = f"""
    You are an expert Hindi AI assistant.
    Generate {BATCH_SIZE} unique pairs of a 'user question' and an 'AI response' in Hindi.
    Topics: Daily life, Science, Career, Fitness.
    
    Output strictly as a JSON list:
    [
        {{"instruction": "Hindi Question", "output": "Hindi Answer"}},
        {{"instruction": "Hindi Question", "output": "Hindi Answer"}}
    ]
    """
    
    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.8
        )
        content = response.choices[0].message.content
        # Clean up markdown code blocks if present
        content = content.replace("```json", "").replace("```", "").strip()
        print(f"DEBUG PREVIEW: {content}")
        return json.loads(content)
    except Exception as e:
        print(f"⚠️ Error: {e}")
        return []

def main():
    print(f"🚀 Starting Data Gen. Target: {TARGET_ROWS} rows.")
    
    # Load existing data to resume progress
    if os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
    else:
        data = []

    while len(data) < TARGET_ROWS:
        print(f"⚡ Fetching... (Current: {len(data)})")
        batch = fetch_batch()
        
        if batch:
            data.extend(batch)
            # Save immediately (Safety feature)
            with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=4)
            print(f"✅ Saved. Total: {len(data)}")
        
        time.sleep(1) # Respect the API

    print(f"🎉 DONE! Dataset ready at {OUTPUT_FILE}")

if __name__ == "__main__":
    main()