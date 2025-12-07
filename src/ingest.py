import json
from typing import List, Dict

def load_faq(path: str) -> List[Dict]:

    # Load data from JSON file and normalize fields
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    entries = []
    for e in data:
        entries.append({
            "question": str(e.get("question","")).strip(),
            "answer": str(e.get("answer","")).strip(),
            "url": str(e.get("url","")).strip(),
            "category": str(e.get("category","")).strip()
        })
        
    return entries
