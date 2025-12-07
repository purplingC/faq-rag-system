import sys
import json
from pathlib import Path
import traceback

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Try importing ask_tngd_bot()
try:
    from src.rag_bot import ask_tngd_bot
except Exception:
    print("ERROR: Failed to import ask_tngd_bot from src.rag_bot.")
    traceback.print_exc()
    sys.exit(1)

# Response output keys 
REQUIRED_KEYS = {"question", "retrieved_chunks", "final_answer", "blocked", "decision"}

def run_query(q: str):
    q = (q or "").strip()
    if not q:
        print("Empty query; nothing to do.")
        return

    print("\n" + "=" * 80)
    print("QUESTION:")
    print(q)
    print("-" * 80)

    try:
        resp = ask_tngd_bot(q)
    except Exception:
        print("ERROR: ask_tngd_bot raised an exception:")
        traceback.print_exc()
        return

    # Formatted JSON
    try:
        pretty = json.dumps(resp, indent=2, ensure_ascii=False)
    except Exception:
        pretty = json.dumps({"raw_response_str": str(resp)}, indent=2, ensure_ascii=False)

    print("RESPONSE:")
    print(pretty)
    print("-" * 80)

    # Count chunks
    if isinstance(resp, dict):
        rc = resp.get("retrieved_chunks")
        if isinstance(rc, list):
            print(f"TOTAL NUMBER OF RETRIEVED CHUNKS: {len(rc)}")

    print("=" * 80 + "\n")


def main():
    if len(sys.argv) > 1:
        q = " ".join(sys.argv[1:]).strip()
        run_query(q)
        return

    # Loop
    print("Type a question and press Enter.")
    try:
        while True:
            q = input("\nPlease enter a question (type 'exit' to quit): ").strip()

            if q.lower() == "exit":
                print("Exiting.")
                break

            if not q:
                print("Please enter a valid question.")
                continue

            run_query(q)

    except (KeyboardInterrupt, EOFError):
        print("\nInterrupted. Exiting.")

    except Exception:
        print("Unexpected error in interactive loop:")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
