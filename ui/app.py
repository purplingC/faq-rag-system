from flask import Flask, request, render_template_string
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.rag_bot import ask_tngd_bot

app = Flask(__name__)

TEMPLATE = """
<!doctype html>
<html>
  <head>
    <meta charset="utf-8"/>
    <title>TNGD FAQ</title>
    <style>
      body { font-family:sans-serif; max-width:800px; margin:auto; padding:20px; }
      textarea { width:100%; height:70px; padding:8px; }
      .answer { background:#f2f4f6; padding:12px; margin-top:10px; border-radius:6px; white-space:pre-wrap; }
    </style>
  </head>

  <body>
    <h1>TNGD FAQ</h1>

    <form method="post">
      <textarea name="q" placeholder="Ask a question...">{{ request.form.get('q','') }}</textarea>
      <button type="submit">Ask</button>
    </form>

    {% if resp %}
      <h2>Final Answer</h2>
      <div class="answer">{{ resp.final_answer }}</div>

      {% if resp.url %}
        <p>Source: <a href="{{ resp.url }}" target="_blank">{{ resp.url }}</a></p>
      {% endif %}
    {% endif %}
  </body>
</html>
"""

@app.route("/", methods=["GET", "POST"])
def index():
    resp = None
    if request.method == "POST":
        q = request.form.get("q","")
        resp = ask_tngd_bot(q)
    return render_template_string(TEMPLATE, resp=resp)

if __name__ == "__main__":
    app.run(
        host="127.0.0.1",
        port=5000,
        debug=False,
        use_reloader=False,
        threaded=False
    )
