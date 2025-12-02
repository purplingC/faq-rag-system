import requests
from bs4 import BeautifulSoup
import json
import time
from urllib.parse import urljoin
from pathlib import Path
from tqdm import tqdm

BASE = "https://support.tngdigital.com.my/hc/en-my"
FAQ_HOME = f"{BASE}/categories/360002280493-Frequently-Asked-Questions-FAQ"

OUTPUT_DIR = Path(__file__).resolve().parents[1] / "data"
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
OUTPUT_FILE = OUTPUT_DIR / "tng_faq.json"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; tngd-rag-scraper/1.0)"
}

def fetch(url):
    resp = requests.get(url, headers=HEADERS, timeout=10)
    resp.raise_for_status()
    return resp.text

def get_sections(html):
    soup = BeautifulSoup(html, "html.parser")
    links = soup.select("a[href*='/sections/']") # fetch category pages
    urls = {urljoin(BASE, a["href"]) for a in links}
    return list(urls)

def get_articles(section_url):
    html = fetch(section_url)
    soup = BeautifulSoup(html, "html.parser")
    links = soup.select("a[href*='/articles/']")
    urls = {urljoin(BASE, a["href"]) for a in links}
    return list(urls)

# Extract Q/A text
def parse_article(url):
    html = fetch(url)
    soup = BeautifulSoup(html, "html.parser")

    # Title
    title_el = soup.select_one("h1")
    question = title_el.get_text(strip=True) if title_el else ""

    # Content
    body_el = soup.select_one("div.article-body") or soup.select_one("div#article-body")
    answer = body_el.get_text("\n", strip=True) if body_el else ""

    # Category
    category_el = soup.select_one("a[href*='/sections/']")
    category = category_el.get_text(strip=True) if category_el else ""

    return {
        "question": question,
        "answer": answer,
        "url": url,
        "category": category
    }

def main():
    print("Fetching FAQ sections...")
    home_html = fetch(FAQ_HOME)
    sections = get_sections(home_html)

    all_articles = {}

    for section in tqdm(sections, desc="Sections"):
        article_urls = get_articles(section)

        for article_url in tqdm(article_urls, desc="Articles", leave=False):
            if article_url in all_articles:
                continue
            try:
                data = parse_article(article_url)
                all_articles[article_url] = data
                time.sleep(0.3)  # polite scraping
            except Exception as e:
                print(f"Failed {article_url}: {e}")

    # Stored in JSON
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(list(all_articles.values()), f, indent=2, ensure_ascii=False)

    print(f"Saved {len(all_articles)} articles → {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
