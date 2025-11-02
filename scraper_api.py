from flask import Flask, render_template, request, jsonify
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
from datetime import datetime
import re, os, json, concurrent.futures, asyncio
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")

app = Flask(__name__)

# ---- GEMINI SETUP ----
if API_KEY:
    genai.configure(api_key=API_KEY)
else:
    print("⚠️ Missing GEMINI_API_KEY in .env")

# ---- FRAUD KEYWORDS ----
FRAUD_KEYWORDS = {
    "fraud","scam","bribe","embezzle","corruption","arrest","probe",
    "investigation","charges","guilty","convicted","illegal","theft","stolen"
}


# ---------------- SCRAPING HELPERS ----------------
def fast_fetch(url):
    """Fetch single page content."""
    try:
        r = requests.get(url, timeout=8)
        if r.status_code == 200:
            return r.text
    except Exception:
        pass
    return None


def get_all_article_links(base_url, max_depth=1):
    """Fast multi-threaded link extraction."""
    visited = set()
    to_visit = {base_url}
    all_links = set()
    parsed_base = urlparse(base_url)

    for _ in range(max_depth):
        new_links = set()
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            future_to_url = {executor.submit(fast_fetch, link): link for link in to_visit}
            for future in concurrent.futures.as_completed(future_to_url):
                link = future_to_url[future]
                html = future.result()
                if not html:
                    continue
                visited.add(link)
                soup = BeautifulSoup(html, "html.parser")
                page_links = {
                    urljoin(base_url, a["href"])
                    for a in soup.find_all("a", href=True)
                }
                valid_links = {
                    l for l in page_links
                    if urlparse(l).netloc == parsed_base.netloc and
                       any(k in l.lower() for k in ["news","article","story","202"])
                }
                all_links.update(valid_links)
                new_links.update(valid_links)
        to_visit = new_links - visited
    all_links.discard(base_url)
    return list(all_links)


def extract_article_date(soup, url):
    """Tries to extract publication date."""
    time_tag = soup.find("time", attrs={"datetime": True})
    if time_tag:
        try:
            return datetime.fromisoformat(time_tag["datetime"].split("T")[0])
        except Exception:
            pass

    for prop in ["article:published_time","og:pubdate","datePublished","pubdate"]:
        meta_tag = soup.find("meta", attrs={"property": prop}) or soup.find("meta", attrs={"name": prop})
        if meta_tag and meta_tag.get("content"):
            match = re.search(r"\d{4}-\d{2}-\d{2}", meta_tag["content"])
            if match:
                try:
                    return datetime.strptime(match.group(), "%Y-%m-%d")
                except Exception:
                    pass

    match = re.search(r'(\d{4})[/-](\d{1,2})[/-](\d{1,2})', url)
    if match:
        try:
            y, m, d = map(int, match.groups())
            return datetime(y, m, d)
        except Exception:
            pass
    return None


def fetch_bbc_article_raw(url, start_dt, end_dt, person_name):
    """Extracts full article content if within date & contains person name."""
    try:
        r = requests.get(url, timeout=8)
        soup = BeautifulSoup(r.text, "html.parser")
        article_tag = soup.find("article")
        if not article_tag:
            return None

        heading_tag = article_tag.find("h1")
        heading = heading_tag.get_text(strip=True) if heading_tag else "No Title"

        article_date = extract_article_date(soup, url)
        if not article_date or not (start_dt <= article_date <= end_dt):
            return None

        paragraphs = [p.get_text(strip=True) for p in article_tag.find_all("p") if p.get_text(strip=True)]
        if not paragraphs:
            return None

        merged = " ".join(paragraphs).lower()
        if person_name.lower() not in merged:
            return None

        return {
            "date": article_date.strftime("%Y-%m-%d"),
            "link": url,
            "heading": heading,
            "paragraphs": paragraphs
        }
    except Exception:
        return None


# ---------------- GEMINI ANALYSIS ----------------
def gemini_analyze_paragraph(paragraph: str):
    """Advanced Gemini analyzer with fine-tuned system prompt."""
    try:
        model = genai.GenerativeModel("gemini-2.0-flash")
        prompt = f"""
You are an **advanced linguistic analysis system** trained for professional-grade sentiment analysis.

TASK OBJECTIVES:
1️⃣ Identify the **core sentiment polarity** ("positive", "negative", or "neutral").
2️⃣ Assign a **tone strength score** (0–100) for emotional intensity.
3️⃣ Detect **dominant emotion** ("joy", "anger", "fear", "sadness", "trust", "disgust", or "none").
4️⃣ Identify the **subject focus** (who/what the paragraph mainly discusses in 3–6 words).
5️⃣ Generate a **concise, factual summary** (3–5 sentences).
6️⃣ Calculate a **readability score** (1–10).
7️⃣ If text implies fraud, illegality, or unethical activity, add “fraud_signal”: true.

Return only JSON:
{{
  "polarity": "positive|negative|neutral",
  "tone_strength": 0–100,
  "emotion": "joy|anger|fear|sadness|trust|disgust|none",
  "subject_focus": "string",
  "summary": "concise summary",
  "readability_score": 1–10,
  "fraud_signal": true|false
}}

Paragraph:
\"\"\"{paragraph}\"\"\"
        """
        resp = model.generate_content(prompt)
        text = (resp.text or "").strip()
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip())
        data = json.loads(text)
        return data
    except Exception as e:
        return {"error": f"Gemini API failed: {e}", "status": "api_key_failed"}


async def analyze_articles_async(person_name, articles):
    """Concurrent paragraph-level Gemini analysis."""
    analyzed = []
    for art in articles:
        paras = [p for p in art.get("paragraphs", []) if person_name.lower() in p.lower()]
        if not paras:
            continue

        summarized = []
        loop = asyncio.get_event_loop()
        tasks = [loop.run_in_executor(None, gemini_analyze_paragraph, p) for p in paras]
        results = await asyncio.gather(*tasks)

        for p, res in zip(paras, results):
            summarized.append({
                "original": p,
                **res
            })

        analyzed.append({
            "date": art["date"],
            "link": art["link"],
            "heading": art["heading"],
            "summarized_paragraphs": summarized
        })
    return analyzed


def compute_overall_stats(person_name, analyzed_articles):
    """Compute combined polarity and fraud probability."""
    total = 0
    counts = {"positive": 0, "negative": 0, "neutral": 0}
    fraud_hits, fraud_base = 0, 0

    for art in analyzed_articles:
        for para in art.get("summarized_paragraphs", []):
            total += 1
            pol = para.get("polarity", "neutral")
            counts[pol] = counts.get(pol, 0) + 1

            if pol == "negative":
                fraud_base += 1
                text = f"{para.get('summary','')} {para.get('original','')}".lower()
                if any(k in text for k in FRAUD_KEYWORDS):
                    fraud_hits += 1

    overall = "neutral"
    if total > 0:
        max_val = max(counts.values())
        for k in ["negative", "positive", "neutral"]:
            if counts[k] == max_val:
                overall = k
                break

    fraud_pct = round((fraud_hits / fraud_base) * 100, 2) if fraud_base else 0
    return {
        "person": person_name,
        "total_paragraphs": total,
        "polarity_counts": counts,
        "overall_polarity": overall,
        "fraud_percentage": fraud_pct
    }


# ---------------- SINGLE FINAL API ----------------
@app.route("/api/final_result", methods=["POST"])
def api_final_result():
    data = request.get_json(silent=True) or {}
    person_name = (data.get("person_name") or "").strip()
    base_url = (data.get("base_url") or "https://www.bbc.com").strip()
    start_date, end_date = data.get("start_date"), data.get("end_date")

    if not person_name or not start_date or not end_date:
        return jsonify({"error": "Missing required fields"}), 400

    try:
        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        end_dt = datetime.strptime(end_date, "%Y-%m-%d")
    except Exception:
        return jsonify({"error": "Invalid date format"}), 400

    links = get_all_article_links(base_url, max_depth=1)
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        results = list(executor.map(lambda l: fetch_bbc_article_raw(l, start_dt, end_dt, person_name), links))
    articles = [a for a in results if a]

    if not articles:
        return jsonify({"status": "no_data", "message": "No articles found"}), 404

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    analyzed = loop.run_until_complete(analyze_articles_async(person_name, articles))
    overall = compute_overall_stats(person_name, analyzed)

    final_result = {
        "person": overall["person"],
        "overall_sentiment": overall["overall_polarity"],
        "fraud_probability": f"{overall['fraud_percentage']}%",
        "paragraphs_analyzed": overall["total_paragraphs"],
        "sentiment_breakdown": overall["polarity_counts"],
        "articles": []
    }

    for art in analyzed:
        final_result["articles"].append({
            "date": art["date"],
            "heading": art["heading"],
            "link": art["link"],
            "paragraphs": [
                {
                    "summary": p.get("summary", ""),
                    "polarity": p.get("polarity", "neutral"),
                    "emotion": p.get("emotion", "none"),
                    "tone_strength": p.get("tone_strength", 50),
                    "subject_focus": p.get("subject_focus", ""),
                    "readability_score": p.get("readability_score", 5),
                    "fraud_signal": p.get("fraud_signal", False)
                }
                for p in art["summarized_paragraphs"]
            ]
        })

    return jsonify({"status": "success", "final_result": final_result})


@app.route("/")
def index():
    return render_template("index.html")

@app.route("/result")
def result_page():
    return render_template("result.html")


if __name__ == "__main__":
    app.run(debug=True, port=5000)
