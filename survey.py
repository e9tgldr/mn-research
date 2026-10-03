"""Deep research: what Mongolian text data and models actually exist, and what
Mongolian problems are already solved?

Three evidence sources:
  1. Hugging Face API - models and datasets, queried several ways so we don't miss
     things that are not tagged with the obvious keyword.
  2. DuckDuckGo HTML endpoint - URL discovery (rate-limits hard after ~15 queries).
  3. Direct fetch of any promising page found.

Output is a plain report; nothing is inferred that the data doesn't show.
"""
import json
import re
import ssl
import sys
import time
import urllib.parse
import urllib.request

UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/126.0 Safari/537.36")
HDRS = {"User-Agent": UA, "Accept-Language": "en,mn;q=0.8"}
CTX = ssl.create_default_context()
CTX.check_hostname = False
CTX.verify_mode = ssl.CERT_NONE


def get(url, timeout=30):
    req = urllib.request.Request(url, headers=HDRS)
    with urllib.request.urlopen(req, timeout=timeout, context=CTX) as r:
        raw = r.read(900000)
    for enc in ("utf-8", "cp1251", "latin-1"):
        try:
            return raw.decode(enc)
        except Exception:
            continue
    return raw.decode("utf-8", "replace")


def jget(url):
    try:
        return json.loads(get(url))
    except Exception as e:  # noqa: BLE001
        return {"__err": "%s: %s" % (type(e).__name__, e)}


def hr(t):
    print("\n" + "=" * 92)
    print(t)
    print("=" * 92)


# --------------------------------------------------------------------------- HF
def hf_models():
    hr("1. HUGGING FACE - MODELS (Mongolian)")
    seen = {}
    for q in ["mongolian", "mongol", "mn-MN", "khalkha"]:
        for m in jget("https://huggingface.co/api/models?search=%s"
                      "&sort=downloads&direction=-1&limit=30" % urllib.parse.quote(q)):
            if isinstance(m, dict) and "modelId" in m:
                seen[m["modelId"]] = m
    # language-tagged
    for m in jget("https://huggingface.co/api/models?filter=language:mn"
                  "&sort=downloads&direction=-1&limit=40"):
        if isinstance(m, dict) and "modelId" in m:
            seen[m["modelId"]] = m
    rows = sorted(seen.values(), key=lambda m: -(m.get("downloads") or 0))
    print("%-60s %-12s %s" % ("model", "downloads", "task"))
    print("-" * 92)
    for m in rows[:40]:
        print("%-60s %-12s %s" % (m["modelId"], m.get("downloads") or 0,
                                  m.get("pipeline_tag") or "-"))
    print("\ntotal distinct Mongolian-ish models found: %d" % len(rows))
    # what TASKS are covered - tells us what the field has bothered to build
    tasks = {}
    for m in rows:
        t = m.get("pipeline_tag") or "unknown"
        tasks[t] = tasks.get(t, 0) + 1
    print("by task:", json.dumps(tasks, indent=2))


def hf_datasets():
    hr("2. HUGGING FACE - DATASETS (Mongolian)")
    seen = {}
    for q in ["mongolian", "mongol", "mn", "khalkha", "ulaanbaatar"]:
        for d in jget("https://huggingface.co/api/datasets?search=%s"
                      "&sort=downloads&direction=-1&limit=30" % urllib.parse.quote(q)):
            if isinstance(d, dict) and "id" in d:
                seen[d["id"]] = d
    for d in jget("https://huggingface.co/api/datasets?filter=language:mn"
                  "&sort=downloads&direction=-1&limit=40"):
        if isinstance(d, dict) and "id" in d:
            seen[d["id"]] = d
    rows = sorted(seen.values(), key=lambda d: -(d.get("downloads") or 0))
    KEY = ("mongol", "mn-", "khalkha")
    print("--- clearly Mongolian-specific ---")
    for d in rows:
        if any(k in d["id"].lower() for k in KEY) and d.get("downloads", 0) > 15:
            print("%-62s %8s" % (d["id"], d.get("downloads") or 0))
    print("\n--- big multilingual corpora that CONTAIN Mongolian ---")
    for d in rows:
        if not any(k in d["id"].lower() for k in KEY) and d.get("downloads", 0) > 5000:
            print("%-62s %8s" % (d["id"], d.get("downloads") or 0))
    print("\ntotal distinct datasets surfaced: %d" % len(rows))


def hf_targeted():
    hr("3. HUGGING FACE - TARGETED: does the specific data we want exist?")
    probes = [
        ("Mongolian scam / phishing text", "mongolian+scam"),
        ("Mongolian spam text", "mongolian+spam"),
        ("Mongolian news corpus", "mongolian+news"),
        ("Mongolian sentiment", "mongolian+sentiment"),
        ("Mongolian telegram", "telegram+mongolian"),
        ("Mongolian exam / MCQ", "mongolian+exam"),
        ("Mongolian NER", "mongolian+ner"),
        ("Mongolian classification", "mongolian+classification"),
    ]
    for label, q in probes:
        r = jget("https://huggingface.co/api/datasets?search=%s&limit=8" % q)
        r2 = jget("https://huggingface.co/api/models?search=%s&limit=8" % q)
        hits = [x.get("id") or x.get("modelId") for x in r if isinstance(x, dict)] if isinstance(r, list) else []
        hits2 = [x.get("modelId") for x in r2 if isinstance(x, dict)] if isinstance(r2, list) else []
        print("\n[%s]" % label)
        print("   datasets: %s" % (hits if hits else "NONE"))
        print("   models  : %s" % (hits2 if hits2 else "NONE"))


# --------------------------------------------------------------------------- web
def ddg(q):
    """DuckDuckGo HTML endpoint. Returns [(title, url)]."""
    try:
        html = get("https://html.duckduckgo.com/html/?q=" + urllib.parse.quote(q))
    except Exception as e:  # noqa: BLE001
        return [("ERR " + type(e).__name__, str(e))]
    out = []
    for m in re.finditer(r'<a[^>]+class="result__a"[^>]+href="([^"]+)"[^>]*>(.*?)</a>',
                         html, re.S):
        url, title = m.group(1), re.sub(r"<[^>]+>", "", m.group(2))
        if "uddg=" in url:
            url = urllib.parse.unquote(url.split("uddg=")[1].split("&")[0])
        out.append((re.sub(r"\s+", " ", title).strip()[:110], url))
    return out[:8]


def web():
    hr("4. WEB SEARCH - what already exists for Mongolian?")
    queries = [
        "Mongolian language NLP dataset",
        "Mongolian scam phishing message Telegram",
        '"Монгол хэл" NLP corpus dataset',
        "Mongolian school Telegram bot",
        "Mongolian text classification model",
        "ЭЕШ шалгалтын сан тест",
        "Mongolia Facebook users statistics",
        "Mongolian language model AI",
        "Монгол хэлний хиймэл оюун ухаан",
        "Mongolian spam SMS detection",
    ]
    for q in queries:
        print("\n[%s]" % q)
        res = ddg(q)
        if not res:
            print("   (no results - possibly rate limited)")
        for t, u in res:
            print("   - %s\n     %s" % (t, u[:105]))
        time.sleep(3)  # be polite; DDG blocks hard after ~15 rapid queries


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    if which in ("all", "hf"):
        hf_models()
        hf_datasets()
        hf_targeted()
    if which in ("all", "web"):
        web()
