"""Follow-up: chase the leads the search surfaced. Direct fetches only."""
import json
import re
import ssl
import urllib.parse
import urllib.request

UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/126.0 Safari/537.36")
HDRS = {"User-Agent": UA, "Accept-Language": "en,mn;q=0.8"}
CTX = ssl.create_default_context()
CTX.check_hostname = False
CTX.verify_mode = ssl.CERT_NONE


def get(url, timeout=35):
    req = urllib.request.Request(url, headers=HDRS)
    with urllib.request.urlopen(req, timeout=timeout, context=CTX) as r:
        raw = r.read(1200000)
    for enc in ("utf-8", "cp1251", "latin-1"):
        try:
            return raw.decode(enc)
        except Exception:
            continue
    return raw.decode("utf-8", "replace")


def txt(html):
    html = re.sub(r"(?is)<(script|style|noscript)[^>]*>.*?</\1>", " ", html)
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", html)).strip()


def hr(t):
    print("\n" + "=" * 90)
    print(t)
    print("=" * 90)


# ---------------------------------------------------------------- HF dataset cards
DATASETS = [
    ("Asakuu/mongolian-mcq-dataset", "multiple-choice questions?"),
    ("dulguun222/mongolian-chat-dataset", "chat / messenger text?"),
    ("robertritz/mongolian_news", "news corpus"),
    ("CMLI-NLP/Mongolian-pretrain-dataset", "pretraining corpus"),
    ("Ganaa0614/mongolian-text-dataset", "general text"),
    ("Billyyy/cleaned-mongolian-dataset", "general text"),
    ("bayartsogt/mongolian-ner", "named entities"),
    ("Blgn94/mongolian-twitter-roberta-base-sentiment-ner", "sentiment"),
]

hr("A. HF DATASET DETAIL - what is actually inside the promising ones")
for did, why in DATASETS:
    print("\n### %s   (%s)" % (did, why))
    try:
        d = json.loads(get("https://huggingface.co/api/datasets/" + did))
    except Exception as e:  # noqa: BLE001
        print("   ERR %s" % e)
        continue
    cd = d.get("cardData") or {}
    print("   downloads: %s | likes: %s" % (d.get("downloads"), d.get("likes")))
    for k in ("size_categories", "task_categories", "language", "tags", "configs"):
        if cd.get(k):
            print("   %s: %s" % (k, json.dumps(cd[k], ensure_ascii=False)[:220]))
    desc = (cd.get("dataset_info") or {})
    print("   files: %s" % [s.get("rfilename") for s in (d.get("siblings") or [])][:8])

# batuka classifier - prior art on Mongolian news classification
hr("B. PRIOR ART: Batuka0901/mongolian_news_classifier")
try:
    m = json.loads(get("https://huggingface.co/api/models/Batuka0901/mongolian_news_classifier"))
    print(json.dumps({k: m.get(k) for k in ("modelId", "downloads", "likes", "pipeline_tag",
                                            "tags", "lastModified")}, ensure_ascii=False, indent=2))
    print("files:", [s.get("rfilename") for s in (m.get("siblings") or [])])
except Exception as e:  # noqa: BLE001
    print("ERR", e)

# ---------------------------------------------------------------- telemetr Mongolia bots
hr("C. MONGOLIAN TELEGRAM BOTS - how big is the ecosystem?")
for url in ("https://telemetr.io/en/bots/mongolia",
            "https://telemetr.io/en/bots/mongolia/telegram_platform"):
    print("\n>>> %s" % url)
    try:
        body = txt(get(url))
        print(body[:1800])
    except Exception as e:  # noqa: BLE001
        print("   ERR %s: %s" % (type(e).__name__, e))

# ---------------------------------------------------------------- prior work
hr("D. PRIOR WORK: Mongolian text classification on GitHub")
try:
    body = txt(get("https://github.com/sharavsambuu/mongolian-text-classification"))
    print(body[:2200])
except Exception as e:  # noqa: BLE001
    print("ERR", e)

hr("E. GITHUB - Mongolian NLP projects (API search)")
for q in ("mongolian+nlp", "mongolian+text+classification", "mongolian+dataset"):
    try:
        r = json.loads(get("https://api.github.com/search/repositories?q=%s&sort=stars&per_page=10" % q))
        print("\n[%s]  total=%s" % (q, r.get("total_count")))
        for it in (r.get("items") or [])[:10]:
            print("   %-52s stars=%-5s %s" % (it["full_name"], it["stargazers_count"],
                                              (it.get("description") or "")[:70]))
    except Exception as e:  # noqa: BLE001
        print("   ERR %s: %s" % (type(e).__name__, e))
