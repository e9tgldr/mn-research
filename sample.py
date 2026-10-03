"""Round 3: sample the actual contents of the promising Mongolian datasets, and
retry web search now that DuckDuckGo has had time to cool down."""
import json
import re
import ssl
import time
import urllib.parse
import urllib.request

UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/126.0 Safari/537.36")
HDRS = {"User-Agent": UA, "Accept-Language": "en,mn;q=0.8"}
CTX = ssl.create_default_context()
CTX.check_hostname = False
CTX.verify_mode = ssl.CERT_NONE


def get(url, timeout=40):
    req = urllib.request.Request(url, headers=HDRS)
    with urllib.request.urlopen(req, timeout=timeout, context=CTX) as r:
        return r.read(1200000).decode("utf-8", "replace")


def hr(t):
    print("\n" + "=" * 90)
    print(t)
    print("=" * 90)


# The datasets-server serves rows WITHOUT downloading the whole dataset.
PROBE = [
    ("robertritz/mongolian_news", "default", "train"),
    ("dulguun222/mongolian-chat-dataset", "default", "train"),
    ("Asakuu/mongolian-mcq-dataset", "default", "train"),
    ("bayartsogt/mongolian-ner", "default", "train"),
    ("Ganaa0614/mongolian-text-dataset", "default", "train"),
    ("Billyyy/cleaned-mongolian-dataset", "default", "train"),
    ("CMLI-NLP/Mongolian-pretrain-dataset", "default", "train"),
]

hr("A. WHAT IS ACTUALLY INSIDE THESE DATASETS (live row samples)")
for ds, cfg, split in PROBE:
    print("\n### %s" % ds)
    try:
        info = json.loads(get("https://datasets-server.huggingface.co/splits?dataset="
                              + urllib.parse.quote(ds)))
        splits = info.get("splits") or []
        if not splits:
            print("   no split info: %s" % json.dumps(info)[:200])
            continue
        c, s = splits[0]["config"], splits[0]["split"]
        url = ("https://datasets-server.huggingface.co/rows?dataset=%s&config=%s"
               "&split=%s&offset=0&length=2" % (urllib.parse.quote(ds), c, s))
        rows = json.loads(get(url))
        feats = rows.get("features") or []
        print("   config=%s split=%s rows=%s cols=%s"
              % (c, s, rows.get("num_rows_total"), [f["name"] for f in feats]))
        for r in (rows.get("rows") or [])[:2]:
            row = r["row"]
            for k, v in row.items():
                sv = str(v).replace("\n", " ")
                print("      %-12s %s" % (k, sv[:160]))
            print("      ---")
    except Exception as e:  # noqa: BLE001
        print("   ERR %s: %s" % (type(e).__name__, e))

# ------------------------------------------------------------------ web retry
def ddg(q):
    try:
        html = get("https://html.duckduckgo.com/html/?q=" + urllib.parse.quote(q))
    except Exception as e:  # noqa: BLE001
        return [("ERR " + type(e).__name__, str(e)[:80])]
    out = []
    for m in re.finditer(r'<a[^>]+class="result__a"[^>]+href="([^"]+)"[^>]*>(.*?)</a>',
                         html, re.S):
        url, title = m.group(1), re.sub(r"<[^>]+>", "", m.group(2))
        if "uddg=" in url:
            url = urllib.parse.unquote(url.split("uddg=")[1].split("&")[0])
        out.append((re.sub(r"\s+", " ", title).strip()[:100], url))
    return out[:6]


hr("B. WEB SEARCH - retry on the questions that matter")
for q in ["Mongolia online scam fraud victims news",
          "Mongolia Facebook penetration users 2025",
          "Mongolian NLP tugstugi resources",
          "Mongolia smartphone internet usage statistics",
          "Монголд цахим залилан"]:
    print("\n[%s]" % q)
    res = ddg(q)
    if not res:
        print("   (still rate limited / no results)")
    for t, u in res:
        print("   - %s\n     %s" % (t, u[:100]))
    time.sleep(8)
