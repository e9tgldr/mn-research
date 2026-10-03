"""Round 4: what tasks does an actual Mongolian AI company train for?

dulguun222/mongolian-chat-dataset is MongolGPT LLC's training set (9,807 rows) and one
sample revealed a legal-complaint CLASSIFICATION task - the same shape as Laya's job.
Enumerate the system prompts to get the real task taxonomy, and check MongolGPT's footprint.
"""
import json
import re
import ssl
import urllib.parse
import urllib.request
from collections import Counter

UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/126.0 Safari/537.36")
HDRS = {"User-Agent": UA}
CTX = ssl.create_default_context()
CTX.check_hostname = False
CTX.verify_mode = ssl.CERT_NONE


def get(url, timeout=45):
    req = urllib.request.Request(url, headers=HDRS)
    with urllib.request.urlopen(req, timeout=timeout, context=CTX) as r:
        return r.read(1500000).decode("utf-8", "replace")


def rows(ds, offset, length):
    u = ("https://datasets-server.huggingface.co/rows?dataset=%s&config=default"
         "&split=train&offset=%d&length=%d" % (urllib.parse.quote(ds), offset, length))
    return json.loads(get(u)).get("rows") or []


def hr(t):
    print("\n" + "=" * 90)
    print(t)
    print("=" * 90)


DS = "dulguun222/mongolian-chat-dataset"
hr("A. MongolGPT TASK TAXONOMY (from system prompts)")
allrows = []
for off in (0, 100, 200, 300, 400):
    try:
        allrows += rows(DS, off, 100)
    except Exception as e:  # noqa: BLE001
        print("  fetch %d failed: %s" % (off, e))
print("sampled %d rows" % len(allrows))

prompts = []
for r in allrows:
    p = (r["row"].get("system_prompt") or "").strip()
    prompts.append(p)

# first line / task sentence is the most descriptive
heads = Counter()
for p in prompts:
    h = re.sub(r"\s+", " ", p.split(".")[0])[:95]
    heads[h] += 1
print("\n--- most common opening lines of the system prompt ---")
for h, n in heads.most_common(25):
    print("  %4d  %s" % (n, h))

print("\n--- rows that look like CLASSIFICATION tasks ---")
seen = set()
for r, p in zip(allrows, prompts):
    low = p.lower()
    if any(k in low for k in ("classify", "category", "classif", "select the category")):
        key = re.sub(r"\s+", " ", p)[:300]
        if key in seen:
            continue
        seen.add(key)
        row = r["row"]
        print("\n  PROMPT : %s" % key)
        print("  INPUT  : %s" % str(row.get("question"))[:150])
        print("  ANSWER : %s" % str(row.get("solution"))[:150])
        if len(seen) >= 8:
            break

hr("B. HOW OFTEN IS THE ANSWER A SHORT CATEGORY LABEL (not prose)?")
short = sum(1 for r in allrows if len(str(r["row"].get("solution") or "")) < 60)
print("  %d / %d answers are under 60 chars -> those are label-style, not explanations"
      % (short, len(allrows)))
print("\n  examples of short-label answers:")
n = 0
for r in allrows:
    s = str(r["row"].get("solution") or "")
    if 0 < len(s) < 45:
        print("    %-22s <= %s" % (s, str(r["row"].get("question"))[:70]))
        n += 1
        if n >= 12:
            break

hr("C. MongolGPT FOOTPRINT ON HUGGING FACE")
for q in ("MongolGPT", "mongolian-chat", "mn-gpt"):
    try:
        res = json.loads(get("https://huggingface.co/api/models?search=%s&limit=10"
                             % urllib.parse.quote(q)))
        print("\n[models: %s] %s" % (q, [m["modelId"] for m in res] if res else "none"))
    except Exception as e:  # noqa: BLE001
        print("  ERR", e)
    try:
        res = json.loads(get("https://huggingface.co/api/datasets?search=%s&limit=10"
                             % urllib.parse.quote(q)))
        print("[datasets: %s] %s" % (q, [d["id"] for d in res] if res else "none"))
    except Exception as e:  # noqa: BLE001
        print("  ERR", e)

hr("D. Mongolian AI companies / orgs on HF")
for org in ("MongolGPT", "CMLI-NLP", "tugstugi", "bayartsogt", "Blgn94"):
    try:
        res = json.loads(get("https://huggingface.co/api/models?author=%s&limit=30"
                             % urllib.parse.quote(org)))
        print("\n%s: %d models" % (org, len(res)))
        for m in res[:12]:
            print("    %-56s dl=%s %s" % (m["modelId"], m.get("downloads"),
                                          m.get("pipeline_tag") or "-"))
    except Exception as e:  # noqa: BLE001
        print("  %s ERR %s" % (org, e))
