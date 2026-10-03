"""What is actually in Billyyy/cleaned-mongolian-dataset?

The question is whether it can train a classifier. That depends on whether it has labels
(it appears not to) and on the quality/length/domain of the text. Measure before advising.
"""
import json
import re
import ssl
import urllib.parse
import urllib.request

UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/126.0 Safari/537.36")
HDRS = {"User-Agent": UA}
CTX = ssl.create_default_context()
CTX.check_hostname = False
CTX.verify_mode = ssl.CERT_NONE

DS = "Billyyy/cleaned-mongolian-dataset"


def get(url, timeout=60):
    req = urllib.request.Request(url, headers=HDRS)
    with urllib.request.urlopen(req, timeout=timeout, context=CTX) as r:
        return r.read(4000000).decode("utf-8", "replace")


def api(path, **kw):
    q = "&".join("%s=%s" % (k, urllib.parse.quote(str(v))) for k, v in kw.items())
    return json.loads(get("https://datasets-server.huggingface.co/%s?%s" % (path, q)))


def hr(t):
    print("\n" + "=" * 90)
    print(t)
    print("=" * 90)


hr("1. SIZE AND SHAPE")
size = api("size", dataset=DS)
for c in size.get("size", {}).get("configs", []):
    print("config: %s" % c["config"])
    for s in c.get("splits", []):
        print("   split=%-8s rows=%s bytes=%s" % (s["split"], s["num_rows"], s.get("num_bytes_parquet_files")))

info = api("info", dataset=DS)
feats = info.get("dataset_info", {}).get("features")
print("\ncolumns: %s" % feats)

hr("2. STATISTICS")
try:
    st = api("statistics", dataset=DS, config="default", split="train")
    for col, d in (st.get("statistics") or {}).items():
        print("\n--- %s ---" % col)
        print(json.dumps(d, ensure_ascii=False)[:700])
except Exception as e:  # noqa: BLE001
    print("ERR", e)

hr("3. SAMPLES ACROSS THE WHOLE FILE (not just the start)")
CYR = re.compile(r"[\u0400-\u04FF]")
LAT = re.compile(r"[A-Za-z]")
offsets = [0, 100000, 250000, 500000, 750000, 900000, 997000]
lengths, cyr_ratios, samples = [], [], []
for off in offsets:
    try:
        r = api("rows", dataset=DS, config="default", split="train", offset=off, length=3)
        for row in (r.get("rows") or []):
            t = str(row["row"].get("text") or "")
            if not t.strip():
                continue
            lengths.append(len(t))
            c = len(CYR.findall(t))
            l = len(LAT.findall(t))
            cyr_ratios.append(100.0 * c / max(1, c + l))
            samples.append((off, t))
    except Exception as e:  # noqa: BLE001
        print("  offset %d ERR %s" % (off, type(e).__name__))

if lengths:
    lengths.sort()
    print("char length: min=%d  p25=%d  median=%d  p75=%d  max=%d"
          % (lengths[0], lengths[len(lengths) // 4], lengths[len(lengths) // 2],
             lengths[3 * len(lengths) // 4], lengths[-1]))
    print("mean Cyrillic share: %.1f%%" % (sum(cyr_ratios) / len(cyr_ratios)))
    print("\n%d samples:" % len(samples))
    for off, t in samples:
        print("\n  [offset %-7d] %s" % (off, t[:280].replace("\n", " ")))

hr("4. WHAT KIND OF TEXT IS IT? (keyword fingerprint)")
kinds = {
    "news": r"онөөдөр|мэдээлэв|яам|засгийн газар|сонгууль|ерөнхийлөгч",
    "story/fiction": r"гэж бодлоо|нүд|зүрх|хараад|гэж хэлэв|аймшиг",
    "chat/social": r"сайн уу|баярлалаа|хөөе|за|хмм|аа",
    "advertising": r"хямдрал|урамшуулал|зар|үнэ|худалдаа|бэлэг",
    "religion": r"бурхан|сүм|шашин|залбирал",
    "legal/gov": r"хууль|тушаал|шийдвэр|гэрээ|эрх",
}
counts = {k: 0 for k in kinds}
for _, t in samples:
    low = t.lower()
    for k, pat in kinds.items():
        if re.search(pat, low):
            counts[k] += 1
print("keyword hits out of %d samples:" % len(samples))
for k, v in sorted(counts.items(), key=lambda kv: -kv[1]):
    print("   %-16s %d" % (k, v))

hr("5. IS THERE ANY LABEL COLUMN ANYWHERE?")
try:
    info_all = json.loads(get("https://huggingface.co/api/datasets/" + DS))
    print("tags:", info_all.get("tags"))
    print("cardData:", json.dumps(info_all.get("cardData"), ensure_ascii=False)[:500])
    print("siblings:", [s["rfilename"] for s in (info_all.get("siblings") or [])])
except Exception as e:  # noqa: BLE001
    print("ERR", e)
