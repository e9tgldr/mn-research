"""Can ModernBERT be adapted to Mongolian?

Two things to establish before answering:
 1. What is mmBERT (the backbone already in use) - is it already ModernBERT-shaped?
 2. What is the REAL context limit of the model on disk, vs the 192 used in training?
"""
import glob
import json
import os
import ssl
import urllib.parse
import urllib.request

UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/126.0 Safari/537.36")
CTX = ssl.create_default_context()
CTX.check_hostname = False
CTX.verify_mode = ssl.CERT_NONE


def get(url, timeout=40):
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=timeout, context=CTX) as r:
        return r.read(2000000).decode("utf-8", "replace")


def show_cfg(label, d):
    keys = ["model_type", "architectures", "max_position_embeddings", "vocab_size",
            "hidden_size", "num_hidden_layers", "num_attention_heads",
            "intermediate_size", "local_attention", "global_attn_every_n_layers",
            "global_rope_theta", "local_rope_theta", "rope_theta", "layer_types",
            "_name_or_path"]
    print("\n--- %s ---" % label)
    for k in keys:
        if d.get(k) is not None:
            v = d[k]
            if isinstance(v, list) and len(v) > 6:
                v = "%s ... (%d entries)" % (v[:4], len(v))
            print("   %-30s %s" % (k, v))


print("=" * 92)
print("1. mmBERT-base - the backbone already in use")
print("=" * 92)
try:
    d = json.loads(get("https://huggingface.co/jhu-clsp/mmBERT-base/raw/main/config.json"))
    show_cfg("jhu-clsp/mmBERT-base config", d)
except Exception as e:  # noqa: BLE001
    print("  ERR", e)
try:
    info = json.loads(get("https://huggingface.co/api/models/jhu-clsp/mmBERT-base"))
    print("   downloads %s   likes %s" % (info.get("downloads"), info.get("likes")))
    print("   tags      %s" % (info.get("tags") or [])[:16])
except Exception as e:  # noqa: BLE001
    print("  ERR", e)

print("\n" + "=" * 92)
print("2. The model ON DISK - what is its real context limit?")
print("=" * 92)
for pat in ("C:/Users/hi/Documents/laya-mn/out_*/config.json",
            "C:/Users/hi/Documents/laya-mn/model/**/config.json",
            "C:/Users/hi/Documents/laya-mn/models/**/config.json"):
    for p in glob.glob(pat, recursive=True):
        try:
            d = json.load(open(p, encoding="utf-8"))
        except Exception as e:  # noqa: BLE001
            print("  %s ERR %s" % (p, e))
            continue
        if "encoder" in d:
            print("\n--- %s ---" % p)
            print("   head keys: %s" % list(d.keys()))
            enc = d["encoder"]
            print("   encoder  : %s" % json.dumps(enc, ensure_ascii=False)[:400])
            continue
        show_cfg(p, d)

print("\n" + "=" * 92)
print("3. Any multilingual / non-English ModernBERT already published?")
print("=" * 92)
for q in ("modernbert", "mmbert", "modernbert multilingual"):
    try:
        r = json.loads(get("https://huggingface.co/api/models?search=%s&sort=downloads"
                           "&direction=-1&limit=20" % urllib.parse.quote(q)))
        print("\n[%s]" % q)
        for m in r:
            mid = m["modelId"].lower()
            if "modernbert" in mid or "mmbert" in mid:
                print("   %-58s dl=%-9s %s" % (m["modelId"], m.get("downloads"),
                                               m.get("pipeline_tag") or "-"))
    except Exception as e:  # noqa: BLE001
        print("  ERR", e)

print("\n" + "=" * 92)
print("4. Local training scripts - what max_len is actually being used?")
print("=" * 92)
for f in glob.glob("C:/Users/hi/Documents/laya-mn/*.py"):
    try:
        txt = open(f, encoding="utf-8").read()
    except Exception:  # noqa: BLE001
        continue
    for kw in ("max_len", "max_len=", "max-position", "MAX_LEN"):
        if kw in txt:
            lines = [l.strip() for l in txt.splitlines() if kw in l][:4]
            print("  %-16s %s" % (os.path.basename(f), " | ".join(lines)))
            break
