"""What is a 'jev' / 'kev' / 'von' model? Read the actual model cards and configs."""
import json
import ssl
import urllib.request

UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/126.0 Safari/537.36")
CTX = ssl.create_default_context()
CTX.check_hostname = False
CTX.verify_mode = ssl.CERT_NONE


def get(url, timeout=40):
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=timeout, context=CTX) as r:
        return r.read(3000000).decode("utf-8", "replace")


MODELS = [
    "jaredpalmer/kev-4b",
    "jaredpalmer/kev-0.8b",
    "chaoliangUNSW/Jev-Style-Qwen3.5-2B-Decision-GGUF",
    "com-kotobalabs/open-jev-deberta-v3-large",
    "mobarmg/jev-schema-scorer-deberta-v3-large",
    "wfzyx/von",
    "lostargon/Tiny-Jev",
]

for m in MODELS:
    print("\n" + "=" * 92)
    print(m)
    print("=" * 92)
    try:
        info = json.loads(get("https://huggingface.co/api/models/" + m))
        for k in ("pipeline_tag", "library_name", "downloads", "likes", "lastModified"):
            print("  %-14s %s" % (k, info.get(k)))
        print("  tags          %s" % (info.get("tags") or [])[:14])
        print("  files         %s" % [s["rfilename"] for s in (info.get("siblings") or [])][:14])
    except Exception as e:  # noqa: BLE001
        print("  api ERR", e)
    for f in ("README.md", "config.json"):
        try:
            body = get("https://huggingface.co/%s/raw/main/%s" % (m, f))
            if f.endswith(".json"):
                d = json.loads(body)
                keep = {k: d.get(k) for k in
                        ("model_type", "architectures", "num_labels", "id2label",
                         "vocab_size", "hidden_size", "max_position_embeddings",
                         "num_hidden_layers", "label2id") if d.get(k) is not None}
                print("  config        %s" % json.dumps(keep, ensure_ascii=False)[:800])
            else:
                body = body[:2600]
                print("  --- README ---")
                print("  " + body.replace("\n", "\n  ")[:2600])
        except Exception as e:  # noqa: BLE001
            print("  %s ERR %s" % (f, type(e).__name__))
