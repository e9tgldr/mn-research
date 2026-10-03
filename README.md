# mn-research

Scratch scripts for Mongolian NLP research — finding and checking published work before committing to an approach.

| Script | Purpose |
|---|---|
| `survey.py` | Broad literature sweep |
| `chase.py` | Follow up specific leads a search surfaced; direct fetches only |
| `modernbert_mn.py` | Check ModernBERT's Mongolian coverage |
| `what_is_jev.py` | Establish what JEV actually is, and how good it really is on Mongolian |
| `inspect_billy.py` | Inspect a specific candidate model |
| `sample.py` | Print samples from a dataset |
| `tasks.py` | Shared helpers |

## Note on the fetching

These scripts set `check_hostname = False` on the SSL context. That is deliberate — several Mongolian government and university sites serve certificates that fail verification — but it does mean the connection is not authenticated. Fine for reading public pages, never for anything carrying credentials.
