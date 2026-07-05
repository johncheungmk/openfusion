# JSONL BOM Errors on Windows

## Symptom

You see an error like:

```text
JSONDecodeError: Unexpected UTF-8 BOM
```

## Cause

Some Windows tools may save text as UTF-8 with a byte-order mark.

## Fix

Recent OpenFusion Lab versions support UTF-8 with BOM.

If you need to repair a file manually:

```powershell
@'
from pathlib import Path

path = Path("local-bench/mini_mmlu_10.jsonl")
text = path.read_text(encoding="utf-8-sig")
path.write_text(text, encoding="utf-8", newline="\n")
'@ | python
```

## Recommendation

Use Python to create JSONL datasets when possible. This avoids hidden encoding issues.
