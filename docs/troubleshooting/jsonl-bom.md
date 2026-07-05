# JSONL BOM Errors

If you see `JSONDecodeError: Unexpected UTF-8 BOM`, the file was saved as UTF-8 with BOM.

Repair with PowerShell:

```powershell
@'
from pathlib import Path
path = Path("local-bench/mini_mmlu_10.jsonl")
text = path.read_text(encoding="utf-8-sig")
path.write_text(text, encoding="utf-8", newline="\n")
'@ | python
```

Recent OpenFusion versions read UTF-8 BOM files correctly.
