# Windows Troubleshooting

## PowerShell JSON quoting

For complex JSON, prefer writing a file or using `ConvertTo-Json` rather than inline curl escaping.

## Pytest temp permission errors

Use:

```powershell
pytest -q --basetemp .pytest-tmp
```

## UTF-8 BOM in JSONL

If a JSONL file fails with `Unexpected UTF-8 BOM`, rewrite it with Python or use a recent OpenFusion version that reads UTF-8 with BOM.
