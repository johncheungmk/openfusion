from __future__ import annotations

from pathlib import Path

from openfusion.evaluation import extract_answer, is_exact_match, load_jsonl, normalize_answer


def test_answer_normalization_and_regex() -> None:
    assert normalize_answer(" **Paris!** ") == "paris"
    assert extract_answer("Reasoning\nFinal answer: B", r"Final answer:\s*(\w+)") == "B"
    assert is_exact_match("Reasoning\nFinal answer: B", ["B"], r"Final answer:\s*(\w+)")


def test_load_jsonl(tmp_path: Path) -> None:
    path = tmp_path / "eval.jsonl"
    path.write_text(
        '{"id":"one","prompt":"2+2?","reference":"4"}\n',
        encoding="utf-8",
    )
    cases = load_jsonl(path)
    assert cases[0].id == "one"
    assert cases[0].reference == "4"
